// Copyright 2024 The Zimtohrli Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <utility>
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "absl/log/check.h"
#include "hwy/aligned_allocator.h"
#include "hwy/base.h"
#include "structmember.h"  // NOLINT // For PyMemberDef
#include "zimt/cam.h"
#include "zimt/mos.h"
#include "zimt/zimtohrli.h"

namespace {

struct PyohrliObject {
  // clang-format off
  PyObject_HEAD
  zimtohrli::Zimtohrli *zimtohrli;
  // clang-format on
};

int Pyohrli_init(PyohrliObject* self, PyObject* args, PyObject* kwds) {
  float sample_rate;
  const char* keywords[] = {"sample_rate", nullptr};
  if (!PyArg_ParseTupleAndKeywords(
          args, kwds, "f", const_cast<char**>(keywords), &sample_rate)) {
    PyErr_SetString(PyExc_TypeError, "unable to parse sample_rate as float");
    return -1;
  }
  try {
    const zimtohrli::Cam default_cam;
    self->zimtohrli = new zimtohrli::Zimtohrli{
        .cam_filterbank =
            zimtohrli::Cam{
                .high_threshold_hz =
                    std::min(sample_rate * 0.5f, default_cam.high_threshold_hz),
            }
                .CreateFilterbank(sample_rate)};
  } catch (const std::bad_alloc&) {
    PyErr_SetNone(PyExc_MemoryError);
    return -1;
  }
  return 0;
}

void Pyohrli_dealloc(PyohrliObject* self) {
  if (self) {
    if (self->zimtohrli) {
      delete self->zimtohrli;
      self->zimtohrli = nullptr;
    }
    Py_TYPE(self)->tp_free((PyObject*)self);
  }
}

struct BufferDeleter {
  void operator()(Py_buffer* buffer) const { PyBuffer_Release(buffer); }
};

PyObject* BadArgument(const std::string& message) {
  PyErr_SetString(PyExc_TypeError, message.c_str());
  return nullptr;
}

// Plain C++ function to copy a Python buffer object to a hwy::AlignedNDArray.
//
// Calls to CopyBuffer never need to be cleaned up (with e.g. delete or DECREF)
// afterwards.
//
// If the return value is std::nullopt that means a Python error is set and the
// current operation should be terminated ASAP.
std::optional<hwy::AlignedNDArray<float, 1>> CopyBuffer(
    PyObject* buffer_object) {
  Py_buffer buffer_view;
  if (PyObject_GetBuffer(buffer_object, &buffer_view, PyBUF_C_CONTIGUOUS)) {
    PyErr_SetString(PyExc_TypeError, "object is not buffer");
    return std::nullopt;
  }
  std::unique_ptr<Py_buffer, BufferDeleter> buffer_view_deleter(&buffer_view);
  if (buffer_view.itemsize != sizeof(float)) {
    PyErr_SetString(PyExc_TypeError, "buffer does not contain floats");
    return std::nullopt;
  }
  if (buffer_view.ndim != 1) {
    PyErr_SetString(PyExc_TypeError, "buffer has more than 1 axis");
    return std::nullopt;
  }
  hwy::AlignedNDArray<float, 1> signal_array({buffer_view.len / sizeof(float)});
  hwy::CopyBytes(buffer_view.buf, signal_array.data(), buffer_view.len);
  return signal_array;
}

PyObject* Pyohrli_distance(PyohrliObject* self, PyObject* const* args,
                           Py_ssize_t nargs) {
  if (nargs != 2) {
    return BadArgument("not exactly 2 arguments provided");
  }
  const std::optional<hwy::AlignedNDArray<float, 1>> signal_a =
      CopyBuffer(args[0]);
  if (!signal_a.has_value()) {
    return nullptr;
  }
  const std::optional<hwy::AlignedNDArray<float, 1>> signal_b =
      CopyBuffer(args[1]);
  if (!signal_b.has_value()) {
    return nullptr;
  }
  const hwy::AlignedNDArray<float, 2> spectrogram_a =
      self->zimtohrli->StreamingSpectrogram((*signal_a)[{}]);
  const hwy::AlignedNDArray<float, 2> spectrogram_b =
      self->zimtohrli->StreamingSpectrogram((*signal_b)[{}]);
  const zimtohrli::Distance distance =
      self->zimtohrli->Distance(false, spectrogram_a, spectrogram_b);
  return PyFloat_FromDouble(distance.value);
}

PyObject* Pyohrli_mos_from_zimtohrli(PyohrliObject* self, PyObject* const* args,
                                     Py_ssize_t nargs) {
  if (nargs != 1) {
    return BadArgument("not exactly 1 argument provided");
  }
  return PyFloat_FromDouble(
      self->zimtohrli->mos_mapper.Map(PyFloat_AsDouble(args[0])));
}

PyMethodDef Pyohrli_methods[] = {
    {"distance", (PyCFunction)Pyohrli_distance, METH_FASTCALL,
     "Returns the distance between the two provided signals."},
    {"mos_from_zimtohrli", (PyCFunction)Pyohrli_mos_from_zimtohrli,
     METH_FASTCALL,
     "Returns an approximate mean opinion score based on the provided "
     "Zimtohrli distance."},
    {nullptr} /* Sentinel */
};

PyTypeObject PyohrliType = {
    // clang-format off
    .ob_base = PyVarObject_HEAD_INIT(nullptr, 0)
    .tp_name = "pyohrli.Pyohrli",
    // clang-format on
    .tp_basicsize = sizeof(PyohrliObject),
    .tp_itemsize = 0,
    .tp_dealloc = (destructor)Pyohrli_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc =
        PyDoc_STR("Python wrapper around the C++ zimtohrli::Zimtohrli type."),
    .tp_methods = Pyohrli_methods,
    .tp_init = (initproc)Pyohrli_init,
    .tp_new = PyType_GenericNew,
};

PyModuleDef PyohrliModule = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "pyohrli",
    .m_doc = "Python wrapper around the C++ zimtohrli library.",
    .m_size = -1,
};

PyMODINIT_FUNC PyInit__pyohrli(void) {
  PyObject* m = PyModule_Create(&PyohrliModule);
  if (m == nullptr) return nullptr;

  if (PyType_Ready(&PyohrliType) < 0) {
    Py_DECREF(m);
    return nullptr;
  };
  if (PyModule_AddObjectRef(m, "Pyohrli", (PyObject*)&PyohrliType) < 0) {
    Py_DECREF(m);
    return nullptr;
  }

  return m;
}

}  // namespace
