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

#include <optional>

#include "absl/log/check.h"
#include "structmember.h"  // NOLINT // For PyMemberDef
#include "zimt/mos.h"
#include "zimt/zimtohrli.h"

namespace {

struct SpectrogramObject {
  // clang-format off
  PyObject_HEAD
  void *spectrogram;
  // clang-format on
};

void Spectrogram_dealloc(SpectrogramObject* self) {
  if (self) {
    if (self->spectrogram) {
      delete static_cast<zimtohrli::Spectrogram*>(self->spectrogram);
      self->spectrogram = nullptr;
    }
    Py_TYPE(self)->tp_free((PyObject*)self);
  }
}

PyTypeObject SpectrogramType = {
    // clang-format off
    .ob_base = PyVarObject_HEAD_INIT(nullptr, 0)
    .tp_name = "pyohrli.Spectrogram",
    // clang-format on
    .tp_basicsize = sizeof(SpectrogramObject),
    .tp_itemsize = 0,
    .tp_dealloc = (destructor)Spectrogram_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = PyDoc_STR("Python wrapper around C++ zimtohrli::Spectrogram."),
    .tp_new = PyType_GenericNew,
};

struct PyohrliObject {
  // clang-format off
  PyObject_HEAD
  void *zimtohrli;
  // clang-format on
};

int Pyohrli_init(PyohrliObject* self, PyObject* args, PyObject* kwds) {
  try {
    self->zimtohrli = new zimtohrli::Zimtohrli{};
  } catch (const std::bad_alloc&) {
    PyErr_SetNone(PyExc_MemoryError);
    return -1;
  }
  return 0;
}

void Pyohrli_dealloc(PyohrliObject* self) {
  if (self) {
    if (self->zimtohrli) {
      delete static_cast<zimtohrli::Zimtohrli*>(self->zimtohrli);
      self->zimtohrli = nullptr;
    }
    Py_TYPE(self)->tp_free((PyObject*)self);
  }
}

struct BufferDeleter {
  void operator()(Py_buffer* buffer) const { PyBuffer_Release(buffer); }
};

// Plain C++ function to analyze a Python buffer object using Zimtohrli.
//
// Calls to Analyze never need to be cleaned up (with e.g. delete or DECREF)
// afterwards.
//
// If the return value is std::nullopt that means a Python error is set and the
// current operation should be terminated ASAP.
std::optional<zimtohrli::Spectrogram> Analyze(
    const zimtohrli::Zimtohrli& zimtohrli, PyObject* buffer_object) {
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
  return std::optional<zimtohrli::Spectrogram>(zimtohrli.Analyze(
      zimtohrli::Span<const float>(static_cast<float*>(buffer_view.buf),
                                   buffer_view.len / sizeof(float))));
}

PyObject* BadArgument(const std::string& message) {
  PyErr_SetString(PyExc_TypeError, message.c_str());
  return nullptr;
}

PyObject* Pyohrli_distance(PyohrliObject* self, PyObject* const* args,
                           Py_ssize_t nargs) {
  if (nargs != 2) {
    return BadArgument("not exactly 2 arguments provided");
  }
  const zimtohrli::Zimtohrli zimtohrli =
      *static_cast<zimtohrli::Zimtohrli*>(self->zimtohrli);
  std::optional<zimtohrli::Spectrogram> spectrogram_a =
      Analyze(zimtohrli, args[0]);
  if (!spectrogram_a.has_value()) {
    return nullptr;
  }
  std::optional<zimtohrli::Spectrogram> spectrogram_b =
      Analyze(zimtohrli, args[1]);
  if (!spectrogram_b.has_value()) {
    return nullptr;
  }
  return PyFloat_FromDouble(
      zimtohrli.Distance(spectrogram_a.value(), spectrogram_b.value()));
}

PyObject* Pyohrli_analyze(PyohrliObject* self, PyObject* const* args,
                          Py_ssize_t nargs) {
  if (nargs != 1) {
    return BadArgument("not exactly 1 argument provided");
  }
  const zimtohrli::Zimtohrli zimtohrli =
      *static_cast<zimtohrli::Zimtohrli*>(self->zimtohrli);
  const std::optional<zimtohrli::Spectrogram> spectrogram =
      Analyze(zimtohrli, args[0]);
  if (!spectrogram.has_value()) {
    return nullptr;
  }
  return PyBytes_FromStringAndSize(
      reinterpret_cast<const char*>(spectrogram->values.get()),
      spectrogram->size() * sizeof(float));
}

PyObject* Pyohrli_num_rotators(PyohrliObject* self, PyObject* const* args,
                               Py_ssize_t nargs) {
  if (nargs != 0) {
    return BadArgument("not exactly 0 arguments provided");
  }
  return PyLong_FromLong(zimtohrli::kNumRotators);
}

PyObject* Pyohrli_sample_rate(PyohrliObject* self, PyObject* const* args,
                              Py_ssize_t nargs) {
  if (nargs != 0) {
    return BadArgument("not exactly 0 arguments provided");
  }
  return PyLong_FromLong(zimtohrli::kSampleRate);
}

PyMethodDef Pyohrli_methods[] = {
    {"num_rotators", (PyCFunction)Pyohrli_num_rotators, METH_FASTCALL,
     "Returns the number of rotators, i.e. the number of dimensions in a "
     "spectrogram."},
    {"analyze", (PyCFunction)Pyohrli_analyze, METH_FASTCALL,
     "Returns a spectrogram of the provided signal."},
    {"distance", (PyCFunction)Pyohrli_distance, METH_FASTCALL,
     "Returns the distance between the two provided signals."},
    {"sample_rate", (PyCFunction)Pyohrli_sample_rate, METH_FASTCALL,
     "Returns the expected sample rate for analyzed audio."},
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

PyObject* MOSFromZimtohrli(PyohrliObject* self, PyObject* const* args,
                           Py_ssize_t nargs) {
  if (nargs != 1) {
    return BadArgument("not exactly 1 argument provided");
  }
  return PyFloat_FromDouble(
      zimtohrli::MOSFromZimtohrli(PyFloat_AsDouble(args[0])));
}

static PyMethodDef PyohrliModuleMethods[] = {
    {"MOSFromZimtohrli", (PyCFunction)MOSFromZimtohrli, METH_FASTCALL,
     "Returns an approximate mean opinion score based on the provided "
     "Zimtohrli distance."},
    {NULL, NULL, 0, NULL},
};

PyModuleDef PyohrliModule = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "pyohrli",
    .m_doc = "Python wrapper around the C++ zimtohrli library.",
    .m_size = -1,
    .m_methods = PyohrliModuleMethods,
};

PyMODINIT_FUNC PyInit__pyohrli(void) {
  PyObject* m = PyModule_Create(&PyohrliModule);
  if (m == nullptr) return nullptr;

  if (PyType_Ready(&SpectrogramType) < 0) {
    Py_DECREF(m);
    return nullptr;
  }
  if (PyModule_AddObjectRef(m, "Spectrogram", (PyObject*)&SpectrogramType) <
      0) {
    Py_DECREF(m);
    return nullptr;
  }

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
