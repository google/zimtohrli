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
#include "structmember.h"  // NOLINT // For PyMemberDef
#include "zimt/cam.h"
#include "zimt/mos.h"
#include "zimt/zimtohrli.h"

namespace {

struct AnalysisObject {
  // clang-format off
  PyObject_HEAD
  zimtohrli::Analysis *analysis;
  // clang-format on
};

void Analysis_dealloc(AnalysisObject* self) {
  if (self->analysis) {
    delete self->analysis;
    self->analysis = nullptr;
  }
  Py_TYPE(self)->tp_free((PyObject*)self);
}

PyTypeObject AnalysisType = {
    // clang-format off
    .ob_base = PyVarObject_HEAD_INIT(nullptr, 0)
    .tp_name = "pyohrli.Analysis",
    // clang-format on
    .tp_basicsize = sizeof(AnalysisObject),
    .tp_itemsize = 0,
    .tp_dealloc = (destructor)Analysis_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = PyDoc_STR("Python wrapper around C++ zimtohrli::Analysis."),
    .tp_new = PyType_GenericNew,
};

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

// Plain C++ function to analyze a Python buffer object using Zimtohrli.
//
// Calls to Analyze never need to be cleaned up (with e.g. delete or DECREF)
// afterwards.
//
// If the return value is std::nullopt that means a Python error is set and the
// current operation should be terminated ASAP.
std::optional<zimtohrli::Analysis> Analyze(
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
  hwy::AlignedNDArray<float, 1> signal_array({buffer_view.len / sizeof(float)});
  hwy::CopyBytes(buffer_view.buf, signal_array.data(), buffer_view.len);
  return std::optional<zimtohrli::Analysis>{
      zimtohrli.Analyze(signal_array[{}])};
}

PyObject* BadArgument(const std::string& message) {
  PyErr_SetString(PyExc_TypeError, message.c_str());
  return nullptr;
}

PyObject* Pyohrli_analyze(PyohrliObject* self, PyObject* const* args,
                          Py_ssize_t nargs) {
  if (nargs != 1) {
    return BadArgument("not exactly 1 argument provided");
  }
  std::optional<zimtohrli::Analysis> analysis =
      Analyze(*self->zimtohrli, args[0]);
  if (!analysis.has_value()) {
    return nullptr;
  }
  AnalysisObject* result = PyObject_New(AnalysisObject, &AnalysisType);
  if (result == nullptr) {
    return nullptr;
  }
  try {
    result->analysis = new zimtohrli::Analysis{
        .energy_channels_db = std::move(analysis->energy_channels_db),
        .partial_energy_channels_db =
            std::move(analysis->partial_energy_channels_db),
        .spectrogram = std::move(analysis->spectrogram)};
    return (PyObject*)result;
  } catch (const std::bad_alloc&) {
    // Technically, this object should be deleted with PyObject_Del, but
    // XDECREF includes a null check which we want anyway.
    Py_XDECREF((PyObject*)result);
    return PyErr_NoMemory();
  }
}

// Plain C++ function to compute distance between two zimtohrli::Analysis.
//
// Calls to Distance never need to be cleaned up (with e.g. delete or DECREF)
// afterwards.
PyObject* Distance(const zimtohrli::Zimtohrli& zimtohrli,
                   const zimtohrli::Analysis& analysis_a,
                   const zimtohrli::Analysis& analysis_b) {
  const zimtohrli::Distance distance =
      zimtohrli.Distance(false, analysis_a.spectrogram, analysis_b.spectrogram);
  return PyFloat_FromDouble(distance.value);
}

PyObject* Pyohrli_analysis_distance(PyohrliObject* self, PyObject* const* args,
                                    Py_ssize_t nargs) {
  if (nargs != 2) {
    return BadArgument("not exactly 2 arguments provided");
  }
  if (!Py_IS_TYPE(args[0], &AnalysisType)) {
    return BadArgument("argument 0 is not an Analysis instance");
  }
  if (!Py_IS_TYPE(args[1], &AnalysisType)) {
    return BadArgument("argument 1 is not an Analysis instance");
  }
  return Distance(*self->zimtohrli, *((AnalysisObject*)args[0])->analysis,
                  *((AnalysisObject*)args[1])->analysis);
}

PyObject* Pyohrli_distance(PyohrliObject* self, PyObject* const* args,
                           Py_ssize_t nargs) {
  if (nargs != 2) {
    return BadArgument("not exactly 2 arguments provided");
  }
  const std::optional<zimtohrli::Analysis> analysis_a =
      Analyze(*self->zimtohrli, args[0]);
  if (!analysis_a.has_value()) {
    return nullptr;
  }
  const std::optional<zimtohrli::Analysis> analysis_b =
      Analyze(*self->zimtohrli, args[1]);
  if (!analysis_b.has_value()) {
    return nullptr;
  }
  return Distance(*self->zimtohrli, analysis_a.value(), analysis_b.value());
}

PyMethodDef Pyohrli_methods[] = {
    {"analyze", (PyCFunction)Pyohrli_analyze, METH_FASTCALL,
     "Returns an analysis of the provided signal."},
    {"analysis_distance", (PyCFunction)Pyohrli_analysis_distance, METH_FASTCALL,
     "Returns the distance between the two provided analyses."},
    {"distance", (PyCFunction)Pyohrli_distance, METH_FASTCALL,
     "Returns the distance between the two provided signals."},
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

  if (PyType_Ready(&AnalysisType) < 0) {
    Py_DECREF(m);
    return nullptr;
  }
  if (PyModule_AddObjectRef(m, "Analysis", (PyObject*)&AnalysisType) < 0) {
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
