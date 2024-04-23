# Copyright 2022 The Zimtohrli Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for google3.third_party.zimtohrli.cpp.python.pyohrli."""

import numpy as np

import unittest
import pyohrli
import functools


def parameterize(*kwargs):
    def decorator(func):
        @functools.wraps(func)
        def call_with_parameters(self, **inner_kwargs):
            for kwarg in kwargs:
                func(self, **kwarg)

        return call_with_parameters

    return decorator


class PyohrliTest(unittest.TestCase):

    def test_getters_setters(self):
        metric = pyohrli.Pyohrli(sample_rate=48000.0)

        time_norm_order = metric.time_norm_order
        self.assertEqual(time_norm_order, 4)
        metric.time_norm_order *= 2
        self.assertEqual(metric.time_norm_order, time_norm_order * 2)

        freq_norm_order = metric.freq_norm_order
        self.assertEqual(freq_norm_order, 4)
        metric.freq_norm_order *= 2
        self.assertEqual(metric.freq_norm_order, freq_norm_order * 2)

        full_scale_sine_db = metric.full_scale_sine_db
        self.assertEqual(full_scale_sine_db, 80)
        metric.full_scale_sine_db *= 2
        self.assertEqual(metric.full_scale_sine_db, full_scale_sine_db * 2)

        unwarp_window = metric.unwarp_window
        self.assertEqual(unwarp_window, 2)
        metric.unwarp_window *= 2
        self.assertEqual(metric.unwarp_window, unwarp_window * 2)

    @parameterize(
        dict(
            testcase_name="zero",
            a_hz=5000.0,
            b_hz=5000.0,
            distance=0,
        ),
        dict(
            testcase_name="small",
            a_hz=5000.0,
            b_hz=5010.0,
            distance=1.473954677581787,
        ),
        dict(
            testcase_name="large",
            a_hz=5000.0,
            b_hz=10000.0,
            distance=52.45160675048828,
        ),
    )
    def test_distance(self, **kwargs):
        sample_rate = 48000.0
        metric = pyohrli.Pyohrli(sample_rate)
        signal_a = np.sin(
            np.linspace(0.0, np.pi * 2 * kwargs["a_hz"], int(sample_rate))
        )
        analysis_a = metric.analyze(signal_a)
        signal_b = np.sin(
            np.linspace(0.0, np.pi * 2 * kwargs["b_hz"], int(sample_rate))
        )
        analysis_b = metric.analyze(signal_b)
        analysis_distance = metric.analysis_distance(analysis_a, analysis_b)
        self.assertLess(abs(analysis_distance - kwargs["distance"]), 1e-4)
        distance = metric.distance(signal_a, signal_b)
        self.assertLess(abs(distance - kwargs["distance"]), 1e-4)

    def test_nyquist_threshold(self):
        sample_rate = 12000.0
        frequency_resolution = 4.0
        metric = pyohrli.Pyohrli(sample_rate)
        signal = np.sin(np.linspace(0.0, np.pi * 2 * 440.0, int(sample_rate)))
        # This would crash the program if pyohrli.cc didn't limit the upper
        # threshold to half the sample rate.
        metric.analyze(signal)


if __name__ == "__main__":
    unittest.main()
