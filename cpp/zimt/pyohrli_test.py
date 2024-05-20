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

    @parameterize(
        dict(
            a_hz=5000.0,
            b_hz=5000.0,
            distance=0,
        ),
        dict(
            a_hz=5000.0,
            b_hz=5010.0,
            distance=3.737211227416992e-05,
        ),
        dict(
            a_hz=5000.0,
            b_hz=10000.0,
            distance=0.3206554651260376,
        ),
    )
    def test_distance(self, a_hz: float, b_hz: float, distance: float):
        sample_rate = 48000.0
        metric = pyohrli.Pyohrli(sample_rate)
        signal_a = np.sin(np.linspace(0.0, np.pi * 2 * a_hz, int(sample_rate)))
        analysis_a = metric.analyze(signal_a)
        signal_b = np.sin(np.linspace(0.0, np.pi * 2 * b_hz, int(sample_rate)))
        analysis_b = metric.analyze(signal_b)
        analysis_distance = metric.analysis_distance(analysis_a, analysis_b)
        self.assertLess(abs(analysis_distance - distance), 1e-3)
        distance = metric.distance(signal_a, signal_b)
        self.assertLess(abs(distance - distance), 1e-3)

    def test_nyquist_threshold(self):
        sample_rate = 12000.0
        metric = pyohrli.Pyohrli(sample_rate)
        signal = np.sin(np.linspace(0.0, np.pi * 2 * 440.0, int(sample_rate)))
        # This would crash the program if pyohrli.cc didn't limit the upper
        # threshold to half the sample rate.
        metric.analyze(signal)

    @parameterize(
        dict(zimtohrli_distance=0.0, mos=5.0),
        dict(zimtohrli_distance=0.1, mos=3.9802114963531494),
        dict(zimtohrli_distance=0.5, mos=1.9183233976364136),
        dict(zimtohrli_distance=0.7, mos=1.5097649097442627),
        dict(zimtohrli_distance=1.0, mos=1.210829496383667),
    )
    def test_mos_from_zimtohrli(self, zimtohrli_distance: float, mos: float):
        self.assertAlmostEqual(
            mos, pyohrli.mos_from_zimtohrli(zimtohrli_distance), places=3
        )


if __name__ == "__main__":
    unittest.main()
