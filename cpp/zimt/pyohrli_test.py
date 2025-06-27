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

    def test_num_rotators(self):
      self.assertEqual(128, pyohrli.Pyohrli().num_rotators)

    def test_sample_rate(self):
      self.assertEqual(48000, pyohrli.Pyohrli().sample_rate)

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
        metric = pyohrli.Pyohrli()
        signal_a = np.sin(np.linspace(0.0, np.pi * 2 * a_hz, int(sample_rate)))
        signal_b = np.sin(np.linspace(0.0, np.pi * 2 * b_hz, int(sample_rate)))
        distance = metric.distance(signal_a, signal_b)
        self.assertLess(abs(distance - distance), 1e-3)

    @parameterize(
        dict(zimtohrli_distance=0.0, mos=5.0),
        dict(zimtohrli_distance=0.001, mos=4.800886631011963),
        dict(zimtohrli_distance=0.01, mos=3.4005415439605713),
        dict(zimtohrli_distance=0.02, mos=2.4406499862670898),
        dict(zimtohrli_distance=0.03, mos=1.8645849227905273),
        dict(zimtohrli_distance=0.04, mos=1.5188679695129395),
    )
    def test_mos_from_zimtohrli(self, zimtohrli_distance: float, mos: float):
        self.assertAlmostEqual(
            mos, pyohrli.mos_from_zimtohrli(zimtohrli_distance), places=3
        )


if __name__ == "__main__":
    unittest.main()
