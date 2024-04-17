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

"""Tests for zimtohrli.python.tf.zimtohrli.loudness."""

import numpy as np
import unittest
import parameterized
import loudness


class LoudnessTest(unittest.TestCase):
    loudness_scale = loudness.Loudness()

    @parameterized.parameterize(
        dict(hz=250.0, phons=20.0, spl=31.643906),
        dict(hz=250.0, phons=40.0, spl=50.082615),
        dict(hz=250.0, phons=60.0, spl=67.254036),
        dict(hz=1000.0, phons=20.0, spl=20.237755),
        dict(hz=1000.0, phons=40.0, spl=40.344223),
        dict(hz=1000.0, phons=60.0, spl=60.403873),
        dict(hz=1500.0, phons=20.0, spl=22.166199),
        dict(hz=1500.0, phons=40.0, spl=43.04581),
        dict(hz=1500.0, phons=60.0, spl=63.532944),
    )
    def testSPLFromPhons(self, hz: float, phons: float, spl: float):
        np.testing.assert_allclose(
            self.loudness_scale.spl_from_phons(phons=phons, freq=hz),
            spl,
            rtol=1e-5,
            atol=0,
        )

    @parameterized.parameterize(
        dict(hz=250.0, phons=21.403091, spl=33.0),
        dict(hz=250.0, phons=41.081264, spl=51.0),
        dict(hz=250.0, phons=59.735756, spl=67.0),
        dict(hz=1000.0, phons=19.789963, spl=20.0),
        dict(hz=1000.0, phons=39.687717, spl=40.0),
        dict(hz=1000.0, phons=59.629295, spl=60.0),
        dict(hz=1500.0, phons=19.868698, spl=22.0),
        dict(hz=1500.0, phons=39.01555, spl=42.0),
        dict(hz=1500.0, phons=59.50945, spl=63.0),
    )
    def testPhonsFromSPL(self, hz: float, phons: float, spl: float):
        np.testing.assert_allclose(
            self.loudness_scale.phons_from_spl(spl=spl, freq=hz),
            phons,
            rtol=1e-5,
            atol=0,
        )


if __name__ == "__main__":
    unittest.main()
