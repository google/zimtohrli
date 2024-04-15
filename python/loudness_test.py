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
from google3.testing.pybase import googletest
from google3.testing.pybase import parameterized
from google3.third_party.zimtohrli.python import loudness


class LoudnessTest(parameterized.TestCase):
  loudness_scale = loudness.Loudness()

  @parameterized.parameters(
      (250.0, 20.0, 32.21215),
      (250.0, 40.0, 50.71194),
      (250.0, 60.0, 67.82744),
      (1000.0, 20.0, 19.89019),
      (1000.0, 40.0, 40.31903),
      (1000.0, 60.0, 60.705994),
      (1500.0, 20.0, 20.86898),
      (1500.0, 40.0, 41.29622),
      (1500.0, 60.0, 61.80821),
  )
  def testSPLFromPhons(self, hz: float, phons: float, spl: float):
    np.testing.assert_allclose(
        self.loudness_scale.spl_from_phons(phons=phons, freq=hz),
        spl,
        rtol=1e-5,
        atol=0,
    )

  @parameterized.parameters(
      (250.0, 20.815575, 33.0),
      (250.0, 40.362747, 51.0),
      (250.0, 59.05414, 67.0),
      (1000.0, 20.132996, 20.0),
      (1000.0, 39.717552, 40.0),
      (1000.0, 59.33892, 60.0),
      (1500.0, 21.137794, 22.0),
      (1500.0, 40.717434, 42.0),
      (1500.0, 61.1927, 63.0),
  )
  def testPhonsFromSPL(self, hz: float, phons: float, spl: float):
    np.testing.assert_allclose(
        self.loudness_scale.phons_from_spl(spl=spl, freq=hz),
        phons,
        rtol=1e-5,
        atol=0,
    )


if __name__ == '__main__':
  googletest.main()
