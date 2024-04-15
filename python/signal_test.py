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
"""Tests for google3.third_party.zimtohrli.python.signal."""

import numpy as np
import scipy
from google3.testing.pybase import googletest
from google3.third_party.zimtohrli.python import cam
from google3.third_party.zimtohrli.python import signal


def _create_chirp() -> signal.Signal:
  sample_rate = 48000.0
  t = np.linspace(0.0, 1.0, int(sample_rate))
  f0 = 1
  f1 = sample_rate / 2
  t1 = 1
  return signal.Signal(
      sample_rate=sample_rate,
      samples=np.asarray(
          scipy.signal.chirp(t, f0, t1, f1, method="logarithmic", phi=90)
      ),
  )


class SignalTest(googletest.TestCase):

  def test_energy(self):
    chirp = _create_chirp()
    chirp_channels = cam.Cam().channel_filter(chirp)
    new_sample_rate = 100
    got_chirp_energy = chirp_channels.energy(
        out_sample_rate=new_sample_rate
    ).samples
    channel_samples = np.asarray(chirp_channels.samples)
    want_chirp_energy = (
        channel_samples.reshape(
            channel_samples.shape[0],
            int(channel_samples.shape[1] * 100 // chirp.sample_rate),
            -1,
        )
        ** 2
    ).mean(axis=-1)
    np.testing.assert_allclose(got_chirp_energy, want_chirp_energy, atol=0.02)


if __name__ == "__main__":
  googletest.main()
