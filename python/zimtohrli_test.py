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
"""Tests for google3.third_party.zimtohrli.python.zimtohrli."""

from google3.testing.pybase import googletest
from google3.third_party.zimtohrli.python import signal
from google3.third_party.zimtohrli.python import zimtohrli


class ZimtohrliTest(googletest.TestCase):

  def test_zimtohrli_spectrogram_and_distance(self):
    sample_rate = 48000.0
    signal_a = np.zeros((int(sample_rate) // 100,))
    signal_a[0] = 1.0
    signal_b = np.zeros((int(sample_rate) // 100,))
    signal_b[0:1] = 0.9
    sound_a = signal.Signal(sample_rate=sample_rate, samples=signal_a)
    sound_b = signal.Signal(sample_rate=sample_rate, samples=signal_b)
    spectrogram_a = zimtohrli.spectrogram(sound_a)
    spectrogram_b = zimtohrli.spectrogram(sound_b)
    distance = zimtohrli.distance(spectrogram_a, spectrogram_b)
    self.assertGreater(distance, 0)


if __name__ == "__main__":
  googletest.main()
