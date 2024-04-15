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
"""Tests for google3.third_party.zimtohrli.python.cam."""

import numpy as np
from google3.testing.pybase import googletest
from google3.testing.pybase import parameterized
from google3.third_party.zimtohrli.python import cam
from google3.third_party.zimtohrli.python import signal


class CamTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.cam = cam.Cam()

  @parameterized.parameters(
      (1, 25.995276471698844),
      (10, 442.29956714831576),
      (20, 1739.4974583218288),
      (30, 5543.983136917903),
  )
  def test_hz_from_cam(self, cam, hz):
    self.assertAlmostEqual(self.cam.hz_from_cam(cam), hz)

  @parameterized.parameters(
      (1, 0.04052563),
      (10, 0.3975194),
      (100, 3.3695753),
      (1000, 15.621448),
      (10000, 35.316578),
  )
  def test_cam_from_hz(self, hz, cam):
    self.assertAlmostEqual(self.cam.cam_from_hz(hz), cam)

  def test_channel_filter(self):
    fs = 48000
    t = 1.0

    def steps(f):
      return np.sin(np.linspace(0, np.pi * 2 * fs * t * f, int(fs * t)))

    signal_freqs = [200, 250, 2000, 2500, 10000, 15000]
    signal_components = [
        10 ** (-10 / freq) * steps(freq) for freq in signal_freqs
    ]
    sig = signal.Signal(
        sample_rate=fs, samples=np.sum(signal_components, axis=0)
    )
    channels = cam.Cam().channel_filter(sig)
    powers = np.var(channels.samples, axis=-1)
    peak_freqs = np.mean(channels.freqs, axis=-1)[
        np.where(powers > np.mean(powers))
    ]
    for freq in signal_freqs:
      self.assertLess((np.min(np.abs(freq - peak_freqs)) / freq), 0.05)
    for freq in peak_freqs:
      self.assertLess(
          (np.min(np.abs(freq - np.asarray(signal_freqs)) / freq)), 0.05
      )

  @parameterized.parameters(
      (80, 600),
      (50, 200),
      (70, 1000),
      (90, 8000),
      (30, 400),
  )
  def test_filter_energy_db(self, db, hz):
    fs = 48000
    t = 1.0
    full_scale_sine_db = 90
    samples = np.sin(np.linspace(0, 2 * np.pi * hz * t, int(fs * t))) * 10 ** (
        (db - full_scale_sine_db) / 20
    )
    sig = signal.Signal(sample_rate=fs, samples=samples)
    channels = cam.Cam().channel_filter(sig)
    probe_channel_idx = -1
    for idx in range(np.asarray(channels.freqs).shape[0]):
      if hz >= channels.freqs[idx][0] and hz < channels.freqs[idx][1]:
        probe_channel_idx = idx
    energy = channels.energy()
    energy_db = energy.to_db()
    # TODO(zond): Figure out why this needs to have 7dB tolerance - should be 3
    # considering the passband ripple is 3.
    np.testing.assert_allclose(
        np.max(energy_db.samples[probe_channel_idx, -1]), db, atol=7
    )


if __name__ == "__main__":
  googletest.main()
