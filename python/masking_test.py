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
"""Tests for google3.third_party.zimtohrli.python.masking."""

import dataclasses
import jax
import numpy as np
from google3.testing.pybase import googletest
from google3.testing.pybase import parameterized
from google3.third_party.zimtohrli.python import cam
from google3.third_party.zimtohrli.python import masking
from google3.third_party.zimtohrli.python import signal


class MaskingTest(parameterized.TestCase):

  @dataclasses.dataclass(frozen=True)
  class Params:
    masker_db_and_hz: tuple[int, int]
    probe_db_and_hz: tuple[int, int]
    want_masked_amount_range: tuple[float, float]

  @parameterized.parameters(
      Params((80, 600), (70, 700), (2, 3)),
      Params((80, 700), (70, 600), (0, 1)),
      Params((75, 600), (80, 700), (-100, 0)),
      Params((80, 600), (60, 700), (20, 25)),
      Params((80, 600), (72, 700), (0.5, 1.5)),
      Params((78, 600), (70, 700), (0.5, 1.5)),
  )
  def test_partial_masking(self, params):
    m = masking.Masking()
    chans = signal.Channels(
        sample_rate=48000,
        freqs=np.asarray([
            [params.masker_db_and_hz[1], params.masker_db_and_hz[1] + 1],
            [params.probe_db_and_hz[1], params.probe_db_and_hz[1] + 1],
        ]),
        samples=np.asarray(
            [[params.masker_db_and_hz[0]], [params.probe_db_and_hz[0]]]
        ),
    )
    self.assertBetween(
        chans.samples[1, -1] - m.partial_loudness(chans).samples[1, -1],
        params.want_masked_amount_range[0],
        params.want_masked_amount_range[1],
    )

  @parameterized.parameters(
      Params((85, 600), (70, 700), (4, 6)),
      Params((80, 700), (73, 600), (0, 1)),
      Params((75, 600), (80, 700), (-100, 0)),
      Params((80, 600), (60, 700), (10, 15)),
      Params((85, 600), (72, 700), (3, 4)),
      Params((83, 600), (70, 700), (3, 4)),
  )
  def test_channel_filtered_partial_masking(self, params):
    fs = 48000
    t = 1
    full_scale_sine_db = 90
    masker_samples = np.sin(
        np.linspace(0, 2 * np.pi * params.masker_db_and_hz[1] * t, int(fs * t))
    ) * 10 ** ((params.masker_db_and_hz[0] - full_scale_sine_db) / 20)
    probe_samples = np.sin(
        np.linspace(0, 2 * np.pi * params.probe_db_and_hz[1] * t, int(fs * t))
    ) * 10 ** ((params.probe_db_and_hz[0] - full_scale_sine_db) / 20)
    sig = signal.Signal(sample_rate=fs, samples=masker_samples + probe_samples)
    channels = cam.Cam().channel_filter(sig)
    probe_channel_idx = -1
    for idx in range(np.asarray(channels.freqs).shape[0]):
      if (
          params.probe_db_and_hz[1] >= channels.freqs[idx][0]
          and params.probe_db_and_hz[1] < channels.freqs[idx][1]
      ):
        probe_channel_idx = idx
    energy = channels.energy()
    energy_db = energy.to_db()
    m = masking.Masking()
    unmasked_probe_energy_db = energy_db.samples[probe_channel_idx, -1]
    partial_loudness_db = m.partial_loudness(energy_db)
    masked_probe_energy_db = partial_loudness_db.samples[
        probe_channel_idx,
        -1,
    ]
    masked_db = unmasked_probe_energy_db - masked_probe_energy_db
    self.assertBetween(
        masked_db,
        params.want_masked_amount_range[0],
        params.want_masked_amount_range[1],
    )


if __name__ == '__main__':
  googletest.main()
