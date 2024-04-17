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
import numpy as np
import unittest
import parameterized
import cam
import masking
import audio_signal


class MaskingTest(unittest.TestCase):

    @dataclasses.dataclass(frozen=True)
    class Params:
        masker_db_and_hz: tuple[int, int]
        probe_db_and_hz: tuple[int, int]
        want_masked_amount_range: tuple[float, float]

    @parameterized.parameterize(
        dict(
            masker_db_and_hz=(80, 600),
            probe_db_and_hz=(70, 700),
            want_masked_amount_range=(2, 3),
        ),
        dict(
            masker_db_and_hz=(80, 700),
            probe_db_and_hz=(70, 600),
            want_masked_amount_range=(0, 1),
        ),
        dict(
            masker_db_and_hz=(75, 600),
            probe_db_and_hz=(80, 700),
            want_masked_amount_range=(-100, 0),
        ),
        dict(
            masker_db_and_hz=(80, 600),
            probe_db_and_hz=(60, 700),
            want_masked_amount_range=(20, 25),
        ),
        dict(
            masker_db_and_hz=(80, 600),
            probe_db_and_hz=(72, 700),
            want_masked_amount_range=(0.5, 1.5),
        ),
        dict(
            masker_db_and_hz=(78, 600),
            probe_db_and_hz=(70, 700),
            want_masked_amount_range=(0.5, 1.5),
        ),
    )
    def test_partial_masking(
        self,
        masker_db_and_hz: tuple[float, float],
        probe_db_and_hz: tuple[float, float],
        want_masked_amount_range: tuple[float, float],
    ):
        m = masking.Masking()
        chans = audio_signal.Channels(
            sample_rate=48000,
            freqs=np.asarray(
                [
                    [masker_db_and_hz[1], masker_db_and_hz[1] + 1],
                    [probe_db_and_hz[1], probe_db_and_hz[1] + 1],
                ]
            ),
            samples=np.asarray([[masker_db_and_hz[0]], [probe_db_and_hz[0]]]),
        )
        self.assertGreaterEqual(
            chans.samples[1, -1] - m.partial_loudness(chans).samples[1, -1],
            want_masked_amount_range[0],
        )
        self.assertLessEqual(
            chans.samples[1, -1] - m.partial_loudness(chans).samples[1, -1],
            want_masked_amount_range[1],
        )

    @parameterized.parameterize(
        dict(
            masker_db_and_hz=(85, 600),
            probe_db_and_hz=(70, 700),
            want_masked_amount_range=(4, 6),
        ),
        dict(
            masker_db_and_hz=(80, 700),
            probe_db_and_hz=(73, 600),
            want_masked_amount_range=(0, 1),
        ),
        dict(
            masker_db_and_hz=(75, 600),
            probe_db_and_hz=(80, 700),
            want_masked_amount_range=(-100, 0),
        ),
        dict(
            masker_db_and_hz=(80, 600),
            probe_db_and_hz=(60, 700),
            want_masked_amount_range=(10, 15),
        ),
        dict(
            masker_db_and_hz=(85, 600),
            probe_db_and_hz=(72, 700),
            want_masked_amount_range=(3, 4),
        ),
        dict(
            masker_db_and_hz=(83, 600),
            probe_db_and_hz=(70, 700),
            want_masked_amount_range=(3, 4),
        ),
    )
    def test_channel_filtered_partial_masking(
        self,
        masker_db_and_hz: tuple[float, float],
        probe_db_and_hz: tuple[float, float],
        want_masked_amount_range: tuple[float, float],
    ):
        fs = 48000
        t = 1
        full_scale_sine_db = 90
        masker_samples = np.sin(
            np.linspace(0, 2 * np.pi * masker_db_and_hz[1] * t, int(fs * t))
        ) * 10 ** ((masker_db_and_hz[0] - full_scale_sine_db) / 20)
        probe_samples = np.sin(
            np.linspace(0, 2 * np.pi * probe_db_and_hz[1] * t, int(fs * t))
        ) * 10 ** ((probe_db_and_hz[0] - full_scale_sine_db) / 20)
        sig = audio_signal.Signal(
            sample_rate=fs, samples=masker_samples + probe_samples
        )
        channels = cam.Cam().channel_filter(sig)
        probe_channel_idx = -1
        for idx in range(np.asarray(channels.freqs).shape[0]):
            if (
                probe_db_and_hz[1] >= channels.freqs[idx][0]
                and probe_db_and_hz[1] < channels.freqs[idx][1]
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
        self.assertGreaterEqual(masked_db, want_masked_amount_range[0])
        self.assertLessEqual(masked_db, want_masked_amount_range[1])


if __name__ == "__main__":
    unittest.main()
