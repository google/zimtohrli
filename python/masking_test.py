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

import numpy as np
import unittest
import parameterized
import masking
import audio_signal
import jax.numpy as jnp
import cam


class MaskingTest(unittest.TestCase):

    @parameterized.parameterize(
        dict(
            masker_db_and_hz=(80, 600),
            probe_db_and_hz=(70, 700),
            want_masked=False,
        ),
        dict(
            masker_db_and_hz=(80, 600),
            probe_db_and_hz=(30, 700),
            want_masked=True,
        ),
        dict(
            masker_db_and_hz=(80, 600),
            probe_db_and_hz=(30, 2500),
            want_masked=False,
        ),
        dict(
            masker_db_and_hz=(80, 600),
            probe_db_and_hz=(30, 580),
            want_masked=True,
        ),
        dict(
            masker_db_and_hz=(80, 600),
            probe_db_and_hz=(30, 200),
            want_masked=False,
        ),
    )
    def test_non_masked_energy(
        self,
        masker_db_and_hz: tuple[float, float],
        probe_db_and_hz: tuple[float, float],
        want_masked: bool,
    ):
        m = masking.Masking()
        chans = audio_signal.Channels(
            sample_rate=48000,
            freqs=np.asarray(
                [
                    [masker_db_and_hz[1], masker_db_and_hz[1] + 1],
                    [probe_db_and_hz[1], probe_db_and_hz[1] + 1],
                ],
            ),
            samples=np.asarray([[masker_db_and_hz[0]], [probe_db_and_hz[0]]]),
        )
        non_masked = m.non_masked_energy(chans)
        self.assertEqual(non_masked.samples[1, 0] < 0, want_masked)


if __name__ == "__main__":
    unittest.main()
