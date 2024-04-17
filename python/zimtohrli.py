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
"""Top level functionality for the Zimtohrli perceptual audio metric."""

import dataclasses
import jax.numpy as jnp
import cam
import loudness
import masking
import audio_signal


@dataclasses.dataclass(frozen=True)
class Zimtohrli:
    """Perceptual audio metric main class.

    Attributes:
      c: The frequency resolution model.
      l: The sound pressure resolution model.
      m: The acoustic masking model.
      frequency_norm_order: The order of the norm across frequencies when
        computing distance.
      time_norm_order: The order of the norm across time steps when computing
        distance.
    """

    c: cam.Cam = dataclasses.field(default_factory=cam.Cam)
    l: loudness.Loudness = dataclasses.field(default_factory=loudness.Loudness)
    m: masking.Masking = dataclasses.field(default_factory=masking.Masking)

    frequency_norm_order: audio_signal.Numerical = 4
    time_norm_order: audio_signal.Numerical = 4

    def spectrogram(
        self,
        signal: audio_signal.Signal,
        full_scale_sine_db: jnp.ndarray = jnp.asarray(90),
        db_epsilon: jnp.ndarray = jnp.asarray(1e-9),
    ) -> audio_signal.Channels:
        """Returns a perceptual spectrogram of the signal.

        Args:
          signal: A digital signal.
          full_scale_sine_db: The reference dB SPL that a sine wave of amplitude 1
            would have in the signal.
          db_epsilon: The epsilon to add to the signal when converting to dB, to log
            of zero.

        Returns:
          The channels of the resulting Zimtohrli spectrogram.
        """
        cam_channels = self.c.channel_filter(signal)
        energy_db = cam_channels.energy().to_db(
            full_scale_sine_db=full_scale_sine_db, db_epsilon=db_epsilon
        )
        partial_energy_db = self.m.partial_loudness(energy_db)
        return self.l.phons_from_spl_for_channels(partial_energy_db)

    def distance(
        self, spectrogram_a: audio_signal.Channels, spectrogram_b: audio_signal.Channels
    ) -> audio_signal.Numerical:
        """Returns the perceptual distance between the two spectrograms.

        Args:
          spectrogram_a: A Zimtohrli spectrogram to compare.
          spectrogram_b: A Zimtohrli spectrogram to compare.

        Returns:
          A number describing the Zimtohrli distance between the spectrograms.
        """
        return jnp.linalg.norm(
            jnp.linalg.norm(
                spectrogram_a.samples - spectrogram_b.samples,
                axis=0,
                ord=self.frequency_norm_order,
            ),
            ord=self.time_norm_order,
        )
