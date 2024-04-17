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
"""Handles masking of sounds."""

import dataclasses
from typing import Optional
import jax
import jax.numpy as jnp
import cam
import audio_signal


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class Masking:
    """Handles masking of sounds.

    Attributes:
      lower_zero_at_20: The negative distance in Cam at which a 20dB masker will
        no longer mask any probe.
      lower_zero_at_80: The negative distance in Cam at which an 80dB masker will
        no longer mask any probe.
      upper_zero_at_20: The  positive distance in Cam at which a 20dB masker will
        no longer mask any probe.
      upper_zero_at_80: The positive distance in Cam at which an 80dB masker will
        no longer mask any probe.
      onset_width: The dB a probe has to be raised above full masking to be masked
        only 'onset_peak'dB.
      onset_peak: The masking of a probe after it has been raised 'onset_width'dB
        above full masking.
      max_mask: The maximum dB above full masking a probe will ever be masked.
      cam_model: The Cam model to use when computing masking.
    """

    lower_zero_at_20: audio_signal.Numerical = -2
    lower_zero_at_80: audio_signal.Numerical = -6
    upper_zero_at_20: audio_signal.Numerical = 2
    upper_zero_at_80: audio_signal.Numerical = 10

    onset_width: audio_signal.Numerical = 10
    onset_peak: audio_signal.Numerical = 6
    max_mask: audio_signal.Numerical = 20

    cam_model: Optional[cam.Cam] = None

    def tree_flatten(self):
        return (
            tuple(getattr(self, field.name) for field in dataclasses.fields(self)),
            None,
        )

    @classmethod
    def tree_unflatten(cls, _, children):
        return cls(*children)

    def full_masking(
        self, cam_delta: jnp.ndarray, masker_level: jnp.ndarray
    ) -> jnp.ndarray:
        """Returns full masking level.

        Args:
          cam_delta: Distance in Cam from masker to probe.
          masker_level: dB of masker

        Returns:
          The level at which a probe would be fully masked.
        """
        lower_zero = jnp.minimum(
            -0.1,
            self.lower_zero_at_20
            + (masker_level - 20)
            * (self.lower_zero_at_80 - self.lower_zero_at_20)
            / 60,
        )
        lower_slope = (masker_level - 20) / -lower_zero
        upper_zero = jnp.maximum(
            0.1,
            self.upper_zero_at_20
            + (masker_level - 20)
            * (self.upper_zero_at_80 - self.upper_zero_at_20)
            / 60,
        )
        upper_slope = (masker_level - 20) / upper_zero
        return jnp.maximum(
            0,
            jnp.minimum(
                lower_slope * (cam_delta - lower_zero),
                upper_slope * (upper_zero - cam_delta),
            ),
        )

    def masked_amount(
        self, full_mask_level: jnp.ndarray, probe_level: jnp.ndarray
    ) -> jnp.ndarray:
        """Returns masked amount of probe for given full_mask_level.

        Args:
          full_mask_level: The dB at which probe would be fully masked.
          probe_level: Objective dB of probe.

        Returns:
          dB of probe that would be masked away.
        """
        onset_delta = jnp.minimum(1e-6, self.onset_peak - full_mask_level)
        onset_slope = onset_delta / self.onset_width
        onset_offset = full_mask_level - full_mask_level / onset_slope
        onset_crossing = full_mask_level + self.onset_width
        max_mask_slope = jnp.minimum(self.onset_peak, full_mask_level) / -(
            full_mask_level + self.max_mask - onset_crossing
        )
        max_mask_offset = full_mask_level + self.max_mask
        return jnp.clip(  # Never mask more than full_masking and never less than 0.
            jnp.minimum(  # Always mask at least enough to reach 0 at full + 20.
                jnp.maximum(  # If we are loud, reduce masking quickly at start
                    (probe_level - onset_offset) * onset_slope,
                    (probe_level - max_mask_offset) * max_mask_slope,
                ),
                (probe_level - full_mask_level - self.max_mask)
                * (full_mask_level / -self.max_mask),
            ),
            0,
            full_mask_level,
        )

    def full_masking_of_channels(
        self, cam_delta: jnp.ndarray, masker_level: jnp.ndarray
    ) -> jnp.ndarray:
        """Returns full masking level of channels.

        Implementation created using jax.vmap in self.__post__init__.

        Args:
          cam_delta: A (num_channels, num_channels)-shaped array of distances, in
            Cam, betwen each channel and each other channel.
          masker_level: A (num_channels, num_steps)-shaped array of energies, in dB,
            of the channels.

        Returns:
          A (num_masking_channels, num_steps, num_masked_channels)-shaped array of
            full masking levels, in dB, between each channel and each other channel.
        """
        del (cam_delta, masker_level)
        return jnp.ndarray([])

    def masked_amount_of_channels(
        self, full_mask_level: jnp.ndarray, probe_level: jnp.ndarray
    ) -> jnp.ndarray:
        """Returns masked amount of channels.

        Implementation created using jax.vmap in self.__post__init__.

        Args:
          full_mask_level: A (num_masking_channels, num_steps, num_masked_channels)-
            shaped array of full masking levels, in dB, between each channel and
            each other channel.
          probe_level: A (num_channels, num_steps)-shaped array of energies, in dB,
            of the channels.

        Returns:
          A (num_masked_channels, num_steps, num_masking_channels)-shaped array of
            the dB being masked in each channel by each other channel.
        """
        del (full_mask_level, probe_level)
        return jnp.ndarray([])

    def __post_init__(self):
        if self.cam_model is None:
            self.cam_model = cam.Cam()

        # [probes], [] => [probes]
        full_masking_multi_probes = jax.vmap(self.full_masking, (0, None), 0)
        # [maskers, probes], [maskers] => [maskers, probes]
        full_masking_multi_maskers_multi_probes = jax.vmap(
            full_masking_multi_probes, (0, 0), 0
        )
        # [maskers, probes], [maskers, steps] => [maskers, steps, probes]
        setattr(
            self,
            "full_masking_of_channels",
            jax.jit(jax.vmap(full_masking_multi_maskers_multi_probes, (None, 1), 1)),
        )

        # [maskers], [] => [maskers]
        masked_amount_multi_maskers = jax.vmap(self.masked_amount, (0, None), 0)
        # [maskers, probes], [probes] => [probes, maskers]
        masked_amount_multi_maskers_multi_probes = jax.vmap(
            masked_amount_multi_maskers, (1, 0), 0
        )
        # [maskers, steps, probes], [probes, steps] => [probes, steps, maskers]
        setattr(
            self,
            "masked_amount_of_channels",
            jax.jit(jax.vmap(masked_amount_multi_maskers_multi_probes, (1, 1), 1)),
        )

    @jax.jit
    def partial_loudness(
        self, energy_channels_db: audio_signal.Channels
    ) -> audio_signal.Channels:
        """Returns partial loudness of energy_channels after masking.

        Args:
          energy_channels_db: Channels containing the energy to be masked.

        Returns:
          energy_channels after having removed masked components.
        """
        cams = self.cam_model.cam_from_hz(energy_channels_db.freqs[:, 0])
        cam_delta = cams[jnp.newaxis, ...] - cams[..., jnp.newaxis]
        full_masking_db = self.full_masking_of_channels(
            cam_delta, energy_channels_db.samples
        )
        masked_amount_db = self.masked_amount_of_channels(
            full_masking_db, energy_channels_db.samples
        )
        masked_amount = 10 ** (masked_amount_db / 10) - 1
        masked_amount_sum = masked_amount.sum(axis=-1)
        masked_amount_sum_db = 10 * jnp.log10(masked_amount_sum + 1)
        partial_loudness_db = jnp.asarray(energy_channels_db.samples) - jnp.asarray(
            masked_amount_sum_db
        )
        return audio_signal.Channels(
            samples=partial_loudness_db,
            freqs=energy_channels_db.freqs,
            sample_rate=energy_channels_db.sample_rate,
        )
