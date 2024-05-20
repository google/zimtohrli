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
      max_mask: The maximum dB above full masking a probe will ever be masked.
      cam_model: The Cam model to use when computing masking.
    """

    lower_zero_at_20: audio_signal.Numerical = -2
    lower_zero_at_80: audio_signal.Numerical = -6
    upper_zero_at_20: audio_signal.Numerical = 2
    upper_zero_at_80: audio_signal.Numerical = 10

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

    @jax.jit
    def non_masked_energy(
        self, energy_channels_db: audio_signal.Channels
    ) -> audio_signal.Channels:
        """Returns energy_channels after full masking.

        Args:
          energy_channels_db: Channels containing the energy to be masked.

        Returns:
          energy_channels after having removed masked components.
        """
        cams = self.cam_model.cam_from_hz(energy_channels_db.freqs[:, 0])
        cam_delta = cams[jnp.newaxis, ...] - cams[..., jnp.newaxis]
        max_full_masking_db = jnp.max(
            self.full_masking_of_channels(cam_delta, energy_channels_db.samples), axis=0
        ).T
        return audio_signal.Channels(
            samples=jnp.where(
                max_full_masking_db >= energy_channels_db.samples,
                energy_channels_db.samples - max_full_masking_db,
                energy_channels_db.samples,
            ),
            freqs=energy_channels_db.freqs,
            sample_rate=energy_channels_db.sample_rate,
        )
