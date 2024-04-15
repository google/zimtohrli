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
"""Generic classes containing signals."""

import dataclasses
from typing import Union
import jax
import jax.numpy as jnp
import numpy as np


Numerical = Union[np.ndarray, jnp.ndarray, float, int]
NumericalArray = Union[
    np.ndarray,
    jnp.ndarray,
    list[float],
    list[int],
    tuple[float, ...],
    tuple[int, ...],
]


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class Signal:
  """Class defining a digital signal at a given sample rate.

  Attributes:
    sample_rate: The number of samples per second in the signal.
    samples: A (num_samples,)-shaped array with the signal samples, in the range
      -1 to 1.
  """

  sample_rate: Numerical
  samples: NumericalArray

  def tree_flatten(self):
    return (dataclasses.astuple(self), None)

  @classmethod
  def tree_unflatten(cls, _, children):
    return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class Channels:
  """Class defining a set of digital signals being related channels.

  Attributes:
    sample_rate: The number of samples per second in the signal.
    samples: A (num_channels, num_samples)-shaped array with the samples of the
      channel signals, in the range -1 to 1.
    freqs: A (num_channels, 2)-shaped array with the low and high pass
      frequencies of the channels.
  """

  sample_rate: Numerical
  samples: NumericalArray
  freqs: NumericalArray

  def tree_flatten(self):
    return (dataclasses.astuple(self), None)

  @classmethod
  def tree_unflatten(cls, _, children):
    return cls(*children)

  def energy(self, out_sample_rate: int = 100) -> "Channels":
    """Returns the energy in the channels.

    Will downsample from the sample rate of the channels to out_sample_rate.

    Args:
      out_sample_rate: The sample rate to downsample to.

    Returns:
      The energy in the channels, downsampled to the out_sample_rate.
    """
    samples = jnp.asarray(self.samples)
    return Channels(
        freqs=self.freqs,
        sample_rate=out_sample_rate,
        samples=jax.image.resize(
            jnp.square(self.samples),
            shape=(
                samples.shape[0],
                int(samples.shape[1] * out_sample_rate / self.sample_rate),
            ),
            method="linear",
        ),
    )

  def to_db(
      self,
      full_scale_sine_db: Numerical = 90,
      db_epsilon: Numerical = 1e-9,
  ) -> "Channels":
    """Returns the channels in dB relative full_scale_sine_db.

    Make sure to only call this on Channels that are the result of calling
    Channels.energy, since otherwise the dB conversion will get negative numbers
    which will cause nans.

    Args:
      full_scale_sine_db: The reference dB SPL of a full scale sine.
      db_epsilon: The epsilon to add to the energy before converting to dB to
        avoid log of zero.

    Returns:
      The energy in the channels in dB, downsample to the out_sample_rate.
    """
    return Channels(
        sample_rate=self.sample_rate,
        freqs=self.freqs,
        samples=full_scale_sine_db
        + 10 * jnp.log10(db_epsilon + jnp.asarray(self.samples)),
    )
