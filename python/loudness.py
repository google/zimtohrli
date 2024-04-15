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

"""Conversion functions between Phons and dB SPL according to ISO226 2003."""

import dataclasses
import jax
import jax.numpy as jnp
from google3.third_party.zimtohrli.python import signal


@dataclasses.dataclass
class Loudness:
  """A model for loudness perception.

  See https://en.wikipedia.org/wiki/Equal-loudness_contour.

  Uses equations and parameters hand-crafted using
  'loudness_parameter_computation.ipynb' to predict the ISO226 parameters.

  Attributes:
    a_f_params: Coefficients to generate the a_f variable in the ISO226 standard
      formula to convert between SPL and Phons.
    l_u_params: Coefficients to generate the l_u variable in the ISO226 standard
      formula to convert between SPL and Phons.
    t_f_params: Coefficients to generate the t_f variable in the ISO226 standard
      formula to convert between SPl and Phons.
  """

  a_f_params: signal.NumericalArray = (
      2.96990409e01,
      -8.55964789e-02,
      -8.86152561e00,
      1.14788453e00,
      1.03191174e-03,
      -6.69102053e-08,
      3.26541421e02,
      9.46236237e-01,
      6.31936884e-02,
  )
  l_u_params: signal.NumericalArray = (
      -2.21227768e02,
      6.48460480e01,
      -1.09949338e00,
      -2.14484299e-03,
      9.03649424e00,
      -1.17511692e00,
      3.49224194e-01,
      8.75495124e00,
      -3.10751297e-01,
      1.58859163e-01,
      1.98339320e04,
      -2.89066521e-01,
      1.42205943e00,
  )
  t_f_params: signal.NumericalArray = (
      1.34376528e02,
      -2.07181996e01,
      3.96974689e01,
      -2.95253463e00,
      1.57064444e-02,
      -5.77031151e-07,
      -5.47106465e-12,
      5.06185681e00,
      5.64349445e-02,
  )

  def _a_f(self, x: signal.Numerical) -> signal.Numerical:
    """Returns the a_f variabel used to convert between SPL and Phons for x.

    The function used is a manually selected using
    loudness_parameter_computation.ipynb.

    Args:
      x: The frequency to generate a_f for.

    Returns:
      a_f for the given frequency.
    """
    return (
        self.a_f_params[0]
        + self.a_f_params[1]
        * jnp.log(jnp.abs(self.a_f_params[2] + self.a_f_params[3] * x) + 1e-8)
        + self.a_f_params[4] * x
        + self.a_f_params[5] * jnp.square(x)
        - 0.1
        * self.a_f_params[6]
        * jnp.exp(
            -(jnp.square(8000 * self.a_f_params[7] - x))
            / 30000000
            * self.a_f_params[8]
        )
    )

  def _l_u(self, x: signal.Numerical) -> signal.Numerical:
    """Returns the l_u variabel used to convert between SPL and Phons for x.

    The function used is a manually selected using
    loudness_parameter_computation.ipynb.

    Args:
      x: The frequency to generate l_u for.

    Returns:
      l_u for the given frequency.
    """
    return (
        self.l_u_params[0]
        + self.l_u_params[1]
        * jnp.log(jnp.abs(self.l_u_params[2] + self.l_u_params[3] * x) + 1e-8)
        + 10
        * self.l_u_params[4]
        * jnp.exp(
            -((560 * self.l_u_params[5] - x) ** 2) / 700000 * self.l_u_params[6]
        )
        + 15
        * self.l_u_params[7]
        * jnp.exp(
            -((3000 * self.l_u_params[8] - x) ** 2)
            / 7000000
            * self.l_u_params[9]
        )
        - 5
        * self.l_u_params[10]
        * jnp.exp(
            -((1400 * self.l_u_params[11] - x) ** 2)
            / 30000
            * self.l_u_params[12]
        )
    )

  def _t_f(self, x: signal.Numerical) -> signal.Numerical:
    """Returns the t_f variable used to convert between SPL and Phons for x.

    The function used is a manually selected using
    loudness_parameter_computation.ipynb.

    Args:
      x: The frequency to generate t_f for.

    Returns:
      t_f for the given frequency.
    """
    return (
        self.t_f_params[0]
        + self.t_f_params[1]
        * jnp.log(jnp.abs(self.t_f_params[2] + self.t_f_params[3] * x) + 1e-8)
        + x * self.t_f_params[4]
        + x**2 * self.t_f_params[5]
        + x**3 * self.t_f_params[6]
        + 4
        * self.t_f_params[7]
        * jnp.exp(-((1500 - x) ** 2) / 100000 * self.t_f_params[8])
    )

  def phons_from_spl(
      self, spl: signal.Numerical, freq: signal.Numerical
  ) -> signal.Numerical:
    """Converts from dB SPL to Phons.

    This function uses the formula from ISO 226:2003 to convert from sound
    pressure to loudness, which isn't the exact inverse of `phons_to_spl`.

    Args:
      spl: Intensities in dB SPL.
      freq: Frequency in Hz.

    Returns:
      The intensity in Phons.
    """
    # a_f, L_U, T_f, and B_f are the symbols used in ISO 226:2003, so keeping
    # the names here makes sense despite naming standards.
    # pylint: disable=invalid-name
    a_f = self._a_f(freq)
    L_U = self._l_u(freq)
    T_f = self._t_f(freq)

    def expf(x: signal.Numerical) -> signal.Numerical:
      return (0.4 * (10 ** ((x + L_U) / 10 - 9))) ** a_f

    B_f = expf(spl) - expf(T_f) + 0.005135
    return 40 * jnp.log10(B_f) + 94

  def spl_from_phons(
      self, phons: signal.Numerical, freq: signal.Numerical
  ) -> signal.Numerical:
    """Convert from dB SPL to Phons.

    This uses the formula from ISO 226:2003 to convert from loudness to sound
    pressure level, which isn't the exact inverse of `spl_to_phons`.

    Args:
      phons: A (n_frequencies, n_time_steps)-shaped array with the input
        intensities in Phons.
      freq: A (n_frequencies,)-shaped array with the input frequencies in Hz.

    Returns:
      The intensity in dB SPL.
    """
    # a_f, L_U, T_f, and A_f are the symbols used in ISO 226:2003, so keeping
    # the names here makes sense despite naming standards.
    # pylint: disable=invalid-name
    a_f = self._a_f(freq)
    L_U = self._l_u(freq)
    T_f = self._t_f(freq)
    A_f = 4.47e-3 * ((10 ** (0.025 * phons)) - 1.15)
    A_f += (0.4 * (10 ** ((T_f + L_U) / 10 - 9))) ** a_f
    return 10.0 / a_f * jnp.log10(A_f) - L_U + 94.0

  def phons_from_spl_for_channels(
      self, channels: signal.Channels
  ) -> signal.Channels:
    """Convert a set of channels from dB SPL to Phons.

    Implementation created using jax.vmap in self.__post__init__.

    Args:
      channels: Channels containing dB SPL values.

    Returns:
      Channels with Phons values.
    """
    # pytype: disable=attribute-error
    return signal.Channels(
        samples=self._phons_from_spl_multi_multi_spl(
            channels.samples, jnp.mean(channels.freqs, axis=-1)
        ),
        freqs=channels.freqs,
        sample_rate=channels.sample_rate,
    )
    # pytype: enable=attribute-error

  def __post_init__(self):
    phons_from_spl_multi_spl = jax.vmap(self.phons_from_spl, (0, None), 0)
    setattr(
        self,
        '_phons_from_spl_multi_multi_spl',
        jax.vmap(phons_from_spl_multi_spl, (0, 0), 0),
    )
