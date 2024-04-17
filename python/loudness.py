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
import audio_signal


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

    a_f_params: audio_signal.NumericalArray = (
        7.07699805e-01,
        9.56847202e-02,
        1.39454848e00,
        5.17191537e00,
        1.42505082e01,
        1.89421310e-02,
        1.40024559e00,
        6.43211270e02,
        4.54202241e-01,
        -2.04904897e00,
    )
    l_u_params: audio_signal.NumericalArray = (
        1.11007486e02,
        1.16744188e01,
        5.09084895e01,
        1.08211378e01,
        1.09495723e00,
        4.79483920e-01,
        1.07492801e00,
        7.73080187e01,
        2.24191011e-01,
        -1.16694726e00,
        1.34093456e00,
        4.71415134e-01,
        1.06548091e00,
        2.37158204e-01,
        -2.78563078e-04,
        -6.89758642e01,
    )
    t_f_params: audio_signal.NumericalArray = (
        -1.64433785e02,
        -2.10692436e01,
        8.30609509e01,
        1.32945921e01,
        4.60795058e00,
        5.10280480e-02,
        1.12728328e00,
        -2.38605854e01,
        4.24443844e-02,
        1.75220290e00,
        9.23388482e-06,
        -3.28913733e-03,
        -5.37290840e01,
    )

    def _a_f(self, x: audio_signal.Numerical) -> audio_signal.Numerical:
        """Returns the a_f variabel used to convert between SPL and Phons for x.

        The function used is a manually selected using
        loudness_parameter_computation.ipynb.

        Args:
          x: The frequency to generate a_f for.

        Returns:
          a_f for the given frequency.
        """
        params = self.a_f_params
        return (
            params[0]
            - params[1] * jnp.log(params[2] * (x - params[3]))
            + 0.04
            * params[4]
            * jnp.exp(-(0.0000001 * params[5] * (x - 14000 * params[6]) ** 2))
            - 0.03
            * params[7]
            * jnp.exp(-(0.0000001 * params[8] * (x - 5000 * params[9]) ** 2))
        )

    def _l_u(self, x: audio_signal.Numerical) -> audio_signal.Numerical:
        """Returns the l_u variabel used to convert between SPL and Phons for x.

        The function used is a manually selected using
        loudness_parameter_computation.ipynb.

        Args:
          x: The frequency to generate l_u for.

        Returns:
          l_u for the given frequency.
        """
        params = self.l_u_params
        return (
            params[0]
            + params[1] * jnp.log(params[2] * (x - params[3]))
            - 5
            * params[4]
            * jnp.exp(-0.00001 * params[5] * (x - 1500 * params[6]) ** 2)
            + 5
            * params[7]
            * jnp.exp(-0.000001 * params[8] * (x - 3000 * params[9]) ** 2)
            - 15
            * params[10]
            * jnp.exp(-0.0000001 * params[11] * (x - 8000 * params[12]) ** 2)
            - 5
            * params[13]
            * jnp.exp(-0.00000001 * params[14] * (x - 20000 * params[15]) ** 2)
        )

    def _t_f(self, x: audio_signal.Numerical) -> audio_signal.Numerical:
        """Returns the t_f variable used to convert between SPL and Phons for x.

        The function used is a manually selected using
        loudness_parameter_computation.ipynb.

        Args:
          x: The frequency to generate t_f for.

        Returns:
          t_f for the given frequency.
        """
        params = self.t_f_params
        return (
            params[0]
            + params[1] * jnp.log(params[2] * (x - params[3]))
            + 5
            * params[4]
            * jnp.exp(-0.00001 * params[5] * (x - 1200 * params[6]) ** 2)
            - 10
            * params[7]
            * jnp.exp(-0.0000001 * params[8] * (x - 3300 * params[9]) ** 2)
            + 20
            * params[10]
            * jnp.exp(-0.00000001 * params[11] * (x - 12000 * params[12]) ** 2)
        )

    def phons_from_spl(
        self, spl: audio_signal.Numerical, freq: audio_signal.Numerical
    ) -> audio_signal.Numerical:
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

        def expf(x: audio_signal.Numerical) -> audio_signal.Numerical:
            return (0.4 * (10 ** ((x + L_U) / 10 - 9))) ** a_f

        B_f = expf(spl) - expf(T_f) + 0.005135
        res = 40 * jnp.log10(B_f) + 94
        return res

    def spl_from_phons(
        self, phons: audio_signal.Numerical, freq: audio_signal.Numerical
    ) -> audio_signal.Numerical:
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
        self, channels: audio_signal.Channels
    ) -> audio_signal.Channels:
        """Convert a set of channels from dB SPL to Phons.

        Implementation created using jax.vmap in self.__post__init__.

        Args:
          channels: Channels containing dB SPL values.

        Returns:
          Channels with Phons values.
        """
        # pytype: disable=attribute-error
        return audio_signal.Channels(
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
            "_phons_from_spl_multi_multi_spl",
            jax.vmap(phons_from_spl_multi_spl, (0, 0), 0),
        )
