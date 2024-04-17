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
"""Elliptic filter design and application."""

import jax
import jax.numpy as jnp
import audio_signal


@jax.jit
def iirfilter(
    coeffs: audio_signal.NumericalArray, samples: audio_signal.NumericalArray
) -> audio_signal.NumericalArray:
    """Filters a signal using a set of IIR filters expressed as num, den, gain.

    Args:
      coeffs: A (num_filters, num_sections, 6)-shaped array of numerators and
        denominators defining the filter, where each filter conforms to the output
        of scipy.signal.ellip(output='sos').
      samples: A (num_samples,)-shaped array of samples.

    Returns:
      A (num_filters, num_samples)-shaped array of filtered samples.
    """
    jax_coeffs = jnp.asarray(coeffs)
    assert jax_coeffs.shape[2] == 6

    def per_sample_fn(per_sample_carry, sample):
        in_buffer, out_buffer, a, b = per_sample_carry

        in_buffer = in_buffer.at[:, :, 1:].set(in_buffer[:, :, :-1])
        in_buffer = in_buffer.at[:, 0, 0].set(sample)

        def per_section_fn(carry, section_index):
            in_buffer, out_buffer, a, b = carry

            out = (
                jnp.einsum(
                    "FC,FC->F", b[:, section_index, :], in_buffer[:, section_index, :]
                )
                - jnp.einsum(
                    "FC,FC->F",
                    a[:, section_index, 1:],
                    out_buffer[:, section_index, :],
                )
            ) * a[:, section_index, 0]

            in_buffer = jax.lax.cond(
                section_index + 1 < in_buffer.shape[1],
                lambda: in_buffer.at[:, section_index + 1, 0].set(out),
                lambda: in_buffer,
            )

            out_carry = (in_buffer, out_buffer, a, b)
            return out_carry, out

        per_section_carry = (in_buffer, out_buffer, a, b)
        (in_buffer, out_buffer, a, b), out = jax.lax.scan(
            per_section_fn, per_section_carry, jnp.arange(a.shape[1])
        )
        out_buffer = out_buffer.at[:, :, 1:].set(out_buffer[:, :, :-1])
        out_buffer = out_buffer.at[:, :, 0].set(out.T)
        out_carry = (in_buffer, out_buffer, a, b)
        return out_carry, out[-1]

    in_buffer = jnp.zeros_like(jax_coeffs[:, :, :3])
    out_buffer = jnp.zeros_like(jax_coeffs[:, :, :2])
    b = coeffs[:, :, :3]
    a = coeffs[:, :, 3:]
    a = a.at[:, :, 0].set(1.0 / a[:, :, 0])
    per_sample_carry = (in_buffer, out_buffer, a, b)
    _, out = jax.lax.scan(per_sample_fn, per_sample_carry, samples)
    return out.T
