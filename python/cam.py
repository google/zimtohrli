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
"""Handles converting between Hz and Cam."""

import dataclasses
from typing import Optional
import jax
import jax.numpy as jnp
import numpy as np
import scipy.signal
import elliptic
import audio_signal


@jax.tree_util.register_static
@dataclasses.dataclass
class Cam:
    """Handles converting between Hz and Cam.

    Also provides utility method to filter a signal into evenly (in Cam-space)
    distributed channels.

    Cam is defined in
    https://en.wikipedia.org/wiki/Equivalent_rectangular_bandwidth.

    Attributes:
      erbs_scale_1: The first scale parameter for the ERBS computation.
      erbs_scale_2: The second scale parameter for the ERBS computation.
      erbs_offset: The offset parameter for the ERBS computation.
      minimum_channel_lower_bound_hz: The minimum frequency human hearing can
        detect.
      minimum_channel_width_hz: The frequency resolution of human hearing in the
        lowest detectable frequency.
      maximum_channel_upper_bound: The maximum frequency human hearing can detect.
      elliptic_order: Order of the elliptic band pass filter when running filter
        bank.
      elliptic_ripple_pass: Max ripple dB in the pass band of the filter.
      elliptic_ripple_stop: Max ripple dB in the stop band of the filter. Ignored
        for filter order 1.
    """

    erbs_scale_1: audio_signal.Numerical = 21.4
    erbs_scale_2: audio_signal.Numerical = 0.00437
    erbs_offset: audio_signal.Numerical = 1.0
    minimum_channel_lower_bound_hz: audio_signal.Numerical = 20.0
    minimum_channel_width_hz: audio_signal.Numerical = 1.0
    maximum_channel_upper_bound: audio_signal.Numerical = 20000.0
    elliptic_order: audio_signal.Numerical = 1
    elliptic_ripple_pass: audio_signal.Numerical = 3
    elliptic_ripple_stop: audio_signal.Numerical = 80

    _hz_freqs: Optional[audio_signal.Numerical] = None

    def tree_flatten(self):
        return (dataclasses.astuple(self), None)

    @classmethod
    def tree_unflatten(cls, _, children):
        return cls(*children)

    def __post_init__(self):
        """Initializes the Cam-equidistant frequency array of filters if missing."""
        if self._hz_freqs is None:
            start_cam = self.cam_from_hz(self.minimum_channel_lower_bound_hz)
            cam_step = (
                self.cam_from_hz(
                    self.minimum_channel_lower_bound_hz + self.minimum_channel_width_hz
                )
                - start_cam
            )
            stop_cam = self.cam_from_hz(self.maximum_channel_upper_bound)
            self._hz_freqs = self.hz_from_cam(np.arange(start_cam, stop_cam, cam_step))

    def hz_from_cam(self, cam: audio_signal.Numerical) -> audio_signal.Numerical:
        """Returns the Hz frequency for the provided Cam frequency."""
        return (10 ** (cam / self.erbs_scale_1) - self.erbs_offset) / self.erbs_scale_2

    def cam_from_hz(self, hz: audio_signal.Numerical) -> audio_signal.Numerical:
        """Returns the Cam frequency for the provided Hz frequency."""
        return self.erbs_scale_1 * np.log10(self.erbs_offset + self.erbs_scale_2 * hz)

    def channel_filter(self, sig: audio_signal.Signal) -> audio_signal.Channels:
        """Returns the signal filtered through a filter bank."""
        freqs = []
        filters = []
        for idx in range(np.asarray(self._hz_freqs).shape[0] - 1):
            bandpass = [self._hz_freqs[idx], self._hz_freqs[idx + 1]]
            freqs.append(bandpass)
            filters.append(
                scipy.signal.ellip(
                    N=self.elliptic_order,
                    rp=self.elliptic_ripple_pass,
                    rs=self.elliptic_ripple_stop,
                    Wn=bandpass,
                    btype="bandpass",
                    output="sos",
                    fs=sig.sample_rate,
                )
            )

        freqs_ary = np.asarray(freqs)

        return audio_signal.Channels(
            sample_rate=sig.sample_rate,
            freqs=freqs_ary,
            samples=elliptic.iirfilter(
                jnp.asarray(filters),
                sig.samples,
            ),
        )
