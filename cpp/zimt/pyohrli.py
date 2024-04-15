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
"""Pyohrli is a Zimtohrli wrapper in Python."""

import numpy as np
import numpy.typing as npt

import _pyohrli


class Analysis:
  """Wrapper around C++ zimtohrli::Analysis."""

  _cc_analysis: _pyohrli.Analysis


class Pyohrli:
  """Wrapper around C++ zimtohrli::Zimtohrli."""

  _cc_pyohrli: _pyohrli.Pyohrli

  def __init__(self, sample_rate: float, frequency_resolution: float):
    """Initializes the instance.

    Args:
      sample_rate: The sample rate this Pyohrli instance expects for signals.
      frequency_resolution: The smallest bandwidth of the channel filters, i.e.
        the expected frequency resolution of human hearing at low frequencies.
    """
    self._cc_pyohrli = _pyohrli.Pyohrli(sample_rate, frequency_resolution)

  def analyze(self, signal: npt.ArrayLike) -> Analysis:
    """Analyzes a signal.

    Args:
      signal: The signal to analyze. A (num_samples,)-shaped array of floats
        between -1 and 1. The expected playout intensity in dB SPL of a 1kHz
        sine wave between -1 and 1 is defined by setting 'full_scale_sine_db' of
        this Pyohrli instance.

    Returns:
      An Analysis instance containing a psychoacoustic analysis of the signal.
    """
    result = Analysis()
    # Disabling protected-access to avoid making Analysis._cc_pyohrli public.
    result._cc_analysis = self._cc_pyohrli.analyze(  # pylint: disable=protected-access
        np.asarray(signal).astype(np.float32).ravel().data,
    )
    return result

  def analysis_distance(
      self, analysis_a: Analysis, analysis_b: Analysis
  ) -> float:
    """Computes the distance between two psychoacoustic analyses.

    Args:
      analysis_a: An Analysis instance to compare.
      analysis_b: Another Analysis instance to compare with.

    Returns:
      The Zimtohrli distance between the two analyses.
    """
    return self._cc_pyohrli.analysis_distance(
        # Disabling protected-access to avoid making Analysis._cc_pyohrli
        # public.
        analysis_a._cc_analysis,  # pylint: disable=protected-access
        analysis_b._cc_analysis,  # pylint: disable=protected-access
    )

  def distance(self, signal_a: npt.ArrayLike, signal_b: npt.ArrayLike) -> float:
    """Computes the distance between two signals.

    See 'analyze' for the signal format.

    Args:
      signal_a: A signal to compare.
      signal_b: Another signal to compare with.

    Returns:
      The Zimtohrli distance between the signals.
    """
    return self._cc_pyohrli.distance(
        np.asarray(signal_a).astype(np.float32).ravel().data,
        np.asarray(signal_b).astype(np.float32).ravel().data,
    )

  @property
  def full_scale_sine_db(self) -> float:
    """Reference intensity for an amplitude 1.0 sine wave at 1kHz.

    Defaults to 80dB SPL.
    """
    return self._cc_pyohrli.get_full_scale_sine_db()

  @full_scale_sine_db.setter
  def full_scale_sine_db(self, value: float):
    self._cc_pyohrli.set_full_scale_sine_db(value)

  @property
  def perceptual_sample_rate(self) -> float:
    """Perceptual sample rate of the Pyohrli instance.

    Human hearing typically detects timing differences of ~10ms, which makes the
    default perceptual sample rate 100Hz.
    """
    return self._cc_pyohrli.perceptual_sample_rate

  @perceptual_sample_rate.setter
  def perceptual_sample_rate(self, value: float):
    self._cc_pyohrli.perceptual_sample_rate = value

  @property
  def time_norm_order(self) -> float:
    """Order of the norm across time steps when computing the distance score."""
    return self._cc_pyohrli.get_time_norm_order()

  @time_norm_order.setter
  def time_norm_order(self, value: float):
    self._cc_pyohrli.set_time_norm_order(value)

  @property
  def freq_norm_order(self) -> float:
    """Order of the norm across frequency channels."""
    return self._cc_pyohrli.get_freq_norm_order()

  @freq_norm_order.setter
  def freq_norm_order(self, value: float):
    self._cc_pyohrli.set_freq_norm_order(value)

  @property
  def unwarp_window(self) -> float:
    """Length of dynamic time warp unwarp window in seconds."""
    return self._cc_pyohrli.unwarp_window

  @unwarp_window.setter
  def unwarp_window(self, value: float):
    self._cc_pyohrli.unwarp_window = value
