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

def mos_from_signals(signal_a: npt.ArrayLike, signal_b: npt.ArrayLike) -> float:
    """Returns the Mean Opinion Score (MOS) between two audio signals using Zimtohrli"""
    distance = Pyohrli().distance(signal_a, signal_b)
    mos = mos_from_zimtohrli(distance)
    return mos

def mos_from_zimtohrli(zimtohrli_distance: float) -> float:
  """Returns an approximate mean opinion score based on the provided Zimtohrli distance."""
  return _pyohrli.MOSFromZimtohrli(zimtohrli_distance)


class Spectrogram:
  """Wrapper around C++ zimtohrli::Spectrogram."""

  _cc_analysis: _pyohrli.Spectrogram


class Pyohrli:
  """Wrapper around C++ zimtohrli::Zimtohrli."""

  _cc_pyohrli: _pyohrli.Pyohrli

  def __init__(self):
    """Initializes the instance."""
    self._cc_pyohrli = _pyohrli.Pyohrli()

  def distance(self, signal_a: npt.ArrayLike, signal_b: npt.ArrayLike) -> float:
    """Computes the distance between two signals.

    The signals must be (num_samples,)-shaped 48kHz [-1, 1] float arrays.

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

  def analyze(self, signal: npt.ArrayLike) -> npt.ArrayLike:
    """Computes the Zimtohrli spectrogram of a signal.compile.

    Args:
      signal: A (num_samples,)-shaped 48kHz [-1, 1] float array.

    Returns:
      A (num_steps, num_dims)-shaped float array with the Zimtohrli
      spectrogram of the signal.
    """
    bts = self._cc_pyohrli.analyze(
        np.asarray(signal).astype(np.float32).ravel().data
    )
    result = np.frombuffer(bts, dtype=np.float32)
    num_rotators = self._cc_pyohrli.num_rotators()
    return result.reshape((result.shape[0] // num_rotators, num_rotators))

  @property
  def sample_rate(self) -> float:
    """Expected sample rate of analyzed audio."""
    return self._cc_pyohrli.sample_rate()

  @property
  def num_rotators(self) -> int:
    """Number of rotators (spectrogram dimensions)."""
    return int(self._cc_pyohrli.num_rotators())

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
  def nsim_step_window(self) -> float:
    """Order of the window in perceptual_sample_rate time steps when compting the NSIM."""
    return self._cc_pyohrli.get_nsim_step_window()

  @nsim_step_window.setter
  def nsim_step_window(self, value: float):
    self._cc_pyohrli.set_nsim_step_window(value)

  @property
  def nsim_channel_window(self) -> float:
    """Order of the window in channels when computing the NSIM."""
    return self._cc_pyohrli.get_nsim_channel_window()

  @nsim_channel_window.setter
  def nsim_channel_window(self, value: float):
    self._cc_pyohrli.set_nsim_channel_window(value)
