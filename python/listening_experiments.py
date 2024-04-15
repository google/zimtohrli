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

"""Functionality for audio signals used in listening experiments."""

import numpy as np
from numpy import typing as np_typing


class ListeningExperiment:
  """Acoustic data for a listening experiment.

  Attributes:
    probe: A (n_time_steps,)-shaped numpy.ndarray of floats.
    variant: A (n_time_steps,)-shaped numpy.ndarray of floats.
    sample_rate: Assumed sample rate in samples per second of both the probe
      and the variant.
  """

  def __init__(self, probe: np_typing.ArrayLike, variant: np_typing.ArrayLike,
               sample_rate: float):
    """Makes sure to handle other Python objects convertible to numpy-arrays.

    Args:
      probe: A (n_time_steps,)-shaped array-like of floats.
      variant: A (n_time_steps,)-shaped array-like of floats.
      sample_rate: Assumed sample rate in samples per second of both the probe
        and the variant.
    """
    self.probe = np.array(probe)
    self.variant = np.array(variant)
    self.sample_rate = sample_rate

  def loop(self,
           snippet: np_typing.ArrayLike,
           repeat: int,
           pause_in_ms: float = 0,
           start_with_zeros: bool = False) -> np.ndarray:
    """Generates loops, optionally adding pauses.

    Takes an input audio signal snippet and returns an audio signal
    repeated multiple times together with a certain amount of silence added
    before each repetition of the snippet. Optionally, the added silence for the
    first snippet can be skipped.

    Args:
      snippet: A (n_time_steps,)-shaped array-like of floats.
      repeat: Number of times the snippet should be repeated.
      pause_in_ms: The length of the silence in ms added before the snippets.
      start_with_zeros: If True, slience will be added also before the first
        repetition of the snippet.

    Returns:
      A np.ndarray of concatenated copies of the snippet and pauses.
    """
    snippet = np.asarray(snippet)
    pause_in_samples = pause_in_ms * self.sample_rate // 1000
    loop = np.tile(np.pad(snippet, [(pause_in_samples, 0)]), [repeat])
    return loop if start_with_zeros else loop[pause_in_samples:]

  def place_in_sequence(self,
                        pause_in_ms: float,
                        repeat: int = 8,
                        variant_position: int = 5) -> np.ndarray:
    """Generates a sequence containing the probe and the variant.

    This generates a sequence consisting of the repetitions of the `probe` and
    `variant`, with interspearsed silence. The `variant` will only sound once.

    Args:
      pause_in_ms: The length of the silence in ms added between the snippets.
      repeat: The total number of repetitions. The probe signal will be repeated
        (repeat - 1)-times in total.
      variant_position: The position of the `variant` in the sequence of
        repetitions. One-indexed, so that valid values here are in {1, ...,
        repeat}.

    Returns:
      A (n_time_steps,)-shaped numpy array of floats.
    """
    return np.concatenate([
        self.loop(
            self.probe,
            repeat=variant_position - 1,
            pause_in_ms=pause_in_ms,
        ),
        self.loop(
            self.variant,
            repeat=1,
            pause_in_ms=pause_in_ms,
            start_with_zeros=True),
        self.loop(
            self.probe,
            repeat=repeat - variant_position,
            pause_in_ms=pause_in_ms,
            start_with_zeros=True)
    ])
