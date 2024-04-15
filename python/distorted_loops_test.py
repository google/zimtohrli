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

"""Tests for zimtohrli.python.distorted_loops.

We test basic functionality of the methods that generate signal snippets,
distortions to apply to the snippets and distorted/undistorted snippet loops.
"""

import numpy as np
import scipy
from google3.testing.pybase import googletest

from google3.third_party.zimtohrli.python import distorted_loops


class DistortedLoopsTest(googletest.TestCase):

  def setUp(self):
    super().setUp()
    # Deterministic rng
    self.rng = np.random.default_rng(seed=1)

  def test_signal_beep(self):
    sine = distorted_loops.signal_beep(1000.)
    np.testing.assert_almost_equal(
        sine, 0.5 * np.sin(2*np.pi*1000*np.arange(4800)/48000))

  def test_frequency_distortion(self):
    sine = distorted_loops.signal_beep(1000.)
    distorted_sine = distorted_loops.frequency_distortion(sine, 100)

    np.testing.assert_almost_equal(
        distorted_sine,
        np.fft.ifft(np.pad(np.fft.fft(sine), [(100, 0)])[:len(sine)]))

  def test_timing_distortion(self):
    sine = distorted_loops.signal_beep(1000.)
    distorted_sine = distorted_loops.timing_distortion(sine, 10)

    np.testing.assert_almost_equal(
        distorted_sine, np.pad(sine, [(480, 0)]))

  def test_intensity_distortion(self):
    sine = distorted_loops.signal_beep(1000.)
    distorted_sine = distorted_loops.intensity_distortion(sine, 20)

    np.testing.assert_almost_equal(
        distorted_sine, 10*sine)

  def test_bandlimited_noise_masker_distortion(self):
    length = 4800
    width_hz = 200
    freq_hz = 1000
    fs = 48000
    stdev = 0.0
    nfft = 524288  # A large power of 2 for the number of fft points
    bandlimited_noise = distorted_loops.gen_bandlimited_noise_masker(
        length, width_hz, freq_hz, stdev, rng=self.rng)

    (_, psd) = scipy.signal.periodogram(
        bandlimited_noise,
        scaling='density',
        return_onesided=False,
        nfft=nfft)
    low = (freq_hz - width_hz // 2) * nfft // fs
    high = (freq_hz + width_hz // 2) * nfft // fs

    # Check that calculated noise power is in [0.9, 1.1]
    power = np.mean(psd[low:high])
    np.testing.assert_allclose(power, 1., rtol=0, atol=0.1)

  def test_white_noise_masker_distortion(self):
    length = 4800
    white_noise = distorted_loops.gen_white_noise_masker(
        length, 0., rng=self.rng)

    (_, psd) = scipy.signal.periodogram(
        white_noise,
        scaling='density',
        return_onesided=False)

    # Check that calculated noise power is in [0.9, 1.1]
    power = np.mean(psd[:])
    np.testing.assert_allclose(power, 1., rtol=0, atol=0.1)

  def test_sine_masker_distortion(self):
    length = 4800
    freq_hz = 1100
    sample_rate = 48000
    masker = distorted_loops.gen_sine_masker(length, 0.5, freq_hz, sample_rate)

    np.testing.assert_almost_equal(
        masker, 0.5 * np.sin(2 * np.pi * 1100 * np.arange(length) / 48000))

  def test_as_distorter_bandlimited_noise_masker(self):
    sine = distorted_loops.signal_beep(1000.)  # Original signal

    width_hz = 200
    freq_hz = 1000
    stdev = 0.0
    fs = 48000
    nfft = 524288  # A large power of 2 for the number of fft points

    bandlimited_noise_masker_distortion = distorted_loops.as_distorter(
        distorted_loops.gen_bandlimited_noise_masker,
        width_hz=width_hz,
        freq_hz=freq_hz,
        stdev=stdev,
        rng=self.rng)

    distorted_sine = bandlimited_noise_masker_distortion(
        sine, 1.0)

    bandlimited_noise = distorted_sine - sine

    (_, psd) = scipy.signal.periodogram(
        bandlimited_noise,
        scaling='density',
        return_onesided=False,
        nfft=nfft)
    low = (freq_hz - width_hz // 2) * nfft // fs
    high = (freq_hz + width_hz // 2) * nfft // fs

    # Check that calculated noise power is in [0.9, 1.1]
    power = np.mean(psd[low:high])
    np.testing.assert_allclose(power, 1., rtol=0, atol=0.1)

  def test_as_distorter_white_noise_masker(self):
    sine = distorted_loops.signal_beep(1000.)  # Original signal

    stdev = 0.0
    nfft = 524288  # A large power of 2 for the number of fft points

    white_noise_masker_distortion = distorted_loops.as_distorter(
        distorted_loops.gen_white_noise_masker, stdev=stdev, rng=self.rng)

    distorted_sine = white_noise_masker_distortion(sine, 1.0)

    white_noise = distorted_sine - sine

    (_, psd) = scipy.signal.periodogram(
        white_noise,
        scaling='density',
        return_onesided=False,
        nfft=nfft)

    # Check that calculated noise power is in [0.9, 1.1]
    power = np.mean(psd[:])
    np.testing.assert_allclose(power, 1., rtol=0, atol=0.1)

  def test_as_distorter_sine_masker(self):
    sine = distorted_loops.signal_beep(1000.)  # Original signal

    amount = 0.5
    freq_hz = 1100.0
    sample_rate = 48000

    sine_masker_distortion = distorted_loops.as_distorter(
        distorted_loops.gen_sine_masker,
        amount=amount,
        freq_hz=freq_hz,
        sample_rate=sample_rate)

    distorted_sine = sine_masker_distortion(
        sine, 1.0)

    masker = distorted_sine - sine

    np.testing.assert_almost_equal(
        masker, 0.5 * np.sin(2 * np.pi * 1100 * np.arange(len(sine)) / 48000))

  def test_distorted_loops(self):
    signal = np.ones((3,))
    distortion = distorted_loops.intensity_distortion
    amount = 20.
    distorted_signal = 10. * signal

    no_dist_loop, dist_loop = distorted_loops.distorted_loops(
        signal, distortion, amount)

    pause = np.zeros((4800,))
    no_dist_loop_expected = np.concatenate(
        [signal, np.tile(np.concatenate([pause, signal]), 7)])

    np.testing.assert_almost_equal(no_dist_loop, no_dist_loop_expected)

    dist_loop_expected = np.concatenate([
        signal,
        np.tile(np.concatenate([pause, signal]), 3),
        np.concatenate([pause, distorted_signal]),
        np.tile(np.concatenate([pause, signal]), 3)
    ])
    np.testing.assert_almost_equal(dist_loop, dist_loop_expected)

if __name__ == '__main__':
  googletest.main()
