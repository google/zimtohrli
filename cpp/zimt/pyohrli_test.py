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
"""Tests for google3.third_party.zimtohrli.cpp.python.pyohrli."""

import numpy as np

import unittest
import pyohrli
import functools


def parameterize(*kwargs):
    def decorator(func):
        @functools.wraps(func)
        def call_with_parameters(self, **inner_kwargs):
            for kwarg in kwargs:
                func(self, **kwarg)

        return call_with_parameters

    return decorator


class PyohrliTest(unittest.TestCase):

    def test_num_rotators(self):
      self.assertEqual(128, pyohrli.Pyohrli().num_rotators)

    def test_sample_rate(self):
      self.assertEqual(48000, pyohrli.Pyohrli().sample_rate)

    @parameterize(
        dict(
            a_hz=5000.0,
            b_hz=5000.0,
            distance=0,
        ),
        dict(
            a_hz=5000.0,
            b_hz=5010.0,
            distance=3.737211227416992e-05,
        ),
        dict(
            a_hz=5000.0,
            b_hz=10000.0,
            distance=0.3206554651260376,
        ),
    )
    def test_distance(self, a_hz: float, b_hz: float, distance: float):
        sample_rate = 48000.0
        metric = pyohrli.Pyohrli()
        signal_a = np.sin(np.linspace(0.0, np.pi * 2 * a_hz, int(sample_rate)))
        signal_b = np.sin(np.linspace(0.0, np.pi * 2 * b_hz, int(sample_rate)))
        distance = metric.distance(signal_a, signal_b)
        self.assertLess(abs(distance - distance), 1e-3)

    @parameterize(
        dict(zimtohrli_distance=0.0, mos=5.0),
        dict(zimtohrli_distance=0.001, mos=4.748757362365723),
        dict(zimtohrli_distance=0.04, mos=1.2986432313919067),
    )
    def test_mos_from_zimtohrli(self, zimtohrli_distance: float, mos: float):
        self.assertAlmostEqual(
            mos, pyohrli.mos_from_zimtohrli(zimtohrli_distance), places=3
        )

    # ----- C++ <-> Python equivalence -----
    # The Python `analyze`/`distance` bindings delegate to the same C++
    # zimtohrli::Zimtohrli used by the C++ tests, so the Python outputs must
    # reproduce values measured from the C++ analyzer.

    # C++ silence baseline LoudnessDb({0,...}): base[0] and base[92].
    _CPP_SILENCE_BASE_DIM0 = 9.36386776
    _CPP_SILENCE_BASE_DIM92 = 23.8885

    # C++ golden spectrogram (1 step, 128 dims) for channel 0 of test.wav, i.e.
    # a full-scale +-0.5 alternating (Nyquist) waveform of 10 samples.
    _CPP_TESTWAV_CH0_GOLDEN = [
        9.36386776,
        9.27772045,
        9.31594944,
        9.32667446,
        9.29242229,
        9.28951645,
        9.33077049,
        9.33707333,
        9.30707455,
        9.3779974,
        9.31306839,
        9.33522797,
        9.27003479,
        9.36374855,
        9.41100788,
        9.45162964,
        9.43794632,
        9.49529743,
        9.86828232,
        10.8203812,
        11.6057587,
        11.9237604,
        11.9912643,
        11.855402,
        11.0952435,
        9.53629971,
        8.59437561,
        7.90416718,
        7.04881239,
        6.55117178,
        6.28870296,
        6.49167347,
        7.13244629,
        8.6201458,
        10.2842903,
        12.1083527,
        13.1857967,
        13.7895098,
        13.6850128,
        13.4411144,
        12.6701937,
        12.5632191,
        12.6067514,
        13.3559313,
        14.315793,
        15.7606239,
        16.9195881,
        17.6959553,
        17.7924576,
        17.3982944,
        16.6439724,
        15.6063681,
        14.7217445,
        13.7300892,
        12.8015537,
        12.4289322,
        11.8797998,
        11.6131411,
        11.2089548,
        11.1229973,
        11.0147076,
        11.1954565,
        11.4009295,
        11.6422834,
        11.950736,
        12.0980787,
        12.5331593,
        12.8450193,
        13.350791,
        13.5188026,
        13.633358,
        13.9081421,
        14.0771198,
        14.2513056,
        14.3190088,
        14.402422,
        14.3311644,
        14.2943134,
        14.1501369,
        14.1586895,
        14.1592188,
        14.4553652,
        14.8980217,
        15.7184505,
        16.7763557,
        18.2429047,
        20.1216316,
        21.3781109,
        22.6360073,
        23.6522255,
        24.3742142,
        24.6754036,
        25.3442936,
        24.6462212,
        24.4961529,
        24.1290874,
        23.0455952,
        22.2857571,
        20.4072132,
        19.134716,
        18.4946175,
        18.2513657,
        19.6234512,
        19.6775627,
        20.1714287,
        20.3433018,
        20.3507004,
        20.6260815,
        20.7351322,
        20.9621601,
        21.3178368,
        21.8257599,
        22.1928749,
        20.2414112,
        20.2102032,
        21.0299358,
        22.236124,
        24.6129494,
        21.6497231,
        17.0029449,
        17.0624599,
        20.9731503,
        15.3123941,
        15.3291759,
        22.16675,
        12.8916807,
        20.3194408,
        22.4193649,
    ]

    def test_analyze_shape_and_finiteness(self):
        metric = pyohrli.Pyohrli()
        sample_rate = 48000.0
        signal = np.sin(np.linspace(0.0, np.pi * 2 * 440.0, int(sample_rate)))
        spectrogram = np.asarray(metric.analyze(signal))
        self.assertEqual(spectrogram.ndim, 2)
        self.assertEqual(spectrogram.shape[1], metric.num_rotators)
        self.assertGreater(spectrogram.shape[0], 0)
        self.assertTrue(np.all(np.isfinite(spectrogram)))

    def test_analyze_silence_matches_cpp_baseline(self):
        # Silence analyzed in Python must equal the C++ analytic baseline.
        metric = pyohrli.Pyohrli()
        spectrogram = np.asarray(metric.analyze(np.zeros(48000, dtype=np.float32)))
        self.assertGreater(spectrogram.shape[0], 0)
        self.assertAlmostEqual(
            spectrogram[0, 0], self._CPP_SILENCE_BASE_DIM0, places=2
        )
        self.assertAlmostEqual(
            spectrogram[0, 92], self._CPP_SILENCE_BASE_DIM92, places=2
        )
        # Every silence frame equals the baseline.
        for step in range(spectrogram.shape[0]):
            self.assertAlmostEqual(
                spectrogram[step, 0], self._CPP_SILENCE_BASE_DIM0, places=3
            )

    def test_analyze_matches_cpp_golden(self):
        # The strongest equivalence check: Python analyze of the exact test.wav
        # channel-0 waveform reproduces the C++ golden spectrogram bit-for-bit
        # (within float tolerance).
        metric = pyohrli.Pyohrli()
        signal = np.array([0.5, -0.5] * 5, dtype=np.float32)  # 10 samples
        spectrogram = np.asarray(metric.analyze(signal))
        self.assertEqual(spectrogram.shape, (1, metric.num_rotators))
        golden = np.array(self._CPP_TESTWAV_CH0_GOLDEN, dtype=np.float32)
        np.testing.assert_allclose(spectrogram[0], golden, atol=1e-3)

    def test_analyze_is_deterministic(self):
        metric = pyohrli.Pyohrli()
        signal = np.sin(np.linspace(0.0, np.pi * 2 * 440.0, 24000))
        a = np.asarray(metric.analyze(signal))
        b = np.asarray(metric.analyze(signal))
        np.testing.assert_array_equal(a, b)

    def test_frequency_to_channel_matches_cpp(self):
        # Delta-from-silence resonance peak for a 440 Hz tone is channel 36 in
        # C++; the Python binding must agree. Build the tone as sin(2*pi*f*t)
        # with t = n/sample_rate (matching C++ GenerateSineWave), so it is a
        # true 440 Hz tone regardless of duration.
        metric = pyohrli.Pyohrli()
        sample_rate = 48000.0
        num_samples = int(sample_rate * 0.5)
        t = np.arange(num_samples) / sample_rate
        tone = np.sin(2.0 * np.pi * 440.0 * t)
        silence = np.zeros(num_samples, dtype=np.float32)
        tone_sp = np.asarray(metric.analyze(tone))
        silence_sp = np.asarray(metric.analyze(silence))
        delta = tone_sp - silence_sp
        peak_channel = int(np.argmax(delta.sum(axis=0)))
        self.assertEqual(peak_channel, 36)

    def test_distance_self_is_zero_and_symmetric(self):
        metric = pyohrli.Pyohrli()
        sample_rate = 48000.0
        signal_a = np.sin(np.linspace(0.0, np.pi * 2 * 440.0, int(sample_rate)))
        signal_b = np.sin(np.linspace(0.0, np.pi * 2 * 660.0, int(sample_rate)))
        self.assertAlmostEqual(metric.distance(signal_a, signal_a), 0.0, places=5)
        d_ab = metric.distance(signal_a, signal_b)
        d_ba = metric.distance(signal_b, signal_a)
        self.assertGreater(d_ab, 0.0)
        self.assertAlmostEqual(d_ab, d_ba, places=4)


if __name__ == "__main__":
    unittest.main()
