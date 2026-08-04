// Copyright 2026 The Zimtohrli Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "zimt/stream_test_golden.h"
#include "zimt/zimtohrli.h"

namespace zimtohrli {
namespace {

constexpr float kPi = 3.14159265358979323846f;

std::vector<float> GenerateSineWave(float frequency, float duration_seconds,
                                    float amplitude = 0.5f,
                                    float sample_rate = 48000.0f) {
  size_t num_samples = static_cast<size_t>(duration_seconds * sample_rate);
  std::vector<float> signal(num_samples);
  for (size_t i = 0; i < num_samples; ++i) {
    float t = static_cast<float>(i) / sample_rate;
    signal[i] = amplitude * std::sin(2.0f * kPi * frequency * t);
  }
  return signal;
}

Spectrogram ProcessInChunks(const Zimtohrli& zimtohrli,
                            Span<const float> signal, size_t chunk_size) {
  ChunkedAnalyzer stream(zimtohrli.samples_per_perceptual_block);
  std::vector<float> frames;
  for (size_t offset = 0; offset < signal.size; offset += chunk_size) {
    size_t current_chunk = std::min(chunk_size, signal.size - offset);
    stream.Process(Span<const float>(&signal.data[offset], current_chunk),
                   frames);
  }
  stream.Flush(frames);
  return Spectrogram(stream.num_steps(), kNumRotators, std::move(frames));
}

Spectrogram ProcessInVariableChunks(const Zimtohrli& zimtohrli,
                                    Span<const float> signal,
                                    const std::vector<size_t>& chunk_sizes) {
  ChunkedAnalyzer stream(zimtohrli.samples_per_perceptual_block);
  std::vector<float> frames;
  size_t offset = 0;
  size_t chunk_idx = 0;
  while (offset < signal.size) {
    size_t chunk_size = chunk_sizes[chunk_idx % chunk_sizes.size()];
    size_t current_chunk = std::min(chunk_size, signal.size - offset);
    stream.Process(Span<const float>(&signal.data[offset], current_chunk),
                   frames);
    offset += current_chunk;
    chunk_idx++;
  }
  stream.Flush(frames);
  return Spectrogram(stream.num_steps(), kNumRotators, std::move(frames));
}

// ---------------------------------------------------------------------------
// Test helpers.
//
// Design principle: every correctness property is checked on (a) single-pass
// Zimtohrli::Analyze, (b) chunked ChunkedAnalyzer, (c) that they agree, PLUS an
// INDEPENDENT ground truth (analytic silence baseline, filterbank resonance
// prediction, amplitude/frequency invariants, decay, finiteness). Parity alone
// is never sufficient.
// ---------------------------------------------------------------------------

// The exact silence output frame: LoudnessDb applied to an all-zero energy
// vector. Zero input keeps every accumulator at 0, so every emitted silence
// frame equals this deterministic 128-vector. Produced by calling LoudnessDb
// directly (an anonymous-namespace inline in the included header), bypassing
// the FIR/resonator/rotator machinery, hence independent of it.
std::vector<float> SilenceBaselineFrame() {
  std::vector<float> b(kNumRotators, 0.0f);
  LoudnessDb(b.data());
  return b;
}

// Deep-copies a spectrogram. The energy-rescaling distance helpers mutate their
// inputs, so callers clone first to keep a shared source spectrogram intact.
Spectrogram CloneSpectrogram(const Spectrogram& src) {
  Spectrogram clone(src.num_steps, src.num_dims);
  std::copy(src.values.get(), src.values.get() + src.size(),
            clone.values.get());
  return clone;
}

std::vector<float> GenerateSilence(size_t n) {
  return std::vector<float>(n, 0.0f);
}

std::vector<float> GenerateDC(float value, size_t n) {
  return std::vector<float>(n, value);
}

// Seeded white noise, uniform in [-amp, amp]. std::mt19937 is portable and
// deterministic, so the signal (and thus the spectrogram) is reproducible.
std::vector<float> GenerateWhiteNoise(uint32_t seed, size_t n,
                                      float amp = 0.5f) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-amp, amp);
  std::vector<float> s(n);
  for (size_t i = 0; i < n; ++i) s[i] = dist(rng);
  return s;
}

std::vector<float> GenerateImpulse(size_t n, size_t pos, float amp = 1.0f) {
  std::vector<float> s(n, 0.0f);
  if (pos < n) s[pos] = amp;
  return s;
}

std::vector<float> GenerateMultiTone(const std::vector<float>& freqs,
                                     const std::vector<float>& amps, float dur,
                                     float sr = 48000.0f) {
  size_t n = static_cast<size_t>(dur * sr);
  std::vector<float> s(n, 0.0f);
  for (size_t i = 0; i < n; ++i) {
    float t = static_cast<float>(i) / sr;
    for (size_t j = 0; j < freqs.size(); ++j) {
      s[i] += amps[j] * std::sin(2.0f * kPi * freqs[j] * t);
    }
  }
  return s;
}

// Linear chirp from f0 to f1 over dur seconds (instantaneous freq = f0 + k*t).
std::vector<float> GenerateChirp(float f0, float f1, float dur,
                                 float amp = 0.5f, float sr = 48000.0f) {
  size_t n = static_cast<size_t>(dur * sr);
  std::vector<float> s(n);
  float k = (f1 - f0) / dur;
  for (size_t i = 0; i < n; ++i) {
    float t = static_cast<float>(i) / sr;
    float phase = 2.0f * kPi * (f0 * t + 0.5f * k * t * t);
    s[i] = amp * std::sin(phase);
  }
  return s;
}

// argmax over channels of (frame - silence baseline) at a given step.
size_t PeakChannelExcess(const Spectrogram& s, size_t step,
                         const std::vector<float>& base) {
  size_t peak = 0;
  float best = -1e30f;
  for (size_t k = 0; k < s.num_dims; ++k) {
    float ex = s[step][k] - base[k];
    if (ex > best) {
      best = ex;
      peak = k;
    }
  }
  return peak;
}

// Delta-from-silence peak channel: the channel maximizing the summed excess of
// a tone spectrogram over a silence spectrogram of identical length. Summing
// over all steps averages out onset/flush transients. This isolates the true
// resonance channel, removing the frequency-independent kMul/noise weighting
// artifact that dominates the raw-dB argmax.
size_t DeltaPeakChannel(const Spectrogram& tone, const Spectrogram& silence) {
  const size_t dims = tone.num_dims;
  std::vector<double> acc(dims, 0.0);
  for (size_t step = 0; step < tone.num_steps; ++step) {
    for (size_t k = 0; k < dims; ++k) {
      acc[k] += static_cast<double>(tone[step][k]) - silence[step][k];
    }
  }
  size_t peak = 0;
  double best = -1e300;
  for (size_t k = 0; k < dims; ++k) {
    if (acc[k] > best) {
      best = acc[k];
      peak = k;
    }
  }
  return peak;
}

// Independent ground-truth prediction of the resonance channel for a pure tone:
// the filterbank channel whose center frequency Freq(i) is closest to freq.
// Uses the header's own Freq() table, so it is independent of the analyzer's
// numerical pipeline.
size_t PredictedResonanceChannel(float freq) {
  size_t best = 0;
  float best_dist = 1e30f;
  for (int i = 0; i < kNumRotators; ++i) {
    float d = std::abs(Freq(i) - freq);
    if (d < best_dist) {
      best_dist = d;
      best = static_cast<size_t>(i);
    }
  }
  return best;
}

bool AllFinite(const Spectrogram& s) {
  for (size_t i = 0; i < s.size(); ++i) {
    if (!std::isfinite(s.values[i])) return false;
  }
  return true;
}

// argmax over channels of the raw spectrogram value at a step. NOTE: for
// moderate tones this is dominated by the kMul loudness weighting (max at
// channel 92) and is therefore frequency-independent; use DeltaPeakChannel /
// PeakChannelExcess for true frequency->channel localization.
size_t RawPeakChannel(const Spectrogram& s, size_t step) {
  size_t peak = 0;
  float best = -1e30f;
  for (size_t k = 0; k < s.num_dims; ++k) {
    if (s[step][k] > best) {
      best = s[step][k];
      peak = k;
    }
  }
  return peak;
}

void ExpectSpectrogramsNear(const Spectrogram& a, const Spectrogram& b,
                            float tol) {
  ASSERT_EQ(a.num_steps, b.num_steps);
  ASSERT_EQ(a.num_dims, b.num_dims);
  for (size_t s = 0; s < a.num_steps; ++s) {
    for (size_t d = 0; d < a.num_dims; ++d) {
      EXPECT_NEAR(a[s][d], b[s][d], tol) << "step " << s << " dim " << d;
    }
  }
}

// Equivalence across various fixed chunk sizes.
TEST(StreamTest, SpectrogramMatchesAcrossVariousChunkSizes) {
  Zimtohrli zimtohrli;
  auto signal = GenerateSineWave(440.0f, 1.0f);

  Spectrogram spec_single_pass = zimtohrli.Analyze(signal);

  std::vector<size_t> chunk_sizes = {1,  2,   5,   10,  16,  31,   32,
                                     64, 100, 256, 512, 564, 1024, 4096};
  for (size_t chunk_size : chunk_sizes) {
    Spectrogram spec_chunked = ProcessInChunks(zimtohrli, signal, chunk_size);

    ASSERT_EQ(spec_single_pass.num_steps, spec_chunked.num_steps)
        << "Mismatch in step count for chunk size " << chunk_size;
    ASSERT_EQ(spec_single_pass.num_dims, spec_chunked.num_dims)
        << "Mismatch in dim count for chunk size " << chunk_size;

    for (size_t step = 0; step < spec_single_pass.num_steps; ++step) {
      for (size_t dim = 0; dim < spec_single_pass.num_dims; ++dim) {
        EXPECT_NEAR(spec_single_pass[step][dim], spec_chunked[step][dim], 1e-4f)
            << "Mismatch at step " << step << ", dim " << dim
            << " for chunk size " << chunk_size;
      }
    }
  }
}

// Variable / pseudo-random chunk sizes.
TEST(StreamTest, SpectrogramMatchesForVariableChunkSizes) {
  Zimtohrli zimtohrli;
  auto signal = GenerateSineWave(440.0f, 1.0f);

  Spectrogram spec_single_pass = zimtohrli.Analyze(signal);

  std::vector<size_t> variable_chunks = {1,    10, 100, 5,  564, 300,
                                         1000, 20, 48,  32, 7,   13};
  Spectrogram spec_chunked =
      ProcessInVariableChunks(zimtohrli, signal, variable_chunks);

  ASSERT_EQ(spec_single_pass.num_steps, spec_chunked.num_steps);
  ASSERT_EQ(spec_single_pass.num_dims, spec_chunked.num_dims);

  for (size_t step = 0; step < spec_single_pass.num_steps; ++step) {
    for (size_t dim = 0; dim < spec_single_pass.num_dims; ++dim) {
      EXPECT_NEAR(spec_single_pass[step][dim], spec_chunked[step][dim], 1e-4f)
          << "Mismatch at step " << step << ", dim " << dim;
    }
  }
}

// Perceptual Distance equivalence between single-pass and chunked processing.
TEST(StreamTest, DistanceWithoutDtwMatchesBetweenChunkedAndSinglePass) {
  Zimtohrli zimtohrli;
  auto signal_ref = GenerateSineWave(440.0f, 1.0f, 0.5f);
  auto signal_test = GenerateSineWave(440.0f, 1.0f, 0.4f);

  Spectrogram ref_single = zimtohrli.Analyze(signal_ref);
  Spectrogram test_single = zimtohrli.Analyze(signal_test);
  float distance_single = zimtohrli.DistanceWithoutDtw(ref_single, test_single);

  Spectrogram ref_chunked = ProcessInChunks(zimtohrli, signal_ref, 512);
  Spectrogram test_chunked = ProcessInChunks(zimtohrli, signal_test, 1024);
  float distance_chunked =
      zimtohrli.DistanceWithoutDtw(ref_chunked, test_chunked);

  EXPECT_NEAR(distance_single, distance_chunked, 1e-5f);
}

// Boundary condition: Empty signal (0 samples).
TEST(StreamTest, HandlesEmptySignalBoundaryCondition) {
  Zimtohrli zimtohrli;
  std::vector<float> empty_signal;

  Spectrogram spec_single = zimtohrli.Analyze(empty_signal);
  Spectrogram spec_chunked = ProcessInChunks(zimtohrli, empty_signal, 100);

  EXPECT_EQ(spec_single.num_steps, 0);
  EXPECT_EQ(spec_chunked.num_steps, 0);
}

// Boundary condition: Inputs shorter than FIR kernel (N < 32 samples).
TEST(StreamTest, HandlesSubKernelSignalsSafely) {
  Zimtohrli zimtohrli;
  for (size_t num_samples : {1, 5, 10, 31}) {
    auto short_signal =
        GenerateSineWave(440.0f, static_cast<float>(num_samples) / 48000.0f);
    short_signal.resize(num_samples);

    Spectrogram spec_single = zimtohrli.Analyze(short_signal);
    Spectrogram spec_chunked_1 = ProcessInChunks(zimtohrli, short_signal, 1);
    Spectrogram spec_chunked_10 = ProcessInChunks(zimtohrli, short_signal, 10);

    EXPECT_EQ(spec_single.num_steps, spec_chunked_1.num_steps);
    EXPECT_EQ(spec_single.num_steps, spec_chunked_10.num_steps);
  }
}

// Boundary condition: Exact FIR kernel length (32 samples).
TEST(StreamTest, HandlesExactKernelLengthBoundaryCondition) {
  Zimtohrli zimtohrli;
  auto signal = GenerateSineWave(440.0f, 32.0f / 48000.0f);
  signal.resize(32);

  Spectrogram spec_single = zimtohrli.Analyze(signal);
  Spectrogram spec_chunked_1 = ProcessInChunks(zimtohrli, signal, 1);
  Spectrogram spec_chunked_7 = ProcessInChunks(zimtohrli, signal, 7);

  EXPECT_EQ(spec_single.num_steps, spec_chunked_1.num_steps);
  EXPECT_EQ(spec_single.num_steps, spec_chunked_7.num_steps);

  for (size_t step = 0; step < spec_single.num_steps; ++step) {
    for (size_t dim = 0; dim < spec_single.num_dims; ++dim) {
      EXPECT_NEAR(spec_single[step][dim], spec_chunked_1[step][dim], 1e-4f);
      EXPECT_NEAR(spec_single[step][dim], spec_chunked_7[step][dim], 1e-4f);
    }
  }
}

// Boundary condition: Exact downsample block boundaries (e.g. 564 + 32, 2 *
// 564 + 32 samples).
TEST(StreamTest, HandlesExactDownsampleBoundaries) {
  Zimtohrli zimtohrli;
  for (size_t blocks : {1, 2, 5}) {
    size_t num_samples = blocks * 564 + 32;
    auto signal =
        GenerateSineWave(440.0f, static_cast<float>(num_samples) / 48000.0f);
    signal.resize(num_samples);

    Spectrogram spec_single = zimtohrli.Analyze(signal);
    Spectrogram spec_chunked = ProcessInChunks(zimtohrli, signal, 100);

    ASSERT_EQ(spec_single.num_steps, spec_chunked.num_steps);
    for (size_t step = 0; step < spec_single.num_steps; ++step) {
      for (size_t dim = 0; dim < spec_single.num_dims; ++dim) {
        EXPECT_NEAR(spec_single[step][dim], spec_chunked[step][dim], 1e-4f);
      }
    }
  }
}

// Deterministic output verification: validates spectrogram dimensions,
// peak energy channel localization near the 440 Hz frequency bin (channel 92),
// and non-zero positive energy values.
TEST(StreamTest, SpectrogramGoldenValues) {
  Zimtohrli zimtohrli;
  auto signal = GenerateSineWave(440.0f, 0.1f);  // 4800 samples at 48kHz
  Spectrogram spec = zimtohrli.Analyze(signal);

  ASSERT_EQ(spec.num_steps, 9);
  ASSERT_EQ(spec.num_dims, 128);

  // For a 440 Hz sine wave, the peak energy in the steady-state frames (e.g.
  // step 4) localizes around channel 92.
  float peak_val = -1000.0f;
  size_t peak_dim = 0;
  for (size_t dim = 0; dim < spec.num_dims; ++dim) {
    if (spec[4][dim] > peak_val) {
      peak_val = spec[4][dim];
      peak_dim = dim;
    }
  }
  EXPECT_EQ(peak_dim, 92);
  EXPECT_GT(peak_val, 0.0f);
}

// Verifies ChunkedAnalyzer::num_steps() matches SpectrogramSteps() across
// boundary signal lengths (0, 1, 31, 32, 563, 564, 565, 1128).
TEST(StreamTest, StepCountMatchesSpectrogramStepsForEdgeCases) {
  Zimtohrli zimtohrli;
  for (size_t num_samples : {0, 1, 31, 32, 563, 564, 565, 1128}) {
    auto signal =
        GenerateSineWave(440.0f, static_cast<float>(num_samples) / 48000.0f);
    signal.resize(num_samples);

    Spectrogram spec = zimtohrli.Analyze(signal);
    size_t expected_steps = zimtohrli.SpectrogramSteps(num_samples);

    EXPECT_EQ(spec.num_steps, expected_steps)
        << "Step count mismatch for " << num_samples << " samples: got "
        << spec.num_steps << ", expected " << expected_steps;

    if (num_samples > 0) {
      Spectrogram spec_chunked = ProcessInChunks(zimtohrli, signal, 100);
      EXPECT_EQ(spec_chunked.num_steps, expected_steps)
          << "Chunked step count mismatch for " << num_samples
          << " samples: got " << spec_chunked.num_steps << ", expected "
          << expected_steps;
    }
  }
}

// Verifies Distance() (with DTW) equivalence between single-pass and
// chunked processing.
TEST(StreamTest, DistanceWithDtwMatchesBetweenChunkedAndSinglePass) {
  Zimtohrli zimtohrli;
  auto signal_ref = GenerateSineWave(440.0f, 1.0f, 0.5f);
  auto signal_test = GenerateSineWave(450.0f, 1.0f, 0.4f);

  Spectrogram ref_single = zimtohrli.Analyze(signal_ref);
  Spectrogram test_single = zimtohrli.Analyze(signal_test);
  float distance_single = zimtohrli.Distance(ref_single, test_single);

  Spectrogram ref_chunked = ProcessInChunks(zimtohrli, signal_ref, 512);
  Spectrogram test_chunked = ProcessInChunks(zimtohrli, signal_test, 1024);
  float distance_chunked = zimtohrli.Distance(ref_chunked, test_chunked);

  EXPECT_NEAR(distance_single, distance_chunked, 1e-5f);
}

// ===========================================================================
// Spectrogram correctness and signal invariant tests.
// ===========================================================================

// Buffer compaction path (buffer_head_ > 4096). Long signal fed in small
// chunks (<< 4096) repeatedly crosses the compaction threshold mid-stream.
TEST(StreamTest, BufferCompactionLongSignalSmallChunks) {
  Zimtohrli z;
  std::vector<float> base = SilenceBaselineFrame();
  auto signal = GenerateSineWave(440.0f, 1.0f, 0.5f);  // 48000 samples
  Spectrogram single = z.Analyze(signal);
  EXPECT_TRUE(AllFinite(single));
  // Independent ground truth: raw argmax is the freq-independent kMul artifact
  // (channel 92); the resonance (delta-from-silence) peak is channel 36; the
  // excess at that peak is large.
  EXPECT_EQ(RawPeakChannel(single, 4), 92u);
  EXPECT_EQ(PeakChannelExcess(single, 4, base), 36u);
  EXPECT_GT(single[4][36] - base[36], 5.0f);
  for (size_t chunk : {size_t(7), size_t(100), size_t(563)}) {
    Spectrogram chunked = ProcessInChunks(z, signal, chunk);
    EXPECT_TRUE(AllFinite(chunked)) << "chunk " << chunk;
    EXPECT_EQ(PeakChannelExcess(chunked, 4, base), 36u) << "chunk " << chunk;
    ExpectSpectrogramsNear(single, chunked, 1e-4f);
  }
}

// 1-sample-at-a-time streaming over a long signal.
TEST(StreamTest, SingleSampleStreamingMatchesAndLocalizesTone) {
  Zimtohrli z;
  std::vector<float> base = SilenceBaselineFrame();
  auto signal = GenerateSineWave(440.0f, 0.5f, 0.5f);  // 24000 samples
  Spectrogram single = z.Analyze(signal);
  Spectrogram chunked = ProcessInChunks(z, signal, 1);
  EXPECT_TRUE(AllFinite(single));
  EXPECT_TRUE(AllFinite(chunked));
  // resonance peak localized identically on both paths.
  EXPECT_EQ(PeakChannelExcess(single, 4, base), 36u);
  EXPECT_EQ(PeakChannelExcess(chunked, 4, base), 36u);
  ExpectSpectrogramsNear(single, chunked, 1e-4f);
}

// Long-signal (10 s) numerical stability: no drift or blow-up.
TEST(StreamTest, LongSignalNumericalStability) {
  Zimtohrli z;
  std::vector<float> base = SilenceBaselineFrame();
  auto signal = GenerateSineWave(440.0f, 10.0f, 0.5f);  // 480000 samples
  Spectrogram single = z.Analyze(signal);
  Spectrogram chunked = ProcessInChunks(z, signal, 2048);
  EXPECT_EQ(single.num_steps, z.SpectrogramSteps(signal.size()));  // 852
  EXPECT_TRUE(AllFinite(single));
  EXPECT_TRUE(AllFinite(chunked));
  // Bounded magnitude: LoudnessDb is log-scale (measured max ~40); anything
  // near float-max would indicate phasor drift/blowup.
  EXPECT_LT(single.max(), 50.0f);
  // Anti-drift: the resonance peak channel is stable across the entire 10 s
  // (RenormalizePhasors keeps the filterbank stationary).
  EXPECT_EQ(PeakChannelExcess(single, 100, base), 36u);
  EXPECT_EQ(PeakChannelExcess(single, 400, base), 36u);
  EXPECT_EQ(PeakChannelExcess(single, 800, base), 36u);
  ExpectSpectrogramsNear(single, chunked, 1e-3f);
}

// Silence -> exact analytic baseline (the strongest absolute-correctness
// check: validates the whole FIR->resonator->rotator->LoudnessDb chain against
// the analytic zero-energy result, independent of parity).
TEST(StreamTest, SilenceProducesExactBaseline) {
  Zimtohrli z;
  std::vector<float> base = SilenceBaselineFrame();
  auto signal = GenerateSilence(48000);
  Spectrogram single = z.Analyze(signal);
  Spectrogram chunked = ProcessInChunks(z, signal, 100);
  ASSERT_GT(single.num_steps, 0u);
  for (size_t s = 0; s < single.num_steps; ++s) {
    for (size_t k = 0; k < single.num_dims; ++k) {
      EXPECT_NEAR(single[s][k], base[k], 1e-4f) << "step " << s << " dim " << k;
      EXPECT_NEAR(chunked[s][k], base[k], 1e-4f)
          << "step " << s << " dim " << k;
    }
  }
  EXPECT_NEAR(single[0][0], 9.36387f, 1e-2f);  // hardcoded analytic sanity
  ExpectSpectrogramsNear(single, chunked, 1e-4f);
}

// DC signal -> stable, finite, deterministic; energy concentrates in the
// lowest channel (independent physical expectation).
TEST(StreamTest, DCSignalStableLowChannelDominant) {
  Zimtohrli z;
  std::vector<float> base = SilenceBaselineFrame();
  auto signal = GenerateDC(0.5f, 48000);
  Spectrogram single = z.Analyze(signal);
  Spectrogram single_again = z.Analyze(signal);
  Spectrogram chunked = ProcessInChunks(z, signal, 512);
  EXPECT_TRUE(AllFinite(single));
  EXPECT_TRUE(AllFinite(chunked));
  ExpectSpectrogramsNear(single, single_again, 0.0f);  // determinism (exact)
  // DC excites the lowest channel most (measured: channel 0).
  EXPECT_EQ(PeakChannelExcess(single, 10, base), 0u);
  EXPECT_GT(single[10][0] - base[0], 1.0f);
  ExpectSpectrogramsNear(single, chunked, 1e-4f);
}

// White noise -> deterministic (seeded), finite, and broadband: it excites
// essentially every filterbank channel well above the silence baseline (unlike
// silence, which excites none).
TEST(StreamTest, WhiteNoiseDeterministicAndBroadband) {
  Zimtohrli z;
  std::vector<float> base = SilenceBaselineFrame();
  auto noise = GenerateWhiteNoise(12345u, 48000, 0.5f);
  auto noise_again = GenerateWhiteNoise(12345u, 48000, 0.5f);
  Spectrogram single = z.Analyze(noise);
  Spectrogram single_again = z.Analyze(noise_again);
  Spectrogram chunked = ProcessInChunks(z, noise, 256);
  EXPECT_TRUE(AllFinite(single));
  EXPECT_TRUE(AllFinite(chunked));
  // Determinism: identical seed -> bit-identical spectrogram.
  ExpectSpectrogramsNear(single, single_again, 0.0f);
  ExpectSpectrogramsNear(single, chunked, 1e-4f);
  // Broadband excitation at a mid-stream step (avoiding onset/flush
  // transients): (essentially) every channel is well above the silence
  // baseline.
  ASSERT_GT(single.num_steps, 60u);
  size_t excited = 0;
  for (size_t k = 0; k < single.num_dims; ++k) {
    if (single[40][k] - base[k] > 0.01f) excited++;
  }
  EXPECT_GE(excited, 120u);  // measured: 128/128
  // Independent contrast: silence excites no channel at the same step.
  auto silence = GenerateSilence(48000);
  Spectrogram sil = z.Analyze(silence);
  size_t sil_excited = 0;
  for (size_t k = 0; k < sil.num_dims; ++k) {
    if (sil[40][k] - base[k] > 0.01f) sil_excited++;
  }
  EXPECT_EQ(sil_excited, 0u);
}

// Impulse -> broadband early excitation that decays (near-)monotonically
// to the silence baseline (leaky accumulators/resonator).
TEST(StreamTest, ImpulseResponseDecaysToBaseline) {
  Zimtohrli z;
  std::vector<float> base = SilenceBaselineFrame();
  auto signal = GenerateImpulse(48000, 0, 1.0f);
  Spectrogram single = z.Analyze(signal);
  Spectrogram chunked = ProcessInChunks(z, signal, 100);
  EXPECT_TRUE(AllFinite(single));
  EXPECT_TRUE(AllFinite(chunked));
  auto total_excess = [&](const Spectrogram& sp, size_t step) {
    double t = 0.0;
    for (size_t k = 0; k < sp.num_dims; ++k) t += sp[step][k] - base[k];
    return t;
  };
  double e0 = total_excess(single, 0);
  double e1 = total_excess(single, 1);
  double e2 = total_excess(single, 2);
  double e3 = total_excess(single, 3);
  EXPECT_GT(e0, 100.0);  // strong early excitation
  EXPECT_GT(e0, e1);
  EXPECT_GT(e1, e2);
  EXPECT_GT(e2, e3);
  EXPECT_LT(total_excess(single, 10), 0.1);                    // decayed
  EXPECT_LT(total_excess(single, single.num_steps - 2), 0.1);  // baseline
  ExpectSpectrogramsNear(single, chunked, 1e-4f);
}

// Multi-tone -> both tones' resonance channels are present in the combined
// spectrogram (self-contained, mapping-independent).
TEST(StreamTest, MultiToneShowsBothTonePeaks) {
  Zimtohrli z;
  std::vector<float> base = SilenceBaselineFrame();
  auto solo1 = GenerateSineWave(440.0f, 1.0f, 0.4f);
  auto solo2 = GenerateSineWave(3000.0f, 1.0f, 0.4f);
  auto combo = GenerateMultiTone({440.0f, 3000.0f}, {0.4f, 0.4f}, 1.0f);
  auto silence = GenerateSilence(solo1.size());
  Spectrogram s1 = z.Analyze(solo1);
  Spectrogram s2 = z.Analyze(solo2);
  Spectrogram sc = z.Analyze(combo);
  Spectrogram sil = z.Analyze(silence);
  size_t p1 = DeltaPeakChannel(s1, sil);
  size_t p2 = DeltaPeakChannel(s2, sil);
  EXPECT_NE(p1, p2);  // the two tones resonate in different channels
  const size_t step = 4;
  float solo1_ex = s1[step][p1] - base[p1];
  float solo2_ex = s2[step][p2] - base[p2];
  EXPECT_GT(solo1_ex, 1.0f);
  EXPECT_GT(solo2_ex, 1.0f);
  // Both tones' signatures survive in the combined signal.
  EXPECT_GT(sc[step][p1] - base[p1], 0.4f * solo1_ex);
  EXPECT_GT(sc[step][p2] - base[p2], 0.4f * solo2_ex);
  Spectrogram combo_chunked = ProcessInChunks(z, combo, 128);
  ExpectSpectrogramsNear(sc, combo_chunked, 1e-4f);
}

// Up-chirp -> resonance peak channel rises monotonically with the
// increasing instantaneous frequency (mapping-independent invariant).
TEST(StreamTest, ChirpPeakChannelIncreasesOverTime) {
  Zimtohrli z;
  std::vector<float> base = SilenceBaselineFrame();
  auto signal = GenerateChirp(300.0f, 6000.0f, 2.0f, 0.5f);
  Spectrogram single = z.Analyze(signal);
  Spectrogram chunked = ProcessInChunks(z, signal, 256);
  EXPECT_TRUE(AllFinite(single));
  EXPECT_TRUE(AllFinite(chunked));
  std::vector<size_t> steps = {20, 60, 100, 140};
  size_t prev = 0;
  for (size_t i = 0; i < steps.size(); ++i) {
    size_t pk = PeakChannelExcess(single, steps[i], base);
    if (i > 0) EXPECT_GT(pk, prev) << "step " << steps[i];
    prev = pk;
    // both paths agree on the peak channel at each step.
    EXPECT_EQ(PeakChannelExcess(chunked, steps[i], base), pk)
        << "step " << steps[i];
  }
  ExpectSpectrogramsNear(single, chunked, 1e-4f);
}

// Full-scale & out-of-range amplitudes. Note: Analyze does NOT clip — an
// amplitude of 5.0f produces larger frame energy than 0.5f, monotonically.
TEST(StreamTest, AmplitudeScalingInvariantsAndFiniteness) {
  Zimtohrli z;
  auto s_small = GenerateSineWave(440.0f, 0.1f, 0.05f);
  auto s_mid = GenerateSineWave(440.0f, 0.1f, 0.5f);
  auto s_full = GenerateSineWave(440.0f, 0.1f, 1.0f);
  auto s_large = GenerateSineWave(440.0f, 0.1f, 5.0f);

  Spectrogram sp_small = z.Analyze(s_small);
  Spectrogram sp_mid = z.Analyze(s_mid);
  Spectrogram sp_full = z.Analyze(s_full);
  Spectrogram sp_large = z.Analyze(s_large);

  EXPECT_TRUE(AllFinite(sp_small));
  EXPECT_TRUE(AllFinite(sp_mid));
  EXPECT_TRUE(AllFinite(sp_full));
  EXPECT_TRUE(AllFinite(sp_large));

  // Monotonicity of peak energy with respect to input amplitude:
  // sp_small < sp_mid < sp_full < sp_large at resonance channel 36.
  EXPECT_LT(sp_small[4][36], sp_mid[4][36]);
  EXPECT_LT(sp_mid[4][36], sp_full[4][36]);
  EXPECT_LT(sp_full[4][36], sp_large[4][36]);

  // Chunked versions also satisfy monotonicity and agree with single-pass.
  Spectrogram ck_large = ProcessInChunks(z, s_large, 100);
  EXPECT_TRUE(AllFinite(ck_large));
  ExpectSpectrogramsNear(sp_large, ck_large, 1e-4f);
}

// The pipeline never manufactures non-finite values from finite input.
TEST(StreamTest, FiniteControlProducesFiniteOutput) {
  Zimtohrli z;
  auto signal = GenerateSineWave(440.0f, 0.25f, 0.5f);
  Spectrogram single = z.Analyze(signal);
  Spectrogram chunked = ProcessInChunks(z, signal, 50);
  EXPECT_TRUE(AllFinite(single));
  EXPECT_TRUE(AllFinite(chunked));
}

// NaN input propagates. Leaky-persistent accumulators never clear NaN, so
// the final frame is fully NaN on both paths, and the finiteness pattern
// agrees.
TEST(StreamTest, NaNInputPropagatesToOutput) {
  Zimtohrli z;
  auto signal = GenerateSineWave(440.0f, 0.25f, 0.5f);  // 12000 samples
  signal[500] = std::numeric_limits<float>::quiet_NaN();

  Spectrogram single = z.Analyze(signal);
  Spectrogram chunked = ProcessInChunks(z, signal, 100);

  ASSERT_EQ(single.num_steps, chunked.num_steps);
  // Finiteness mask agrees across all steps and dimensions.
  for (size_t s = 0; s < single.num_steps; ++s) {
    for (size_t k = 0; k < single.num_dims; ++k) {
      EXPECT_EQ(std::isfinite(single[s][k]), std::isfinite(chunked[s][k]))
          << "step " << s << " dim " << k;
    }
  }
  // At the final step, the accumulators are completely NaN on both paths.
  for (size_t k = 0; k < single.num_dims; ++k) {
    EXPECT_TRUE(std::isnan(single[single.num_steps - 1][k]));
    EXPECT_TRUE(std::isnan(chunked[chunked.num_steps - 1][k]));
  }
}

// Inf input propagates as a non-finite value (empirically NaN via Inf-Inf
// in the FIR / rotators). Both paths produce identical non-finiteness.
TEST(StreamTest, InfInputPropagatesToOutput) {
  Zimtohrli z;
  auto signal = GenerateSineWave(440.0f, 0.25f, 0.5f);
  signal[500] = std::numeric_limits<float>::infinity();

  Spectrogram single = z.Analyze(signal);
  Spectrogram chunked = ProcessInChunks(z, signal, 100);

  ASSERT_EQ(single.num_steps, chunked.num_steps);
  for (size_t s = 0; s < single.num_steps; ++s) {
    for (size_t k = 0; k < single.num_dims; ++k) {
      EXPECT_EQ(std::isfinite(single[s][k]), std::isfinite(chunked[s][k]))
          << "step " << s << " dim " << k;
    }
  }
  for (size_t k = 0; k < single.num_dims; ++k) {
    EXPECT_FALSE(std::isfinite(single[single.num_steps - 1][k]));
    EXPECT_FALSE(std::isfinite(chunked[chunked.num_steps - 1][k]));
  }
}

// Frequency to channel mapping across the spectrum (100 Hz ... 15 kHz).
// Independent ground truth: resonance channel (maximizing excess over the
// silence baseline) tracks frequency monotonically and aligns with the
// filterbank's predicted resonance channels. Tested on BOTH paths.
TEST(StreamTest, FrequencyToChannelMappingMatchesResonance) {
  Zimtohrli z;
  std::vector<float> base = SilenceBaselineFrame();
  const std::vector<float> freqs = {100.0f,  250.0f,  440.0f,  1000.0f,
                                    2500.0f, 5000.0f, 10000.0f};
  std::vector<size_t> ch_single;
  std::vector<size_t> ch_chunked;
  for (float f : freqs) {
    auto sig = GenerateSineWave(f, 0.25f, 0.5f);  // 12000 samples -> ~21 steps
    Spectrogram sp_single = z.Analyze(sig);
    Spectrogram sp_chunked = ProcessInChunks(z, sig, 200);

    size_t d_single = PeakChannelExcess(sp_single, 10, base);
    size_t d_chunked = PeakChannelExcess(sp_chunked, 10, base);

    // Exact parity between single-pass and chunked frequency mapping.
    EXPECT_EQ(d_single, d_chunked) << "freq " << f;

    // physical ground truth: near the nearest filterbank center frequency
    // (within resonator/FIR shaping) and NOT the raw kMul artifact channel 92.
    size_t predicted = PredictedResonanceChannel(f);
    size_t diff =
        d_single > predicted ? d_single - predicted : predicted - d_single;
    EXPECT_LE(diff, 8u) << "freq " << f;
    EXPECT_NE(d_single, 92u) << "freq " << f;
    ch_single.push_back(d_single);
    ch_chunked.push_back(d_chunked);
  }
  // Strict monotonicity across the tested frequency grid.
  for (size_t i = 1; i < ch_single.size(); ++i) {
    EXPECT_LT(ch_single[i - 1], ch_single[i])
        << "freq " << freqs[i - 1] << " vs " << freqs[i];
    EXPECT_LT(ch_chunked[i - 1], ch_chunked[i])
        << "freq " << freqs[i - 1] << " vs " << freqs[i];
  }
}

// Raw-dB argmax peak vs delta-from-silence peak.
// Documents that the raw spectrogram argmax is pinned at channel 92 across all
// frequencies because kMul[92] == 1.7779 (the maximum weighting coefficient),
// whereas the excess over silence baseline B correctly isolates frequency.
TEST(StreamTest, RawPeakIsFrequencyIndependentArtifact) {
  Zimtohrli z;
  for (float f : {440.0f, 2000.0f}) {
    auto tone = GenerateSineWave(f, 0.5f, 0.5f);
    Spectrogram sp = z.Analyze(tone);
    EXPECT_EQ(RawPeakChannel(sp, 4), 92u) << "freq " << f;
  }
}

// Metric invariant: self-distance is identically zero on both paths.
TEST(StreamTest, SelfDistanceIsZero) {
  Zimtohrli z;
  auto sig = GenerateSineWave(440.0f, 0.5f, 0.5f);
  Spectrogram single = z.Analyze(sig);
  Spectrogram chunked = ProcessInChunks(z, sig, 100);

  // Self-distance on single-pass:
  Spectrogram c1 = CloneSpectrogram(single);
  Spectrogram c2 = CloneSpectrogram(single);
  EXPECT_NEAR(z.DistanceWithoutDtw(c1, c2), 0.0f, 1e-6f);
  c1 = CloneSpectrogram(single);
  c2 = CloneSpectrogram(single);
  EXPECT_NEAR(z.Distance(c1, c2), 0.0f, 1e-6f);

  // Self-distance on chunked:
  c1 = CloneSpectrogram(chunked);
  c2 = CloneSpectrogram(chunked);
  EXPECT_NEAR(z.DistanceWithoutDtw(c1, c2), 0.0f, 1e-6f);
  c1 = CloneSpectrogram(chunked);
  c2 = CloneSpectrogram(chunked);
  EXPECT_NEAR(z.Distance(c1, c2), 0.0f, 1e-6f);

  // Cross-distance between single-pass and chunked (should also be ~0):
  c1 = CloneSpectrogram(single);
  c2 = CloneSpectrogram(chunked);
  EXPECT_NEAR(z.DistanceWithoutDtw(c1, c2), 0.0f, 1e-5f);
}

// Metric invariant: distance is symmetric (dist(A,B) == dist(B,A)) on
// both single-pass and chunked paths.
TEST(StreamTest, DistanceIsSymmetric) {
  Zimtohrli z;
  auto sA = GenerateSineWave(440.0f, 0.5f, 0.5f);
  auto sB = GenerateSineWave(880.0f, 0.5f, 0.5f);

  Spectrogram singleA = z.Analyze(sA);
  Spectrogram singleB = z.Analyze(sB);
  Spectrogram chunkedA = ProcessInChunks(z, sA, 256);
  Spectrogram chunkedB = ProcessInChunks(z, sB, 256);

  // Single-pass symmetry:
  Spectrogram cA = CloneSpectrogram(singleA);
  Spectrogram cB = CloneSpectrogram(singleB);
  float dAB = z.DistanceWithoutDtw(cA, cB);
  cA = CloneSpectrogram(singleA);
  cB = CloneSpectrogram(singleB);
  float dBA = z.DistanceWithoutDtw(cB, cA);
  EXPECT_NEAR(dAB, dBA, 1e-6f);

  // Chunked symmetry:
  cA = CloneSpectrogram(chunkedA);
  cB = CloneSpectrogram(chunkedB);
  float dAB_ck = z.DistanceWithoutDtw(cA, cB);
  cA = CloneSpectrogram(chunkedA);
  cB = CloneSpectrogram(chunkedB);
  float dBA_ck = z.DistanceWithoutDtw(cB, cA);
  EXPECT_NEAR(dAB_ck, dBA_ck, 1e-6f);

  // Cross-agreement:
  EXPECT_NEAR(dAB, dAB_ck, 1e-4f);
}

// Flush on an empty analyzer is a no-op that emits 0 frames.
TEST(StreamTest, FlushOnEmptyIsNoOp) {
  ChunkedAnalyzer analyzer(564);
  std::vector<float> frames;
  analyzer.Flush(frames);
  EXPECT_EQ(analyzer.num_steps(), 0u);
  EXPECT_TRUE(frames.empty());
  // Consecutive flush on empty is also a no-op.
  analyzer.Flush(frames);
  EXPECT_EQ(analyzer.num_steps(), 0u);
  EXPECT_TRUE(frames.empty());
}

// Flush is idempotent and finalize-emits identical frames regardless of
// how many times it is called after a stream completes.
TEST(StreamTest, FlushIsIdempotentAndMatchesSinglePass) {
  Zimtohrli z;
  auto sig = GenerateSineWave(440.0f, 0.1f, 0.5f);  // 4800 samples -> 9 steps
  ChunkedAnalyzer analyzer(z.samples_per_perceptual_block);
  std::vector<float> frames;
  analyzer.Process(sig, frames);
  analyzer.Flush(frames);
  size_t steps_after_first = analyzer.num_steps();
  size_t frames_after_first = frames.size();
  ASSERT_EQ(steps_after_first, 9u);

  // Second Flush() emits no new frames and leaves step count unchanged.
  analyzer.Flush(frames);
  EXPECT_EQ(analyzer.num_steps(), steps_after_first);
  EXPECT_EQ(frames.size(), frames_after_first);

  // Output matches single-pass Analyze.
  Spectrogram single = z.Analyze(sig);
  ASSERT_EQ(single.num_steps, steps_after_first);
  for (size_t s = 0; s < single.num_steps; ++s) {
    for (size_t k = 0; k < single.num_dims; ++k) {
      EXPECT_NEAR(single[s][k], frames[s * single.num_dims + k], 1e-4f);
    }
  }
}

// Process() calls after Flush() continue the stream seamlessly.
TEST(StreamTest, ProcessAfterFlushContinuesStream) {
  Zimtohrli z;
  auto s1 = GenerateSineWave(440.0f, 0.1f, 0.5f);  // 4800 samples
  auto s2 = GenerateSineWave(880.0f, 0.1f, 0.5f);  // 4800 samples

  ChunkedAnalyzer analyzer(z.samples_per_perceptual_block);
  std::vector<float> frames;
  analyzer.Process(s1, frames);
  analyzer.Flush(frames);
  size_t steps_first = analyzer.num_steps();
  ASSERT_EQ(steps_first, 9u);

  analyzer.Process(s2, frames);
  analyzer.Flush(frames);
  size_t steps_total = analyzer.num_steps();
  EXPECT_GT(steps_total, steps_first);
  EXPECT_EQ(frames.size(), steps_total * kNumRotators);
  for (float f : frames) EXPECT_TRUE(std::isfinite(f));
}

// Zimtohrli::Analyze overloads delegate to ChunkedAnalyzer and produce
// identical output to ChunkedAnalyzer directly.
TEST(StreamTest, AnalyzeDelegatesToChunkedAnalyzer) {
  Zimtohrli z;
  auto sig = GenerateSineWave(440.0f, 0.1f, 0.5f);
  // (a) Return-by-value overload:
  Spectrogram sp1 = z.Analyze(sig);
  // (b) Preallocated-buffer overload:
  Spectrogram sp2(sp1.num_steps, sp1.num_dims);
  z.Analyze(sig, sp2);
  // (c) Direct ChunkedAnalyzer usage:
  ChunkedAnalyzer direct(z.samples_per_perceptual_block);
  std::vector<float> frames;
  direct.Process(sig, frames);
  direct.Flush(frames);
  Spectrogram ref(direct.num_steps(), kNumRotators, std::move(frames));

  ASSERT_EQ(sp1.num_steps, ref.num_steps);
  for (size_t s = 0; s < sp1.num_steps; ++s) {
    for (size_t k = 0; k < sp1.num_dims; ++k) {
      EXPECT_FLOAT_EQ(sp1[s][k], ref[s][k]);  // identical single-Process path
      EXPECT_FLOAT_EQ(sp1[s][k], sp2[s][k]);  // both overloads agree
    }
  }
  Spectrogram ck = ProcessInChunks(z, sig, 100);
  ExpectSpectrogramsNear(sp1, ck, 1e-4f);
  EXPECT_EQ(RawPeakChannel(sp1, 4), 92u);
}

// ===========================================================================
// Downsample window shape and long-signal phasor stability tests.
// ===========================================================================

// Multi-frame golden that pins the sigmoid overlap-add window shape.
TEST(StreamTest, SigmoidDownsampleWindowShapeGolden) {
  Zimtohrli z;
  auto signal =
      GenerateSineWave(440.0f, 0.1f, 0.5f);  // 4800 samples -> 9 steps
  Spectrogram single = z.Analyze(signal);
  Spectrogram chunked = ProcessInChunks(z, signal, 100);
  ASSERT_EQ(single.num_steps, 9u);
  ASSERT_EQ(single.num_dims, static_cast<size_t>(kNumRotators));
  ASSERT_EQ(sizeof(kSigmoidWindowGolden) / sizeof(kSigmoidWindowGolden[0]),
            single.size());
  for (size_t s = 0; s < single.num_steps; ++s) {
    for (size_t d = 0; d < single.num_dims; ++d) {
      const float g = kSigmoidWindowGolden[s * single.num_dims + d];
      EXPECT_NEAR(single[s][d], g, 1e-3f)
          << "single step " << s << " dim " << d;
      EXPECT_NEAR(chunked[s][d], g, 1e-3f)
          << "chunked step " << s << " dim " << d;
    }
  }
  ExpectSpectrogramsNear(single, chunked, 1e-4f);
}

// RenormalizePhasors anti-drift test. A sustained pure tone drives the
// filterbank into a stationary regime; RenormalizePhasors (every downsample_
// samples) prevents phasor-magnitude drift, so the resonance-channel energy
// remains stationary across the whole signal.
TEST(StreamTest, RenormalizePhasorsKeepsSteadyToneStationary) {
  Zimtohrli z;
  auto signal = GenerateSineWave(440.0f, 10.0f, 0.5f);  // 480000 samples
  Spectrogram single = z.Analyze(signal);
  Spectrogram chunked = ProcessInChunks(z, signal, 2048);
  ASSERT_EQ(single.num_steps, z.SpectrogramSteps(signal.size()));  // 852
  ASSERT_GT(single.num_steps, 840u);
  EXPECT_TRUE(AllFinite(single));
  EXPECT_TRUE(AllFinite(chunked));
  EXPECT_LT(single.max(), 50.0f);  // no blow-up

  // Resonance channel for the 440 Hz tone.
  constexpr size_t kResCh = 36;
  // Windowed means average out the per-frame phase ripple of a non-integer
  // number of periods per downsample block, isolating any slow drift.
  auto win_mean = [](const Spectrogram& sp, size_t ch, size_t a, size_t b) {
    double acc = 0.0;
    for (size_t st = a; st < b; ++st) acc += sp[st][ch];
    return acc / static_cast<double>(b - a);
  };
  const double early_single = win_mean(single, kResCh, 50, 70);
  const double late_single = win_mean(single, kResCh, 820, 840);
  const double early_chunked = win_mean(chunked, kResCh, 50, 70);
  const double late_chunked = win_mean(chunked, kResCh, 820, 840);

  const double rel_drift_single =
      std::abs(late_single - early_single) / std::abs(early_single);
  const double rel_drift_chunked =
      std::abs(late_chunked - early_chunked) / std::abs(early_chunked);
  EXPECT_LT(rel_drift_single, 1e-4) << "single-pass resonance energy drifted";
  EXPECT_LT(rel_drift_chunked, 1e-4) << "chunked resonance energy drifted";

  EXPECT_NEAR(late_single, 23.5533719, 1e-2) << "single-pass late-mean golden";
  EXPECT_NEAR(late_chunked, 23.5533719, 1e-2) << "chunked late-mean golden";

  ExpectSpectrogramsNear(single, chunked, 1e-3f);
}

// ===========================================================================
// Additional streaming equivalence and boundary condition tests.
// ===========================================================================

std::vector<float> GenerateChordWave(const std::vector<float>& frequencies,
                                     float duration_seconds,
                                     float amplitude = 0.5f,
                                     float sample_rate = 48000.0f) {
  size_t num_samples = static_cast<size_t>(duration_seconds * sample_rate);
  std::vector<float> signal(num_samples, 0.0f);
  if (frequencies.empty()) return signal;
  float norm_amplitude = amplitude / static_cast<float>(frequencies.size());
  for (size_t i = 0; i < num_samples; ++i) {
    float t = static_cast<float>(i) / sample_rate;
    float sample_val = 0.0f;
    for (float freq : frequencies) {
      sample_val += norm_amplitude * std::sin(2.0f * kPi * freq * t);
    }
    signal[i] = sample_val;
  }
  return signal;
}

std::vector<float> GenerateChirpWave(float freq_start, float freq_end,
                                     float duration_seconds,
                                     float amplitude = 0.5f,
                                     float sample_rate = 48000.0f) {
  size_t num_samples = static_cast<size_t>(duration_seconds * sample_rate);
  std::vector<float> signal(num_samples);
  for (size_t i = 0; i < num_samples; ++i) {
    float t = static_cast<float>(i) / sample_rate;
    float phase = 2.0f * kPi *
                  (freq_start * t +
                   0.5f * (freq_end - freq_start) * t * t / duration_seconds);
    signal[i] = amplitude * std::sin(phase);
  }
  return signal;
}

std::vector<float> GenerateTransientWave(
    float frequency, float duration_seconds,
    const std::vector<float>& amplitude_steps,
    float step_duration_seconds = 0.2f, float sample_rate = 48000.0f) {
  size_t num_samples = static_cast<size_t>(duration_seconds * sample_rate);
  size_t samples_per_step =
      static_cast<size_t>(step_duration_seconds * sample_rate);
  if (samples_per_step == 0) samples_per_step = 1;
  std::vector<float> signal(num_samples);
  for (size_t i = 0; i < num_samples; ++i) {
    float t = static_cast<float>(i) / sample_rate;
    size_t step_idx = (i / samples_per_step) % amplitude_steps.size();
    float amp = amplitude_steps[step_idx];
    signal[i] = amp * std::sin(2.0f * kPi * frequency * t);
  }
  return signal;
}

std::vector<float> GenerateImpulseBurstWave(
    float base_freq, float duration_seconds, float base_amp = 0.1f,
    float burst_amp = 1.0f, size_t burst_interval_samples = 4800,
    size_t burst_length_samples = 48, float sample_rate = 48000.0f) {
  size_t num_samples = static_cast<size_t>(duration_seconds * sample_rate);
  std::vector<float> signal(num_samples);
  for (size_t i = 0; i < num_samples; ++i) {
    float t = static_cast<float>(i) / sample_rate;
    bool in_burst = ((i % burst_interval_samples) < burst_length_samples);
    float amp = in_burst ? burst_amp : base_amp;
    signal[i] = amp * std::sin(2.0f * kPi * base_freq * t);
  }
  return signal;
}

// Asserts chunked (fixed and variable chunk sizes) processing is numerically
// equal to single-pass, and that NSIM==1 / DistanceWithoutDtw==0, for a signal.
void VerifyStreamingEquivalence(const Zimtohrli& zimtohrli,
                                Span<const float> signal,
                                const std::vector<size_t>& chunk_sizes,
                                const char* test_name) {
  Spectrogram spec_single = zimtohrli.Analyze(signal);

  for (size_t chunk_size : chunk_sizes) {
    Spectrogram spec_chunked = ProcessInChunks(zimtohrli, signal, chunk_size);

    ASSERT_EQ(spec_single.num_steps, spec_chunked.num_steps)
        << "[" << test_name << "] Step count mismatch for chunk size "
        << chunk_size;
    ASSERT_EQ(spec_single.num_dims, spec_chunked.num_dims)
        << "[" << test_name << "] Dim count mismatch for chunk size "
        << chunk_size;

    for (size_t step = 0; step < spec_single.num_steps; ++step) {
      for (size_t dim = 0; dim < spec_single.num_dims; ++dim) {
        EXPECT_FLOAT_EQ(spec_single[step][dim], spec_chunked[step][dim])
            << "[" << test_name << "] Numerical mismatch at step " << step
            << ", dim " << dim << " for chunk size " << chunk_size;
      }
    }

    float nsim_val = NSIM(spec_single, spec_chunked, zimtohrli.nsim_step_window,
                          zimtohrli.nsim_channel_window);
    EXPECT_FLOAT_EQ(nsim_val, 1.0f)
        << "[" << test_name << "] NSIM bound violated for chunk size "
        << chunk_size;

    Spectrogram copy_single = CloneSpectrogram(spec_single);
    Spectrogram copy_chunked = CloneSpectrogram(spec_chunked);
    float dist_no_dtw = zimtohrli.DistanceWithoutDtw(copy_single, copy_chunked);
    EXPECT_FLOAT_EQ(dist_no_dtw, 0.0f)
        << "[" << test_name
        << "] DistanceWithoutDtw bound violated for chunk size " << chunk_size;
  }

  std::vector<size_t> variable_chunks = {128,  512,  100, 2048, 564,
                                         1000, 4096, 33,  7};
  Spectrogram spec_var =
      ProcessInVariableChunks(zimtohrli, signal, variable_chunks);
  ASSERT_EQ(spec_single.num_steps, spec_var.num_steps)
      << "[" << test_name << "] Step count mismatch for variable chunks";
  ASSERT_EQ(spec_single.num_dims, spec_var.num_dims)
      << "[" << test_name << "] Dim count mismatch for variable chunks";

  for (size_t step = 0; step < spec_single.num_steps; ++step) {
    for (size_t dim = 0; dim < spec_single.num_dims; ++dim) {
      EXPECT_FLOAT_EQ(spec_single[step][dim], spec_var[step][dim])
          << "[" << test_name << "] Numerical mismatch at step " << step
          << ", dim " << dim << " for variable chunks";
    }
  }

  float nsim_var = NSIM(spec_single, spec_var, zimtohrli.nsim_step_window,
                        zimtohrli.nsim_channel_window);
  EXPECT_FLOAT_EQ(nsim_var, 1.0f)
      << "[" << test_name << "] NSIM bound violated for variable chunks";

  Spectrogram copy_single_var = CloneSpectrogram(spec_single);
  Spectrogram copy_var = CloneSpectrogram(spec_var);
  float dist_var = zimtohrli.DistanceWithoutDtw(copy_single_var, copy_var);
  EXPECT_FLOAT_EQ(dist_var, 0.0f)
      << "[" << test_name
      << "] DistanceWithoutDtw bound violated for variable chunks";
}

// Downsample factors <= 0 must be clamped to 1 (identical output to factor 1).
TEST(StreamTest, ChunkedAnalyzerZeroAndNegativeDownsample) {
  auto signal = GenerateSineWave(440.0f, 0.05f);

  ChunkedAnalyzer analyzer_1(1);
  ChunkedAnalyzer analyzer_zero(0);
  ChunkedAnalyzer analyzer_neg(-5);

  std::vector<float> frames_1;
  std::vector<float> frames_zero;
  std::vector<float> frames_neg;

  analyzer_1.Process(signal, frames_1);
  analyzer_1.Flush(frames_1);

  analyzer_zero.Process(signal, frames_zero);
  analyzer_zero.Flush(frames_zero);

  analyzer_neg.Process(signal, frames_neg);
  analyzer_neg.Flush(frames_neg);

  ASSERT_EQ(analyzer_1.num_steps(), analyzer_zero.num_steps());
  ASSERT_EQ(analyzer_1.num_steps(), analyzer_neg.num_steps());
  ASSERT_EQ(frames_1.size(), frames_zero.size());
  ASSERT_EQ(frames_1.size(), frames_neg.size());

  for (size_t i = 0; i < frames_1.size(); ++i) {
    EXPECT_FLOAT_EQ(frames_1[i], frames_zero[i])
        << "Zero downsample factor mismatch at index " << i;
    EXPECT_FLOAT_EQ(frames_1[i], frames_neg[i])
        << "Negative downsample factor mismatch at index " << i;
  }
}

// Sub-kernel-length signals produce numerically identical output whether run
// single-pass or streamed one-sample or whole-signal at a time.
TEST(StreamTest, SubKernelNumericalValuesMatchSinglePass) {
  Zimtohrli zimtohrli;
  for (size_t num_samples : {1, 5, 10, 31}) {
    auto short_signal =
        GenerateSineWave(440.0f, static_cast<float>(num_samples) / 48000.0f);
    short_signal.resize(num_samples);

    Spectrogram spec_single = zimtohrli.Analyze(short_signal);
    Spectrogram spec_chunked_1 = ProcessInChunks(zimtohrli, short_signal, 1);
    Spectrogram spec_chunked_full =
        ProcessInChunks(zimtohrli, short_signal, num_samples);

    ASSERT_EQ(spec_single.num_steps, spec_chunked_1.num_steps)
        << "Step count mismatch for short signal length " << num_samples;
    ASSERT_EQ(spec_single.num_steps, spec_chunked_full.num_steps)
        << "Step count mismatch for short signal length " << num_samples;

    for (size_t step = 0; step < spec_single.num_steps; ++step) {
      for (size_t dim = 0; dim < spec_single.num_dims; ++dim) {
        EXPECT_FLOAT_EQ(spec_single[step][dim], spec_chunked_1[step][dim])
            << "Numerical mismatch at step " << step << ", dim " << dim
            << " for chunk size 1, signal length " << num_samples;
        EXPECT_FLOAT_EQ(spec_single[step][dim], spec_chunked_full[step][dim])
            << "Numerical mismatch at step " << step << ", dim " << dim
            << " for full chunk, signal length " << num_samples;
      }
    }
  }
}

// Pre-allocated Spectrogram buffers safely truncate output when smaller than
// produced and zero-pad when larger than produced.
TEST(StreamTest, PreAllocatedSpectrogramTruncationAndPadding) {
  Zimtohrli zimtohrli;
  auto signal = GenerateSineWave(440.0f, 0.1f);  // 4800 samples
  Spectrogram expected = zimtohrli.Analyze(signal);

  ASSERT_GT(expected.num_steps, 2);

  // Truncation case: pre-allocated Spectrogram has fewer steps than produced.
  size_t trunc_steps = expected.num_steps - 2;
  Spectrogram trunc_spec(trunc_steps, kNumRotators);
  zimtohrli.Analyze(signal, trunc_spec);
  EXPECT_EQ(trunc_spec.num_steps, trunc_steps);
  for (size_t step = 0; step < trunc_steps; ++step) {
    for (size_t dim = 0; dim < kNumRotators; ++dim) {
      EXPECT_FLOAT_EQ(trunc_spec[step][dim], expected[step][dim])
          << "Truncation mismatch at step " << step << ", dim " << dim;
    }
  }

  // Zero-padding case: pre-allocated Spectrogram has more steps than produced.
  size_t pad_steps = expected.num_steps + 3;
  Spectrogram pad_spec(pad_steps, kNumRotators);
  zimtohrli.Analyze(signal, pad_spec);
  EXPECT_EQ(pad_spec.num_steps, pad_steps);
  for (size_t step = 0; step < expected.num_steps; ++step) {
    for (size_t dim = 0; dim < kNumRotators; ++dim) {
      EXPECT_FLOAT_EQ(pad_spec[step][dim], expected[step][dim])
          << "Padding mismatch in active region at step " << step << ", dim "
          << dim;
    }
  }
  for (size_t step = expected.num_steps; step < pad_steps; ++step) {
    for (size_t dim = 0; dim < kNumRotators; ++dim) {
      EXPECT_FLOAT_EQ(pad_spec[step][dim], 0.0f)
          << "Expected zero-padding at step " << step << ", dim " << dim;
    }
  }
}

// Buffer compaction across the 4096-sample threshold preserves sample alignment
// and produces spectrograms identical to single-pass, using chunk patterns that
// straddle the exact boundary (4095/4096/4097).
TEST(StreamTest, BufferCompactionBoundaryStress) {
  Zimtohrli zimtohrli;
  auto signal = GenerateSineWave(440.0f, 10000.0f / 48000.0f);  // 10000 samples
  signal.resize(10000);

  Spectrogram spec_single = zimtohrli.Analyze(signal);

  std::vector<std::vector<size_t>> stress_patterns = {
      {4095, 10, 4095, 10},
      {4096, 4096, 1808},
      {4097, 4097, 1806},
      {4095, 1, 3, 4097},
  };

  for (const auto& pattern : stress_patterns) {
    Spectrogram spec_chunked =
        ProcessInVariableChunks(zimtohrli, signal, pattern);

    ASSERT_EQ(spec_single.num_steps, spec_chunked.num_steps);
    ASSERT_EQ(spec_single.num_dims, spec_chunked.num_dims);

    for (size_t step = 0; step < spec_single.num_steps; ++step) {
      for (size_t dim = 0; dim < spec_single.num_dims; ++dim) {
        EXPECT_FLOAT_EQ(spec_single[step][dim], spec_chunked[step][dim])
            << "Mismatch at step " << step << ", dim " << dim
            << " around compaction boundary";
      }
    }
  }
}

// Streaming matches single-pass across varied real-world signal durations.
TEST(StreamTest, E2E_VariedSignalDurations) {
  Zimtohrli zimtohrli;
  std::vector<float> test_durations = {2.0f, 5.0f, 7.5f, 10.0f};
  std::vector<size_t> chunk_sizes = {256, 564, 1024, 4096, 16384};

  for (float duration : test_durations) {
    // Sustained tonal mixture (440Hz + 880Hz) representing continuous audio.
    auto signal = GenerateChordWave({440.0f, 880.0f}, duration, 0.6f);
    std::string test_name =
        "Duration_" + std::to_string(static_cast<int>(duration)) + "s";
    VerifyStreamingEquivalence(zimtohrli, signal, chunk_sizes,
                               test_name.c_str());
  }
}

// Streaming matches single-pass across chords and harmonic series.
TEST(StreamTest, E2E_SyntheticWaveforms_ChordsAndHarmonics) {
  Zimtohrli zimtohrli;
  std::vector<size_t> chunk_sizes = {100, 512, 1024, 4096};

  // 1. A Major Triad chord (A4, C#5, E5) for 3.0s.
  auto triad = GenerateChordWave({440.0f, 554.37f, 659.25f}, 3.0f, 0.7f);
  VerifyStreamingEquivalence(zimtohrli, triad, chunk_sizes, "AMajorTriad_3s");

  // 2. Harmonic series (8 harmonics of 100Hz fundamental) for 3.0s.
  auto harmonics = GenerateChordWave(
      {100.0f, 200.0f, 300.0f, 400.0f, 500.0f, 600.0f, 700.0f, 800.0f}, 3.0f,
      0.8f);
  VerifyStreamingEquivalence(zimtohrli, harmonics, chunk_sizes, "Harmonics_3s");

  // 3. Dissonant multi-frequency cluster for 2.5s.
  auto cluster = GenerateChordWave({250.0f, 265.0f, 1200.0f, 3400.0f, 8192.0f},
                                   2.5f, 0.6f);
  VerifyStreamingEquivalence(zimtohrli, cluster, chunk_sizes, "Cluster_2_5s");
}

// Streaming matches single-pass across upward, downward, and wide-band chirps.
TEST(StreamTest, E2E_SyntheticWaveforms_FrequencySweptChirps) {
  Zimtohrli zimtohrli;
  std::vector<size_t> chunk_sizes = {256, 564, 2048, 8192};

  // 1. Upward linear chirp (100Hz -> 4000Hz over 3.0s).
  auto up_chirp = GenerateChirpWave(100.0f, 4000.0f, 3.0f, 0.6f);
  VerifyStreamingEquivalence(zimtohrli, up_chirp, chunk_sizes, "UpChirp_3s");

  // 2. Downward linear chirp (8000Hz -> 200Hz over 2.5s).
  auto down_chirp = GenerateChirpWave(8000.0f, 200.0f, 2.5f, 0.6f);
  VerifyStreamingEquivalence(zimtohrli, down_chirp, chunk_sizes,
                             "DownChirp_2_5s");

  // 3. Wide-band chirp (50Hz -> 16000Hz over 4.0s).
  auto wide_chirp = GenerateChirpWave(50.0f, 16000.0f, 4.0f, 0.5f);
  VerifyStreamingEquivalence(zimtohrli, wide_chirp, chunk_sizes,
                             "WideChirp_4s");
}

// Streaming matches single-pass across amplitude steps, impulse bursts, and
// rapid gating transients.
TEST(StreamTest, E2E_SyntheticWaveforms_DynamicAmplitudeTransients) {
  Zimtohrli zimtohrli;
  std::vector<size_t> chunk_sizes = {64, 512, 1000, 4096};

  // 1. Step amplitude transients changing every 200ms over 3.0s.
  auto step_transients = GenerateTransientWave(
      440.0f, 3.0f, {0.1f, 0.9f, 0.0f, 0.8f, 0.05f}, 0.2f);
  VerifyStreamingEquivalence(zimtohrli, step_transients, chunk_sizes,
                             "StepTransients_3s");

  // 2. Impulse burst transients (1.0 amp burst every 100ms for 1ms) over 3.0s.
  auto impulse_bursts =
      GenerateImpulseBurstWave(300.0f, 3.0f, 0.05f, 0.95f, 4800, 48);
  VerifyStreamingEquivalence(zimtohrli, impulse_bursts, chunk_sizes,
                             "ImpulseBursts_3s");

  // 3. Rapid on/off gating every 50ms over 2.5s.
  auto rapid_gating = GenerateTransientWave(1000.0f, 2.5f, {0.8f, 0.0f}, 0.05f);
  VerifyStreamingEquivalence(zimtohrli, rapid_gating, chunk_sizes,
                             "RapidGating_2_5s");
}

// Perceptual distance/NSIM bounds agree between single-pass and multi-chunk
// processing when comparing distinct signals.
TEST(StreamTest, E2E_PerceptualDistanceBoundsCrossValidation) {
  Zimtohrli zimtohrli;

  // Case A: Compare two different chord intensities / harmonic ratios.
  auto chord_ref = GenerateChordWave({440.0f, 554.37f, 659.25f}, 2.0f, 0.7f);
  auto chord_test = GenerateChordWave({440.0f, 554.37f, 659.25f}, 2.0f, 0.35f);

  Spectrogram ref_single = zimtohrli.Analyze(chord_ref);
  Spectrogram test_single = zimtohrli.Analyze(chord_test);
  Spectrogram ref_chunked = ProcessInChunks(zimtohrli, chord_ref, 1024);
  Spectrogram test_chunked = ProcessInChunks(zimtohrli, chord_test, 2048);

  // NSIM without DTW comparison between ref and test.
  float nsim_single = NSIM(ref_single, test_single, zimtohrli.nsim_step_window,
                           zimtohrli.nsim_channel_window);
  float nsim_chunked =
      NSIM(ref_chunked, test_chunked, zimtohrli.nsim_step_window,
           zimtohrli.nsim_channel_window);
  EXPECT_FLOAT_EQ(nsim_single, nsim_chunked);

  // DistanceWithoutDtw comparison between ref and test.
  Spectrogram copy_ref_s1 = CloneSpectrogram(ref_single);
  Spectrogram copy_test_s1 = CloneSpectrogram(test_single);
  Spectrogram copy_ref_c1 = CloneSpectrogram(ref_chunked);
  Spectrogram copy_test_c1 = CloneSpectrogram(test_chunked);
  float dist_no_dtw_single =
      zimtohrli.DistanceWithoutDtw(copy_ref_s1, copy_test_s1);
  float dist_no_dtw_chunked =
      zimtohrli.DistanceWithoutDtw(copy_ref_c1, copy_test_c1);
  EXPECT_FLOAT_EQ(dist_no_dtw_single, dist_no_dtw_chunked);

  // Distance (with DTW) comparison between ref and test.
  Spectrogram copy_ref_s2 = CloneSpectrogram(ref_single);
  Spectrogram copy_test_s2 = CloneSpectrogram(test_single);
  Spectrogram copy_ref_c2 = CloneSpectrogram(ref_chunked);
  Spectrogram copy_test_c2 = CloneSpectrogram(test_chunked);
  float dist_dtw_single = zimtohrli.Distance(copy_ref_s2, copy_test_s2);
  float dist_dtw_chunked = zimtohrli.Distance(copy_ref_c2, copy_test_c2);
  EXPECT_FLOAT_EQ(dist_dtw_single, dist_dtw_chunked);

  // Case B: Compare upward chirp vs downward chirp.
  auto up_chirp = GenerateChirpWave(200.0f, 3000.0f, 1.5f, 0.5f);
  auto down_chirp = GenerateChirpWave(3000.0f, 200.0f, 1.5f, 0.5f);

  Spectrogram up_single = zimtohrli.Analyze(up_chirp);
  Spectrogram down_single = zimtohrli.Analyze(down_chirp);
  Spectrogram up_chunked = ProcessInChunks(zimtohrli, up_chirp, 512);
  Spectrogram down_chunked = ProcessInChunks(zimtohrli, down_chirp, 4096);

  Spectrogram copy_up_s = CloneSpectrogram(up_single);
  Spectrogram copy_down_s = CloneSpectrogram(down_single);
  Spectrogram copy_up_c = CloneSpectrogram(up_chunked);
  Spectrogram copy_down_c = CloneSpectrogram(down_chunked);

  float dist_chirp_single = zimtohrli.Distance(copy_up_s, copy_down_s);
  float dist_chirp_chunked = zimtohrli.Distance(copy_up_c, copy_down_c);
  EXPECT_FLOAT_EQ(dist_chirp_single, dist_chirp_chunked);
}

}  // namespace
}  // namespace zimtohrli
