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

}  // namespace
}  // namespace zimtohrli
