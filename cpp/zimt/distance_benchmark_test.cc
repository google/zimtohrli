// Copyright 2024 The Zimtohrli Authors. All Rights Reserved.
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

#include <cmath>
#include <cstddef>
#include <vector>

#include "benchmark/benchmark.h"
#include "gtest/gtest.h"
#include "zimt/zimtohrli.h"

namespace zimtohrli {

namespace {

// Generate a simple sine wave at given frequency
std::vector<float> GenerateSineWave(float frequency, float duration_seconds, 
                                    float sample_rate = 48000.0f) {
  const size_t num_samples = static_cast<size_t>(duration_seconds * sample_rate);
  std::vector<float> samples(num_samples);
  const float angular_freq = 2.0f * M_PI * frequency / sample_rate;
  
  for (size_t i = 0; i < num_samples; ++i) {
    samples[i] = 0.5f * std::sin(angular_freq * i);
  }
  
  return samples;
}

static void BM_ZimtohrliFullPipeline(benchmark::State& state) {
  // Generate two 2-second sine waves at slightly different frequencies
  const float duration = 2.0f;  // 2 seconds
  const std::vector<float> signal_a = GenerateSineWave(440.0f, duration);  // A4
  const std::vector<float> signal_b = GenerateSineWave(445.0f, duration);  // Slightly sharp A4
  
  Zimtohrli zimtohrli;
  
  // Benchmark the full pipeline
  for (auto _ : state) {
    Spectrogram spec_a = zimtohrli.Analyze({signal_a.data(), signal_a.size()});
    Spectrogram spec_b = zimtohrli.Analyze({signal_b.data(), signal_b.size()});
    float distance = zimtohrli.Distance(spec_a, spec_b);
    benchmark::DoNotOptimize(distance);
  }
  
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_ZimtohrliFullPipeline);

}  // namespace

}  // namespace zimtohrli