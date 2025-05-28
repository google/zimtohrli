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

#include "zimt/zimtohrli.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/types/span.h"
#include "zimt/dtw.h"
#include "zimt/fourier_bank.h"
#include "zimt/nsim.h"

namespace zimtohrli {

EnergyAndMaxAbsAmplitude Measure(Span<const float> signal) {
  float signal_max = 0;
  float signal_energy = 0;
  for (size_t index = 0; index < signal.size; ++index) {
    const float amplitude = signal.data[index];
    signal_energy += amplitude * amplitude;
    signal_max = std::max(signal_max, std::abs(amplitude));
  }
  return {.energy_db_fs =
              20 * std::log10(signal_energy / static_cast<float>(signal.size)),
          .max_abs_amplitude = signal_max};
}

EnergyAndMaxAbsAmplitude NormalizeAmplitude(float max_abs_amplitude,
                                            Span<float> signal) {
  float signal_max = 0;
  float signal_energy = 0;
  for (size_t index = 0; index < signal.size; ++index) {
    signal_max = std::max(signal_max, std::abs(signal.data[index]));
  }
  const float scaling = max_abs_amplitude / signal_max;
  for (size_t index = 0; index < signal.size; ++index) {
    const float new_amplitude = scaling * signal.data[index];
    signal_energy += new_amplitude * new_amplitude;
    signal.data[index] = new_amplitude;
  }
  return {.energy_db_fs =
              20 * std::log10(signal_energy / static_cast<float>(signal.size)),
          .max_abs_amplitude = max_abs_amplitude};
}

float Zimtohrli::Distance(const Spectrogram& spectrogram_a,
                          const Spectrogram& spectrogram_b) const {
  CHECK_EQ(spectrogram_a.num_dims, spectrogram_b.num_dims);
  std::vector<std::pair<size_t, size_t>> time_pairs;
  time_pairs = DTW(spectrogram_a, spectrogram_b);
  return NSIM(spectrogram_a, spectrogram_b, time_pairs, nsim_step_window,
              nsim_channel_window);
}

void Zimtohrli::Analyze(Span<const float> signal,
                        Spectrogram& spectrogram) const {
  CHECK_EQ(spectrogram.num_dims, kNumRotators);
  Rotators rots;
  rots.FilterAndDownsample(signal.data, signal.size, spectrogram.values.data(),
                           spectrogram.num_steps, spectrogram.num_dims,
                           signal.size / spectrogram.num_steps);
}

std::vector<std::vector<float>> Zimtohrli::Compare(
    const AudioBuffer& frames_a,
    Span<const AudioBuffer* const> frames_b_span) const {
  for (size_t b_index = 0; b_index < frames_b_span.size; ++b_index) {
    CHECK_EQ(frames_a.num_channels, frames_b_span[b_index]->num_channels);
  }
  std::vector<std::vector<float>> distance_b_vector(frames_b_span.size);
  for (size_t audio_channel_index = 0;
       audio_channel_index < frames_a.num_channels; ++audio_channel_index) {
    size_t num_steps_a =
        static_cast<size_t>(std::ceil(static_cast<float>(frames_a.num_frames) *
                                      perceptual_sample_rate / kSampleRate));
    Spectrogram current_spec_a(num_steps_a, kNumRotators);
    Analyze(frames_a[audio_channel_index], current_spec_a);
    for (size_t b_index = 0; b_index < frames_b_span.size; ++b_index) {
      size_t num_steps_b = static_cast<size_t>(
          std::ceil(static_cast<float>(frames_b_span[b_index]->num_frames) *
                    perceptual_sample_rate / kSampleRate));
      Spectrogram spec_b(num_steps_b, kNumRotators);
      Analyze((*frames_b_span[b_index])[audio_channel_index], spec_b);
      const float distance = Distance(current_spec_a, spec_b);
      distance_b_vector[b_index].push_back(distance);
    }
  }
  return distance_b_vector;
}

}  // namespace zimtohrli
