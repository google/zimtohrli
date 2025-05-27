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
#include "hwy/aligned_allocator.h"
#include "zimt/dtw.h"
#include "zimt/fourier_bank.h"
#include "zimt/nsim.h"

// This file uses a lot of magic from the SIMD library Highway.
// In simplified terms, it will compile the code for multiple architectures
// using the "foreach_target.h" header file, and use the special namespace
// convention HWY_NAMESPACE to find the code to adapt to the SIMD functions,
// which are then called via HWY_DYNAMIC_DISPATCH. This leads to a lot of
// hard-to-explain Highway-related conventions being followed, like this here
// #define that makes this entire file be included by Highway in the process of
// building.
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "zimt/zimtohrli.cc"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
// Must come after foreach_target.h to avoid redefinition errors.
#include "hwy/contrib/math/math-inl.h"
#include "hwy/highway.h"

// This is Highway magic conventions.
HWY_BEFORE_NAMESPACE();
namespace zimtohrli {

namespace HWY_NAMESPACE {

const hwy::HWY_NAMESPACE::ScalableTag<float> d;
using Vec = hwy::HWY_NAMESPACE::Vec<decltype(d)>;

void HwyAbsDiff(const hwy::AlignedNDArray<float, 2>& array_a,
                const hwy::AlignedNDArray<float, 2>& array_b,
                hwy::AlignedNDArray<float, 2>& result,
                const std::vector<std::pair<size_t, size_t>>& time_pairs) {
  for (const auto& sample_pair : time_pairs) {
    for (size_t channel_index = 0; channel_index < array_a.shape()[1];
         channel_index += Lanes(d)) {
      Store(AbsDiff(
                Load(d, array_a[{sample_pair.first}].data() + channel_index),
                Load(d, array_b[{sample_pair.second}].data() + channel_index)),
            d, result[{sample_pair.first}].data() + channel_index);
    }
  }
}

// Returns 10^(db / 20):
// y = 10^(db / 20)
// ln(y) = db/20 * ln(10)
// y = e^(db/20 * ln(10))
Vec LinearAmplitudeFromDb(const Vec& db, const Vec& log_10_div_20) {
  return Exp(d, Mul(db, log_10_div_20));
}

// Returns 20 * log10(linear_amplitude + epsilon).
Vec DbFromLinearAmplitude(const Vec& linear_amplitude, const Vec& epsilon_vec,
                          const Vec& twenty_vec) {
  return Mul(twenty_vec, Log10(d, Add(linear_amplitude, epsilon_vec)));
}

void HwySubtractDb(const Zimtohrli& z,
                   const hwy::AlignedNDArray<float, 2>& array_a,
                   const hwy::AlignedNDArray<float, 2>& array_b,
                   hwy::AlignedNDArray<float, 2>& result,
                   const std::vector<std::pair<size_t, size_t>>& time_pairs) {
  const Vec log_10_div_20 = Set(d, log(10) / 20);
  const Vec twenty_vec = Set(d, 20);
  const size_t num_channels = array_a.shape()[1];
  const Vec epsilon_vec = Set(d, z.epsilon);
  for (const auto& sample_pair : time_pairs) {
    for (size_t channel_index = 0; channel_index < num_channels;
         channel_index += Lanes(d)) {
      const Vec array_a_linear_amplitude = LinearAmplitudeFromDb(
          Load(d, array_a[{sample_pair.first}].data() + channel_index),
          log_10_div_20);
      const Vec array_b_linear_amplitude = LinearAmplitudeFromDb(
          Load(d, array_b[{sample_pair.second}].data() + channel_index),
          log_10_div_20);
      const Vec noise_linear_amplitude =
          AbsDiff(array_a_linear_amplitude, array_b_linear_amplitude);
      Store(DbFromLinearAmplitude(noise_linear_amplitude, epsilon_vec,
                                  twenty_vec),
            d, result[{sample_pair.first}].data() + channel_index);
    }
  }
}

float HwyDistance(const Zimtohrli& z,
                  const hwy::AlignedNDArray<float, 2>& spectrogram_a,
                  const hwy::AlignedNDArray<float, 2>& spectrogram_b,
                  const std::vector<std::pair<size_t, size_t>>& time_pairs) {
  // Since NSIM is a similarity measure, where 1.0 is "perfectly similar", we
  // subtract it from 1.0 to get a distance metric instead.
  return 1.0f -
         NSIMHwy(spectrogram_a, spectrogram_b, time_pairs,
                 std::min(spectrogram_a.shape()[0], z.nsim_step_window),
                 std::min(spectrogram_a.shape()[1], z.nsim_channel_window));
}

EnergyAndMaxAbsAmplitude HwyMeasure(hwy::Span<const float> signal) {
  const size_t signal_samples = signal.size();
  const float* signal_data = signal.data();
  float signal_max = 0;
  float signal_energy = 0;
  for (size_t index = 0; index < signal_samples; index += Lanes(d)) {
    const Vec amplitude = Load(d, signal_data + index);
    signal_energy += ReduceSum(d, Mul(amplitude, amplitude));
    signal_max = std::max(signal_max, ReduceMax(d, Abs(amplitude)));
  }
  return {.energy_db_fs = 20 * std::log10(signal_energy /
                                          static_cast<float>(signal_samples)),
          .max_abs_amplitude = signal_max};
}

EnergyAndMaxAbsAmplitude HwyNormalizeAmplitude(float max_abs_amplitude,
                                               hwy::Span<float> signal) {
  const size_t signal_samples = signal.size();
  float* signal_data = signal.data();
  float signal_max = 0;
  float signal_energy = 0;
  for (size_t index = 0; index < signal_samples; index += Lanes(d)) {
    signal_max =
        std::max(signal_max, ReduceMax(d, Abs(Load(d, signal_data + index))));
  }
  const Vec scaling = Set(d, max_abs_amplitude / signal_max);
  for (size_t index = 0; index < signal_samples; index += Lanes(d)) {
    const Vec new_amplitude = Mul(scaling, Load(d, signal_data + index));
    signal_energy += ReduceSum(d, new_amplitude * new_amplitude);
    Store(new_amplitude, d, signal_data + index);
  }
  return {.energy_db_fs = 20 * std::log10(signal_energy /
                                          static_cast<float>(signal_samples)),
          .max_abs_amplitude = max_abs_amplitude};
}

}  // namespace HWY_NAMESPACE

}  // namespace zimtohrli
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace zimtohrli {

HWY_EXPORT(HwyNormalizeAmplitude);
HWY_EXPORT(HwyMeasure);
HWY_EXPORT(HwyAbsDiff);
HWY_EXPORT(HwySubtractDb);
HWY_EXPORT(HwyDistance);

float Zimtohrli::Distance(
    bool verbose, const hwy::AlignedNDArray<float, 2>& spectrogram_a,
    const hwy::AlignedNDArray<float, 2>& spectrogram_b) const {
  if (unwarp_window_seconds == 0) {
    CHECK_EQ(spectrogram_a.shape()[0], spectrogram_b.shape()[0]);
  }
  CHECK_EQ(spectrogram_a.shape()[1], spectrogram_b.shape()[1]);
  std::vector<std::pair<size_t, size_t>> time_pairs;
  if (unwarp_window_seconds != 0) {
    time_pairs =
        ChainDTW(spectrogram_a, spectrogram_b,
                 static_cast<size_t>(unwarp_window_seconds * 0 /* ignored */));
  } else {
    time_pairs.reserve(spectrogram_a.shape()[0]);
    for (size_t index = 0; index < spectrogram_a.shape()[0]; ++index) {
      time_pairs.push_back({index, index});
    }
  }
  return HWY_DYNAMIC_DISPATCH(HwyDistance)(*this, spectrogram_a, spectrogram_b,
                                           time_pairs);
}

void Zimtohrli::Spectrogram(hwy::Span<const float> signal,
                            hwy::AlignedNDArray<float, 2>& spectrogram) const {
  int downsample = signal.size() / spectrogram.shape()[0];

  Rotators rots;
  std::vector<float> out(spectrogram.shape()[0] * kNumRotators);

  rots.FilterAndDownsample(signal.data(), signal.size(), out.data(),
                           spectrogram.shape()[0], kNumRotators, downsample);

  for (size_t i = 0; i < spectrogram.shape()[0]; ++i) {
    for (size_t k = 0; k < kNumRotators; ++k) {
      spectrogram[{i}][k] = out[i * kNumRotators + k];
    }
  }
}

std::vector<std::vector<float>> Zimtohrli::Compare(
    const hwy::AlignedNDArray<float, 2>& frames_a,
    absl::Span<const hwy::AlignedNDArray<float, 2>* const> frames_b_span)
    const {
  for (const auto& frames_b : frames_b_span) {
    if (unwarp_window_seconds == 0) {
      CHECK_EQ(frames_a.shape()[0], frames_b->shape()[0]);
    }
    CHECK_EQ(frames_a.shape()[1], frames_b->shape()[1]);
  }
  const size_t num_audio_channels = frames_a.shape()[0];
  std::vector<hwy::AlignedNDArray<float, 2>> audio_delta_vector;
  audio_delta_vector.reserve(frames_b_span.size());
  std::vector<std::vector<float>> distance_b_vector(frames_b_span.size());
  for (size_t audio_channel_index = 0; audio_channel_index < num_audio_channels;
       ++audio_channel_index) {
    size_t num_samples_a =
        static_cast<size_t>(std::ceil(static_cast<float>(frames_a.shape()[0]) *
                                      perceptual_sample_rate / kSampleRate));
    hwy::AlignedNDArray<float, 2> current_spec_a({num_samples_a, kNumRotators});
    for (size_t b_index = 0; b_index < frames_b_span.size(); ++b_index) {
      size_t num_samples_b = static_cast<size_t>(
          std::ceil(static_cast<float>(frames_b_span[b_index]->shape()[0]) *
                    perceptual_sample_rate / kSampleRate));
      hwy::AlignedNDArray<float, 2> spec_b({num_samples_b, kNumRotators});
      const float distance = Distance(false, current_spec_a, spec_b);
      distance_b_vector[b_index].push_back(distance);
    }
  }
  return distance_b_vector;
}

EnergyAndMaxAbsAmplitude Measure(hwy::Span<const float> signal) {
  return HWY_DYNAMIC_DISPATCH(HwyMeasure)(signal);
}

EnergyAndMaxAbsAmplitude NormalizeAmplitude(float max_abs_amplitude,
                                            hwy::Span<float> signal) {
  return HWY_DYNAMIC_DISPATCH(HwyNormalizeAmplitude)(max_abs_amplitude, signal);
}

}  // namespace zimtohrli

#endif  // HWY_ONCE
