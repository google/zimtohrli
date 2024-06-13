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
#include "zimt/filterbank.h"
#include "zimt/fourier_bank.h"
#include "zimt/masking.h"
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

template <bool verbose>
Distance HwyDistance(const Zimtohrli& z,
                     const hwy::AlignedNDArray<float, 2>& spectrogram_a,
                     const hwy::AlignedNDArray<float, 2>& spectrogram_b,
                     const std::vector<std::pair<size_t, size_t>>& time_pairs) {
  // Since NSIM is a similarity measure, where 1.0 is "perfectly similar", we
  // subtract it from 1.0 to get a distance metric instead.
  Distance result{
      .value = 1.0f -
               NSIM(spectrogram_a, spectrogram_b, time_pairs,
                    std::min(spectrogram_a.shape()[0], z.nsim_step_window),
                    std::min(spectrogram_a.shape()[1], z.nsim_channel_window))};
  if constexpr (verbose) {
    const Vec log_10_div_20 = Set(d, log(10) / 20);
    const Vec twenty_vec = Set(d, 20);
    const size_t num_channels = spectrogram_a.shape()[1];
    const Vec epsilon_vec = Set(d, z.epsilon);
    for (size_t time_index = 0; time_index < time_pairs.size(); ++time_index) {
      const std::pair<size_t, size_t> t = time_pairs[time_index];
      for (size_t channel_index = 0; channel_index < num_channels;
           channel_index += Lanes(d)) {
        const Vec spec_a_db =
            Load(d, spectrogram_a[{t.first}].data() + channel_index);
        const Vec spec_b_db =
            Load(d, spectrogram_b[{t.second}].data() + channel_index);
        const auto manage_local_maximum = [&](const Vec& vec,
                                              SpectrogramDelta& target) {
          if (ReduceMax(d, vec) > target.value) {
            hwy::AlignedNDArray<float, 1> lane_values({Lanes(d)});
            Store(vec, d, lane_values.data());
            for (size_t index = 0; index < Lanes(d); ++index) {
              if (lane_values[{}][index] > target.value) {
                const size_t local_channel_index = channel_index + index;
                target.value = lane_values[{}][index];
                target.spectrogram_a_value =
                    spectrogram_a[{t.first}][local_channel_index];
                target.spectrogram_b_value =
                    spectrogram_b[{t.second}][local_channel_index];
                target.sample_a_index = t.first;
                target.sample_b_index = t.second;
                target.channel_index = local_channel_index;
              }
            }
          }
        };

        const Vec spec_a_linear_amplitude =
            LinearAmplitudeFromDb(spec_a_db, log_10_div_20);
        const Vec spec_b_linear_amplitude =
            LinearAmplitudeFromDb(spec_b_db, log_10_div_20);
        const Vec noise_linear_amplitude =
            AbsDiff(spec_a_linear_amplitude, spec_b_linear_amplitude);
        const Vec noise_db_vec = DbFromLinearAmplitude(noise_linear_amplitude,
                                                       epsilon_vec, twenty_vec);
        manage_local_maximum(noise_db_vec, result.max_absolute_delta);
        const Vec delta_db_vec = AbsDiff(spec_a_db, spec_b_db);
        manage_local_maximum(delta_db_vec, result.max_relative_delta);
      }
    }
  }
  return result;
}

Distance HwyDistanceVerbose(
    const Zimtohrli& z, const hwy::AlignedNDArray<float, 2>& spectrogram_a,
    const hwy::AlignedNDArray<float, 2>& spectrogram_b,
    const std::vector<std::pair<size_t, size_t>>& time_pairs) {
  return HwyDistance<true>(z, spectrogram_a, spectrogram_b, time_pairs);
}

Distance HwyDistanceFast(
    const Zimtohrli& z, const hwy::AlignedNDArray<float, 2>& spectrogram_a,
    const hwy::AlignedNDArray<float, 2>& spectrogram_b,
    const std::vector<std::pair<size_t, size_t>>& time_pairs) {
  return HwyDistance<false>(z, spectrogram_a, spectrogram_b, time_pairs);
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
HWY_EXPORT(HwyDistanceVerbose);
HWY_EXPORT(HwyDistanceFast);
HWY_EXPORT(HwyAbsDiff);
HWY_EXPORT(HwySubtractDb);

Distance Zimtohrli::Distance(
    bool verbose, const hwy::AlignedNDArray<float, 2>& spectrogram_a,
    const hwy::AlignedNDArray<float, 2>& spectrogram_b) const {
  if (unwarp_window_seconds == 0) {
    CHECK_EQ(spectrogram_a.shape()[0], spectrogram_b.shape()[0]);
  }
  CHECK_EQ(spectrogram_a.shape()[1], spectrogram_b.shape()[1]);
  std::vector<std::pair<size_t, size_t>> time_pairs;
  if (unwarp_window_seconds != 0) {
    time_pairs = ChainDTW(spectrogram_a, spectrogram_b,
                          static_cast<size_t>(unwarp_window_seconds *
                                              cam_filterbank->sample_rate));
  } else {
    time_pairs.reserve(spectrogram_a.shape()[0]);
    for (size_t index = 0; index < spectrogram_a.shape()[0]; ++index) {
      time_pairs.push_back({index, index});
    }
  }
  if (verbose) {
    return HWY_DYNAMIC_DISPATCH(HwyDistanceVerbose)(*this, spectrogram_a,
                                                    spectrogram_b, time_pairs);
  } else {
    return HWY_DYNAMIC_DISPATCH(HwyDistanceFast)(*this, spectrogram_a,
                                                 spectrogram_b, time_pairs);
  }
}

void Zimtohrli::Spectrogram(
    hwy::Span<const float> signal, FilterbankState& state,
    hwy::AlignedNDArray<float, 2>& channels,
    hwy::AlignedNDArray<float, 2>& energy_channels_db,
    hwy::AlignedNDArray<float, 2>& partial_energy_channels_db,
    hwy::AlignedNDArray<float, 2>& spectrogram) const {
  CHECK_EQ(signal.size(), channels.shape()[0]);
  CHECK_GE(channels.shape()[0], energy_channels_db.shape()[0]);
  CHECK_EQ(channels.shape()[1], energy_channels_db.shape()[1]);
  CHECK_EQ(energy_channels_db.shape()[0],
           partial_energy_channels_db.shape()[0]);
  CHECK_EQ(energy_channels_db.shape()[1],
           partial_energy_channels_db.shape()[1]);
  CHECK_EQ(partial_energy_channels_db.shape()[0], spectrogram.shape()[0]);
  CHECK_EQ(partial_energy_channels_db.shape()[1], spectrogram.shape()[1]);
  // Using a tabuli::Rotators instead of the cam_filterbank filter.
  std::vector<float> freqs;
  std::vector<float> gains;
  for (size_t i = 0; i < cam_filterbank->filter.Size(); ++i) {
    freqs.push_back(cam_filterbank->thresholds_hz[{1}][i]);
    gains.push_back(1.0);
  }
  tabuli::Rotators rots(1, freqs, gains, cam_filterbank->sample_rate, 1.0f);
  rots.Filter(signal, channels);
  ComputeEnergy(channels, energy_channels_db);
  ToDb(energy_channels_db, full_scale_sine_db, epsilon, energy_channels_db);
  if (apply_masking) {
    masking.CutFullyMasked(energy_channels_db, cam_filterbank->cam_delta,
                           partial_energy_channels_db);
  } else {
    hwy::CopyBytes(energy_channels_db.data(), partial_energy_channels_db.data(),
                   energy_channels_db.memory_size() * sizeof(float));
  }
  if (apply_loudness) {
    loudness.PhonsFromSPL(partial_energy_channels_db,
                          cam_filterbank->thresholds_hz, spectrogram);

  } else {
    hwy::CopyBytes(partial_energy_channels_db.data(), spectrogram.data(),
                   partial_energy_channels_db.memory_size() * sizeof(float));
  }
}

void Zimtohrli::Spectrogram(
    hwy::Span<const float> signal, hwy::AlignedNDArray<float, 2>& channels,
    hwy::AlignedNDArray<float, 2>& energy_channels_db,
    hwy::AlignedNDArray<float, 2>& partial_energy_channels_db,
    hwy::AlignedNDArray<float, 2>& spectrogram) const {
  FilterbankState new_state = cam_filterbank->filter.NewState();
  Spectrogram(signal, new_state, channels, energy_channels_db,
              partial_energy_channels_db, spectrogram);
}

Analysis Zimtohrli::Analyze(hwy::Span<const float> signal,
                            FilterbankState& state,
                            hwy::AlignedNDArray<float, 2>& channels) const {
  const size_t num_downscaled_samples = static_cast<size_t>(std::max(
      1.0f, std::ceil(static_cast<float>(signal.size()) *
                      perceptual_sample_rate / cam_filterbank->sample_rate)));
  hwy::AlignedNDArray<float, 2> energy_channels_db(
      {num_downscaled_samples, channels.shape()[1]});
  hwy::AlignedNDArray<float, 2> partial_energy_channels_db(
      {num_downscaled_samples, channels.shape()[1]});
  hwy::AlignedNDArray<float, 2> spectrogram(
      {num_downscaled_samples, channels.shape()[1]});
  Spectrogram(signal, state, channels, energy_channels_db,
              partial_energy_channels_db, spectrogram);
  return {.energy_channels_db = std::move(energy_channels_db),
          .partial_energy_channels_db = std::move(partial_energy_channels_db),
          .spectrogram = std::move(spectrogram)};
}

Analysis Zimtohrli::Analyze(hwy::Span<const float> signal,
                            hwy::AlignedNDArray<float, 2>& channels) const {
  FilterbankState new_state = cam_filterbank->filter.NewState();
  return Analyze(signal, new_state, channels);
}

AnalysisDTW::AnalysisDTW(const Analysis& analysis_a, const Analysis& analysis_b,
                         size_t window_size) {
  this->energy_channels_db =
      ChainDTW(analysis_a.energy_channels_db, analysis_b.energy_channels_db,
               window_size);
  this->partial_energy_channels_db =
      ChainDTW(analysis_a.partial_energy_channels_db,
               analysis_b.partial_energy_channels_db, window_size);
  this->spectrogram =
      ChainDTW(analysis_a.spectrogram, analysis_b.spectrogram, window_size);
}

AnalysisDTW::AnalysisDTW(size_t length) {
  std::vector<std::pair<size_t, size_t>> time_pairs(length);
  for (size_t index = 0; index < length; ++index) {
    time_pairs[index] = {index, index};
  }
  this->energy_channels_db = time_pairs;
  this->partial_energy_channels_db = time_pairs;
  this->spectrogram = time_pairs;
}

Comparison Zimtohrli::Compare(
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
  std::vector<Analysis> analysis_a;
  analysis_a.reserve(frames_b_span.size());
  std::vector<std::vector<Analysis>> analysis_b_vector(frames_b_span.size());
  std::vector<std::vector<Analysis>> analysis_absolute_delta_vector(
      frames_b_span.size());
  std::vector<std::vector<Analysis>> analysis_relative_delta_vector(
      frames_b_span.size());
  std::vector<std::vector<AnalysisDTW>> dtw(frames_b_span.size());
  for (size_t audio_channel_index = 0; audio_channel_index < num_audio_channels;
       ++audio_channel_index) {
    hwy::AlignedNDArray<float, 2> channels_a(
        {frames_a.shape()[1], cam_filterbank->filter.Size()});
    Analysis current_analysis_a =
        Analyze(frames_a[{audio_channel_index}], channels_a);
    for (size_t b_index = 0; b_index < frames_b_span.size(); ++b_index) {
      hwy::AlignedNDArray<float, 2> channels_b(
          {frames_b_span[b_index]->shape()[1], cam_filterbank->filter.Size()});
      Analysis current_analysis_b =
          Analyze((*frames_b_span[b_index])[{audio_channel_index}], channels_b);

      const AnalysisDTW current_analysis_dtw =
          unwarp_window_seconds == 0
              ? AnalysisDTW(current_analysis_a.spectrogram.shape()[0])
              : AnalysisDTW(current_analysis_a, current_analysis_b,
                            static_cast<size_t>(unwarp_window_seconds *
                                                cam_filterbank->sample_rate));

      const hwy::AlignedNDArray<float, 2>& frames_b = *frames_b_span[b_index];
      if (audio_channel_index == 0) {
        audio_delta_vector.push_back(hwy::AlignedNDArray<float, 2>(
            {num_audio_channels, frames_a.shape()[1]}));
      }
      hwy::AlignedNDArray<float, 2>& audio_delta = audio_delta_vector[b_index];
      const size_t dtw_block_size =
          frames_a.shape()[1] /
          current_analysis_a.energy_channels_db.shape()[0];
      for (const auto& dtw_block_pair :
           current_analysis_dtw.energy_channels_db) {
        const size_t first_dtw_offset = dtw_block_pair.first * dtw_block_size;
        float* audio_delta_data =
            audio_delta[{audio_channel_index}].data() + first_dtw_offset;
        const float* frames_a_data =
            frames_a[{audio_channel_index}].data() + first_dtw_offset;
        const float* frames_b_data = frames_b[{audio_channel_index}].data() +
                                     dtw_block_pair.second * dtw_block_size;
        for (size_t index = 0; index < dtw_block_size; ++index) {
          audio_delta_data[index] = frames_a_data[index] - frames_b_data[index];
        }
      }

      hwy::AlignedNDArray<float, 2> absolute_delta_energy_channels_db(
          {current_analysis_dtw.energy_channels_db.back().first + 1,
           current_analysis_a.energy_channels_db.shape()[1]});
      HWY_DYNAMIC_DISPATCH(HwySubtractDb)
      (*this, current_analysis_a.energy_channels_db,
       current_analysis_b.energy_channels_db, absolute_delta_energy_channels_db,
       current_analysis_dtw.energy_channels_db);

      hwy::AlignedNDArray<float, 2> relative_delta_energy_channels_db(
          {current_analysis_dtw.energy_channels_db.back().first + 1,
           current_analysis_a.energy_channels_db.shape()[1]});
      HWY_DYNAMIC_DISPATCH(HwyAbsDiff)
      (current_analysis_a.energy_channels_db,
       current_analysis_b.energy_channels_db, relative_delta_energy_channels_db,
       current_analysis_dtw.energy_channels_db);

      hwy::AlignedNDArray<float, 2> absolute_delta_partial_energy_channels_db(
          {current_analysis_dtw.partial_energy_channels_db.back().first + 1,
           current_analysis_a.partial_energy_channels_db.shape()[1]});
      HWY_DYNAMIC_DISPATCH(HwySubtractDb)
      (*this, current_analysis_a.partial_energy_channels_db,
       current_analysis_b.partial_energy_channels_db,
       absolute_delta_partial_energy_channels_db,
       current_analysis_dtw.partial_energy_channels_db);

      hwy::AlignedNDArray<float, 2> relative_delta_partial_energy_channels_db(
          {current_analysis_dtw.partial_energy_channels_db.back().first + 1,
           current_analysis_a.partial_energy_channels_db.shape()[1]});
      HWY_DYNAMIC_DISPATCH(HwyAbsDiff)
      (current_analysis_a.partial_energy_channels_db,
       current_analysis_b.partial_energy_channels_db,
       relative_delta_partial_energy_channels_db,
       current_analysis_dtw.partial_energy_channels_db);

      hwy::AlignedNDArray<float, 2> absolute_delta_spectrogram(
          {current_analysis_dtw.spectrogram.back().first + 1,
           current_analysis_a.spectrogram.shape()[1]});
      HWY_DYNAMIC_DISPATCH(HwySubtractDb)
      (*this, current_analysis_a.spectrogram, current_analysis_b.spectrogram,
       absolute_delta_spectrogram, current_analysis_dtw.spectrogram);

      hwy::AlignedNDArray<float, 2> relative_delta_spectrogram(
          {current_analysis_dtw.spectrogram.back().first + 1,
           current_analysis_a.spectrogram.shape()[1]});
      HWY_DYNAMIC_DISPATCH(HwyAbsDiff)
      (current_analysis_a.spectrogram, current_analysis_b.spectrogram,
       relative_delta_spectrogram, current_analysis_dtw.spectrogram);

      analysis_b_vector[b_index].push_back(std::move(current_analysis_b));
      dtw[b_index].push_back(std::move(current_analysis_dtw));
      analysis_absolute_delta_vector[b_index].push_back(Analysis{
          .energy_channels_db = std::move(absolute_delta_energy_channels_db),
          .partial_energy_channels_db =
              std::move(absolute_delta_partial_energy_channels_db),
          .spectrogram = std::move(absolute_delta_spectrogram)});
      analysis_relative_delta_vector[b_index].push_back(Analysis{
          .energy_channels_db = std::move(relative_delta_energy_channels_db),
          .partial_energy_channels_db =
              std::move(relative_delta_partial_energy_channels_db),
          .spectrogram = std::move(relative_delta_spectrogram)});
    }
    analysis_a.push_back(std::move(current_analysis_a));
  }
  return {.analysis_a = std::move(analysis_a),
          .analysis_b = std::move(analysis_b_vector),
          .dtw = std::move(dtw),
          .analysis_absolute_delta = std::move(analysis_absolute_delta_vector),
          .analysis_relative_delta = std::move(analysis_relative_delta_vector),
          .frames_delta = std::move(audio_delta_vector)};
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
