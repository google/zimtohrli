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

#ifndef CPP_ZIMT_NSIM_H_
#define CPP_ZIMT_NSIM_H_

#include <cmath>
#include <utility>
#include <vector>

#include "hwy/aligned_allocator.h"
#include "zimt/spectrogram.h"

namespace {

template <typename T>
Spectrogram WindowMean(size_t num_steps, size_t num_channels,
                       size_t step_window, size_t channel_window,
                       T input_loader) {
  Spectrogram tmp_a(num_steps, num_channels);
  Spectrogram tmp_b(num_steps, num_channels);

  // Populate tmp_b with prefix sums across the step axis.
  {
    float* channel_prefix_sum_data = tmp_b.step(0);
    for (size_t channel_index = 0; channel_index < num_channels;
         ++channel_index) {
      channel_prefix_sum_data[channel_index] = input_loader(0, channel_index);
    }
  }
  for (size_t step_index = 1; step_index < num_steps; ++step_index) {
    float* channel_prefix_sum_data = tmp_b.step(step_index);
    const float* channel_prev_prefix_sum_data = tmp_b.step(step_index - 1);
    for (size_t channel_index = 0; channel_index < num_channels;
         ++channel_index) {
      channel_prefix_sum_data[channel_index] =
          input_loader(step_index, channel_index) +
          channel_prev_prefix_sum_data[channel_index];
    }
  }

  // Populate tmp_a with windowed sums across the step axis using the prefix
  // sums in tmp_b.
  // 1: Copy the step_window first rows from tmp_b to tmp_a.
  std::memcpy(tmp_a.values.get(), tmp_b.values.get(),
              step_window * num_channels * sizeof(float));
  // 2: Compute windowed sums by subtracting prefix sums from each other.
  for (size_t step_index = step_window; step_index < num_steps; ++step_index) {
    const float* curr_window_sum_data = tmp_b.step(step_index);
    const float* prev_window_sum_data = tmp_b.step(step_index - step_window);
    float* channel_window_sum_data = tmp_a.step(step_index);
    for (size_t channel_index = 0; channel_index < num_channels;
         ++channel_index) {
      channel_window_sum_data[channel_index] =
          curr_window_sum_data[channel_index] -
          prev_window_sum_data[channel_index];
    }
  }

  for (size_t step_index = 0; step_index < num_steps; ++step_index) {
    // Populate tmp_b with prefix sums across the channel axis of the windowed
    // sums across the step axis in tmp_a.
    {
      const float* channel_window_sum_data = tmp_a.step(step_index);
      float* step_prefix_sum_data = tmp_b.step(step_index);
      step_prefix_sum_data[0] = channel_window_sum_data[0];
      for (size_t channel_index = 1; channel_index < num_channels;
           ++channel_index) {
        step_prefix_sum_data[channel_index] =
            step_prefix_sum_data[channel_index - 1] +
            channel_window_sum_data[channel_index];
      }
    }
    // Populate tmp_a with windowed sums across steps-and-channels axes using
    // the "prefix sums across the channel axis and windowed sums across the
    // step axis" of tmp_b.
    {
      const float* step_prefix_sum_data = tmp_b.step(step_index);
      float* step_window_sum_data = tmp_a.step(step_index);
      std::memcpy(step_window_sum_data, step_prefix_sum_data,
                  channel_window * sizeof(float));
      for (size_t channel_index = channel_window; channel_index < num_channels;
           ++channel_index) {
        step_window_sum_data[channel_index] =
            step_prefix_sum_data[channel_index] -
            step_prefix_sum_data[channel_index - channel_window];
      }
    }
  }

  // Divide all windowed sums by step_window * channel_window to make them mean
  // values.
  const float reciprocal = 1.0 / (step_window * channel_window);
  for (size_t step_index = 0; step_index < num_steps; ++step_index) {
    float* result_data = tmp_a.step(step_index);
    for (size_t channel_index = 0; channel_index < num_channels;
         ++channel_index) {
      result_data[channel_index] *= reciprocal;
    }
  }

  return tmp_a;
}

// Returns a slightly nonstandard version of the NSIM neural structural
// similarity metric between arrays a and b.
//
// step_window and channel_window are the number of time steps and channels
// in the array over which to window the mean, standard deviance, and
// covariance measures in NSIM.
//
// time_pairs is the dynamic time warp computed between spectrograms a and
// b, i.e. pairs of time step indices where a and b are considered to match
// each other in time.
//
// See https://doi.org/10.1016/j.specom.2011.09.004 for details.
float NSIM(const Spectrogram& a, const Spectrogram& b,
           const std::vector<std::pair<size_t, size_t>>& time_pairs,
           size_t step_window, size_t channel_window) {
  assert(a.num_dims == b.num_dims);
  const size_t num_channels = a.num_dims;
  const size_t num_steps = time_pairs.size();

  const Spectrogram mean_a =
      WindowMean(num_steps, num_channels, step_window, channel_window,
                 [&](size_t step_index, size_t channel_index) {
                   return a.step(time_pairs[step_index].first)[channel_index];
                 });
  const Spectrogram mean_b =
      WindowMean(num_steps, num_channels, step_window, channel_window,
                 [&](size_t step_index, size_t channel_index) {
                   return b.step(time_pairs[step_index].second)[channel_index];
                 });
  // NB: This computes (value - mean) using the mean computed for the window
  // at the same position as the value, so that each value gets a different mean
  // subtracted.
  const Spectrogram var_a =
      WindowMean(num_steps, num_channels, step_window, channel_window,
                 [&](size_t step_index, size_t channel_index) {
                   const float delta =
                       a.step(time_pairs[step_index].first)[channel_index] -
                       mean_a.step(step_index)[channel_index];
                   return delta * delta;
                 });
  const Spectrogram var_b =
      WindowMean(num_steps, num_channels, step_window, channel_window,
                 [&](size_t step_index, size_t channel_index) {
                   const float delta =
                       b.step(time_pairs[step_index].second)[channel_index] -
                       mean_b.step(step_index)[channel_index];
                   return delta * delta;
                 });
  const Spectrogram cov =
      WindowMean(num_steps, num_channels, step_window, channel_window,
                 [&](size_t step_index, size_t channel_index) {
                   const float delta_a =
                       a.step(time_pairs[step_index].first)[channel_index] -
                       mean_a.step(step_index)[channel_index];
                   const float delta_b =
                       b.step(time_pairs[step_index].second)[channel_index] -
                       mean_b.step(step_index)[channel_index];
                   return delta_a * delta_b;
                 });

  // nsim-inspired ad hoc aggregation
  // main changes:
  // The aggregation tries to be more L1 than L2
  // Clamping of structure value
  // Adding a small amount of a-b L1 diff
  //
  // These changes were measured to be small improvements on a multi-corpus
  // test.
  const float C1 = 12.611504825401516;
  const float C3 = 8.296128127652123;
  const float C4 = 4.3913017618850251e-05;
  const float C5 = 5.2623574756945486e-07;
  const float C6 = 1.9650853180896431e-06;
  const float C7 = 0.00015815354596871763;
  const float C8 = 0.55786604844807497;

  float nsim_sum = 0.0;
  for (size_t step_index = 0; step_index < num_steps; ++step_index) {
    for (size_t channel_index = 0; channel_index < num_channels;
         ++channel_index) {
      const float mean_a_vec = mean_a.step(step_index)[channel_index];
      const float mean_b_vec = mean_b.step(step_index)[channel_index];
      const float std_a_vec = std::sqrt(var_a.step(step_index)[channel_index]);
      const float std_b_vec = std::sqrt(var_b.step(step_index)[channel_index]);
      const float cov_vec = cov.step(step_index)[channel_index];
      const float intensity =
          (2 * std::sqrt(mean_a_vec * mean_b_vec) + C1) /
          (std::abs(mean_a_vec) + std::abs(mean_b_vec) + C1);
      const float structure_base =
          (cov_vec + C3) / (std_a_vec * std_b_vec + C3);
      const float structure_clamped = structure_base < C8 ? C8 : structure_base;
      const float structure =
          std::sqrt(std::sqrt(structure_clamped + C4) + C5) + C6;
      const float nsim = intensity * structure;
      const float aval = a.step(time_pairs[step_index].first)[channel_index];
      const float bval = b.step(time_pairs[step_index].second)[channel_index];
      const float diff = aval - bval;
      const float sqrdiff = C7 * std::abs(diff);
      const float nsim2 = nsim + sqrdiff;
      nsim_sum += nsim2;
    }
  }
  return nsim_sum / static_cast<float>(num_steps * num_channels);
}

}  // namespace

namespace zimtohrli {

// Returns an array shaped exactly like source, where each element is the mean
// of the zero-padded step_window x channel_window rectangle of preceding
// elements.
hwy::AlignedNDArray<float, 2> WindowMeanHwy(
    const hwy::AlignedNDArray<float, 2>& source, size_t step_window,
    size_t channel_window);

// Returns a slightly nonstandard version of the NSIM neural structural
// similarity metric between arrays a and b.
//
// step_window and channel_window are the number of time steps and channels in
// the array over which to window the mean, standard deviance, and covariance
// measures in NSIM.
//
// time_pairs is the dynamic time warp computed between array a and array b,
// i.e. pairs of time step indices where array a and array b are considered to
// match each other in time.
//
// See https://doi.org/10.1016/j.specom.2011.09.004 for details.
float NSIMHwy(const hwy::AlignedNDArray<float, 2>& a,
              const hwy::AlignedNDArray<float, 2>& b,
              const std::vector<std::pair<size_t, size_t>>& time_pairs,
              size_t step_window, size_t channel_window);

}  // namespace zimtohrli

#endif  // CPP_ZIMT_NSIM_H_