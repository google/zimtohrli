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

#ifndef CPP_ZIMT_DTW_H_
#define CPP_ZIMT_DTW_H_

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <utility>
#include <vector>

#include "hwy/aligned_allocator.h"

namespace zimtohrli {

namespace {

// A simple buffer of float samples describing a spectrogram with a given number
// of steps and feature dimensions.
struct Spectrogram {
  const float* step(size_t n) const { return values + n * num_dims; }
  float* step(size_t n) { return values + n * num_dims; }
  size_t num_steps;
  size_t num_dims;
  float* values;
};

// A simple buffer of double cost values describing the time warp costs between
// two spectrograms.
struct CostMatrix {
  const double get(size_t step_a, size_t step_b) {
    return values[step_a * steps_b + step_b];
  }
  void set(size_t step_a, size_t step_b, double value) {
    values[step_a * steps_b + step_b] = value;
  }
  CostMatrix(size_t steps_a, size_t steps_b)
      : steps_a(steps_a), steps_b(steps_b) {
    values = std::vector<double>(steps_a * steps_b);
  }
  size_t steps_a;
  size_t steps_b;
  std::vector<double> values;
};

// Computes the norm of the delta between two spectrograms at two given steps.
double delta_norm(const Spectrogram& a, const Spectrogram& b, size_t step_a,
                  size_t step_b) {
  assert(a.num_dims == b.num_dims);
  const float* dims_a = a.step(step_a);
  const float* dims_b = b.step(step_b);
  double result = 0;
  for (size_t index = 0; index < a.num_dims; index++) {
    float delta = dims_a[index] - dims_b[index];
    result += delta * delta;
  }
  return std::pow(result, 0.23289303544689094);
}

// Computes the DTW (https://en.wikipedia.org/wiki/Dynamic_time_warping)
// between two arrays.
std::vector<std::pair<size_t, size_t>> DTW(const Spectrogram& spec_a,
                                           const Spectrogram& spec_b) {
  // Sanity check that both spectrograms have the same number of feature
  // dimensions.
  assert(spec_a.num_dims == spec_b.num_dims);
  // Initialize a cost matrix with 0 at the start point, infinity at [*, 0] and
  // [0, *].
  CostMatrix cost_matrix(spec_a.num_steps, spec_b.num_steps);
  for (size_t step_a = 0; step_a < spec_a.num_steps; step_a++) {
    cost_matrix.set(step_a, 0, std::numeric_limits<double>::infinity());
  }
  for (size_t step_b = 0; step_b < spec_b.num_steps; step_b++) {
    cost_matrix.set(0, step_b, std::numeric_limits<double>::infinity());
  }
  cost_matrix.set(0, 0, 0);
  // Compute cost as cost as weighted sum of feature dimension norms to each
  // cell.
  static const double kMul00 = 0.98585952515276176;
  for (size_t spec_a_index = 1; spec_a_index < spec_a.num_steps;
       ++spec_a_index) {
    for (size_t spec_b_index = 1; spec_b_index < spec_b.num_steps;
         ++spec_b_index) {
      const double cost_at_index =
          delta_norm(spec_a, spec_b, spec_a_index, spec_b_index);
      const double sync_cost = cost_matrix.get(spec_a_index, spec_b_index);
      const double unsync_cost =
          std::min(cost_matrix.get(spec_a_index - 1, spec_b_index),
                   cost_matrix.get(spec_a_index, spec_b_index - 1));
      const double costmin = std::min(sync_cost + kMul00 * cost_at_index,
                                      unsync_cost + cost_at_index);
      cost_matrix.set(spec_a_index, spec_b_index, costmin);
    }
  }

  // Track the cheapest path through the cost matrix.
  std::vector<std::pair<size_t, size_t>> result;
  std::pair<size_t, size_t> pos = {0, 0};
  result.push_back(pos);
  while (pos.first + 1 < spec_a.num_steps &&
         pos.second + 1 < spec_b.num_steps) {
    double min_cost = std::numeric_limits<double>::max();
    for (const auto& test_pos :
         {std::pair<size_t, size_t>{pos.first + 1, pos.second + 1},
          std::pair<size_t, size_t>{pos.first + 1, pos.second},
          std::pair<size_t, size_t>{pos.first, pos.second + 1}}) {
      double cost = cost_matrix.get(pos.first, pos.second);
      if (cost < min_cost) {
        min_cost = cost;
        pos = test_pos;
      }
    }
    result.push_back(pos);
  }
  return result;
}

// ChainDTW used to be a windowed version of DTW, but now it's just a wrapper
// around DTW using hwy data structures. Replace it with a call to DTW with the
// Spectrogram structs to get rid of hwy.
std::vector<std::pair<size_t, size_t>> ChainDTW(
    const hwy::AlignedNDArray<float, 2>& spec_a,
    const hwy::AlignedNDArray<float, 2>& spec_b, size_t window_size) {
  assert(spec_a.shape()[1] == spec_b.shape()[1]);
  std::vector<float> spectrogram_a_data(spec_a.shape()[0] * spec_a.shape()[1]);
  std::vector<float> spectrogram_b_data(spec_b.shape()[0] * spec_b.shape()[1]);
  Spectrogram spectrogram_a{.num_steps = spec_a.shape()[0],
                            .num_dims = spec_a.shape()[1],
                            .values = spectrogram_a_data.data()};
  Spectrogram spectrogram_b{.num_steps = spec_b.shape()[0],
                            .num_dims = spec_b.shape()[1],
                            .values = spectrogram_b_data.data()};
  for (size_t step = 0; step < spectrogram_a.num_steps; step++) {
    std::memcpy(spectrogram_a.step(step), spec_a[{step}].data(),
                sizeof(float) * spec_a.shape()[1]);
  }
  for (size_t step = 0; step < spectrogram_b.num_steps; step++) {
    std::memcpy(spectrogram_b.step(step), spec_b[{step}].data(),
                sizeof(float) * spec_b.shape()[1]);
  }
  return DTW(spectrogram_a, spectrogram_b);
}

}  // namespace

}  // namespace zimtohrli

#endif  // CPP_ZIMT_DTW_H_
