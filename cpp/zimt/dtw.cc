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

#include "zimt/dtw.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <limits>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "hwy/aligned_allocator.h"

// This file uses a lot of magic from the SIMD library Highway.
// In simplified terms, it will compile the code for multiple architectures
// using the "foreach_target.h" header file, and use the special namespace
// convention HWY_NAMESPACE to find the code to adapt to the SIMD functions,
// which are then called via HWY_DYNAMIC_DISPATCH. This leads to a lot of
// hard-to-explain Highway-related conventions being followed, like this here
// #define that makes this entire file be included by Highway in the process of
// building.
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "zimt/dtw.cc"
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

float HwyMax(hwy::Span<const float> span) {
  float max = -std::numeric_limits<float>::infinity();
  for (size_t index = 0; index < span.size(); index += Lanes(d)) {
    max = std::max(max, ReduceMax(d, Load(d, span.data() + index)));
  }
  return max;
}

float HwyMin(hwy::Span<const float> span) {
  float min = std::numeric_limits<float>::infinity();
  for (size_t index = 0; index < span.size(); index += Lanes(d)) {
    min = std::min(min, ReduceMin(d, Load(d, span.data() + index)));
  }
  return min;
}

float HwyDeltaNorm(hwy::Span<const float> span_a,
                   hwy::Span<const float> span_b) {
  CHECK_EQ(span_a.size(), span_b.size());
  Vec sumvec = Set(d, 0.0f);
  for (size_t index = 0; index < span_a.size(); index += Lanes(d)) {
    const Vec delta =
        Sub(Load(d, span_a.data() + index), Load(d, span_b.data() + index));
    sumvec = MulAdd(delta, delta, sumvec);
  }
  return sqrt(static_cast<double>(ReduceSum(d, sumvec)));
}

}  // namespace HWY_NAMESPACE

}  // namespace zimtohrli
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace zimtohrli {

HWY_EXPORT(HwyDeltaNorm);
HWY_EXPORT(HwyMax);
HWY_EXPORT(HwyMin);

namespace {

struct ArraySlice {
  const hwy::AlignedNDArray<float, 2>& array;
  const size_t offset;
  const size_t size;

  std::array<size_t, 2> shape() const { return {size, array.shape()[1]}; }

  hwy::Span<const float> operator[](std::array<const size_t, 1> indices) const {
    return array[{indices[0] + offset}];
  }
};

std::vector<std::pair<size_t, size_t>> DTWSlice(
    const ArraySlice& spec_a, const ArraySlice& spec_b,
    hwy::AlignedNDArray<float, 2>& cost_matrix) {
  CHECK_EQ(cost_matrix.shape()[0], spec_a.shape()[0]);
  CHECK_EQ(cost_matrix.shape()[1], spec_b.shape()[0]);
  for (size_t spec_b_index = 0; spec_b_index < spec_b.shape()[0];
       ++spec_b_index) {
    cost_matrix[{0}][spec_b_index] = std::numeric_limits<float>::infinity();
  }
  for (size_t spec_a_index = 1; spec_a_index < spec_a.shape()[0];
       ++spec_a_index) {
    for (size_t spec_b_index = 0; spec_b_index < spec_b.shape()[0];
         ++spec_b_index) {
      cost_matrix[{spec_a_index}][spec_b_index] =
          std::numeric_limits<float>::infinity();
    }
  }
  cost_matrix[{0}][0] = 0;
  for (size_t spec_a_index = 1; spec_a_index < spec_a.shape()[0];
       ++spec_a_index) {
    for (size_t spec_b_index = 1; spec_b_index < spec_b.shape()[0];
         ++spec_b_index) {
      const float cost = HWY_DYNAMIC_DISPATCH(HwyDeltaNorm)(
          spec_a[{spec_a_index}], spec_b[{spec_b_index}]);
      cost_matrix[{spec_a_index}][spec_b_index] =
          cost +
          std::min(cost_matrix[{spec_a_index - 1}][spec_b_index - 1],
                   std::min(cost_matrix[{spec_a_index - 1}][spec_b_index],
                            cost_matrix[{spec_a_index}][spec_b_index - 1]));
    }
  }

  std::vector<std::pair<size_t, size_t>> result;
  std::pair<size_t, size_t> pos = {0, 0};
  result.push_back(pos);
  while (pos.first + 1 < cost_matrix.shape()[0] &&
         pos.second + 1 < cost_matrix.shape()[1]) {
    std::vector<std::pair<size_t, size_t>> possible_next_positions = {
        {pos.first + 1, pos.second + 1},
        {pos.first + 1, pos.second},
        {pos.first, pos.second + 1}};
    std::vector<float> direction_costs(possible_next_positions.size());
    std::transform(
        possible_next_positions.begin(), possible_next_positions.end(),
        direction_costs.begin(),
        [&](const auto& pos) { return cost_matrix[{pos.first}][pos.second]; });
    int min_cost_direction = std::distance(
        direction_costs.begin(),
        std::min_element(direction_costs.begin(), direction_costs.end()));
    pos = possible_next_positions[min_cost_direction];
    result.push_back(pos);
  }
  return result;
}

}  // namespace

std::vector<std::pair<size_t, size_t>> DTW(
    const hwy::AlignedNDArray<float, 2>& spec_a,
    const hwy::AlignedNDArray<float, 2>& spec_b) {
  hwy::AlignedNDArray<float, 2> cost_matrix(
      {spec_a.shape()[0], spec_b.shape()[0]});
  return DTWSlice({spec_a, 0, spec_b.shape()[0]},
                  {spec_b, 0, spec_a.shape()[0]}, cost_matrix);
}

std::vector<std::pair<size_t, size_t>> ChainDTW(
    const hwy::AlignedNDArray<float, 2>& spec_a,
    const hwy::AlignedNDArray<float, 2>& spec_b, size_t window_size) {
  std::pair<size_t, size_t> offset = {0, 0};
  std::vector<std::pair<size_t, size_t>> result = {offset};
  hwy::AlignedNDArray<float, 2> cost_matrix(
      {std::min(window_size, spec_a.shape()[0]),
       std::min(window_size, spec_b.shape()[0])});
  // Do DTW on one window at a time, moving our offset forward to the end of
  // each computed DTW.
  while (offset.first + 1 < spec_a.shape()[0] &&
         offset.second + 1 < spec_b.shape()[0]) {
    const ArraySlice slice_a = {
        spec_a, offset.first,
        std::min(window_size, spec_a.shape()[0] - offset.first)};
    const ArraySlice slice_b = {
        spec_b, offset.second,
        std::min(window_size, spec_b.shape()[0] - offset.second)};
    if (cost_matrix.shape()[0] != slice_a.shape()[0] ||
        cost_matrix.shape()[1] != slice_b.shape()[0]) {
      cost_matrix = hwy::AlignedNDArray<float, 2>(
          {slice_a.shape()[0], slice_b.shape()[0]});
    }
    std::vector<std::pair<size_t, size_t>> dtw =
        DTWSlice(slice_a, slice_b, cost_matrix);
    // If we have more than one entire window before reaching the end, then
    // throw away the forward half of the DTW to allow a wider search outside
    // what the last search ended at.
    if (spec_a.shape()[0] - dtw.back().first - offset.first > window_size &&
        spec_b.shape()[0] - dtw.back().second - offset.second > window_size) {
      dtw = std::vector<std::pair<size_t, size_t>>(
          dtw.begin(), dtw.begin() + dtw.size() / 2);
    }
    // Don't add the start point for the last window, it's already in the
    // result.
    for (size_t dtw_index = 1; dtw_index < dtw.size(); ++dtw_index) {
      result.push_back({dtw[dtw_index].first + offset.first,
                        dtw[dtw_index].second + offset.second});
    }
    offset = result.back();
  }
  return result;
}

}  // namespace zimtohrli

#endif  // HWY_ONCE
