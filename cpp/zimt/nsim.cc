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

#include "zimt/nsim.h"

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
#define HWY_TARGET_INCLUDE "zimt/nsim.cc"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
// Must come after foreach_target.h to avoid redefinition errors.
#include "hwy/highway.h"

// This is Highway magic conventions.
HWY_BEFORE_NAMESPACE();
namespace zimtohrli {

namespace HWY_NAMESPACE {

const hwy::HWY_NAMESPACE::ScalableTag<float> d;
using Vec = hwy::HWY_NAMESPACE::Vec<decltype(d)>;

template <typename T>
hwy::AlignedNDArray<float, 2> HwyWindowMean(size_t num_steps,
                                            size_t num_channels,
                                            size_t step_window,
                                            size_t channel_window,
                                            T input_loader) {
  hwy::AlignedNDArray<float, 2> tmp_a({num_steps, num_channels});
  hwy::AlignedNDArray<float, 2> tmp_b({num_steps, num_channels});

  // Populate tmp_b with prefix sums across the step axis.
  {
    float* channel_prefix_sum_data = tmp_b[{0}].data();
    for (size_t channel_index = 0; channel_index < num_channels;
         channel_index += Lanes(d)) {
      Store(input_loader(0, channel_index), d,
            channel_prefix_sum_data + channel_index);
    }
  }
  for (size_t step_index = 1; step_index < num_steps; ++step_index) {
    float* channel_prefix_sum_data = tmp_b[{step_index}].data();
    const float* channel_prev_prefix_sum_data = tmp_b[{step_index - 1}].data();
    for (size_t channel_index = 0; channel_index < num_channels;
         channel_index += Lanes(d)) {
      Store(Add(input_loader(step_index, channel_index),
                Load(d, channel_prev_prefix_sum_data + channel_index)),
            d, channel_prefix_sum_data + channel_index);
    }
  }

  // Populate tmp_a with windowed sums across the step axis using the prefix
  // sums in tmp_b.
  // Simply copy the step_window first rows from tmp_b to tmp_a.
  hwy::CopyBytes(tmp_b.data(), tmp_a.data(),
                 step_window * tmp_a.memory_shape()[1] * sizeof(float));
  // Then compute windowed sums by subtracting prefix sums from each other.
  for (size_t step_index = step_window; step_index < num_steps; ++step_index) {
    const float* curr_window_sum_data = tmp_b[{step_index}].data();
    const float* prev_window_sum_data =
        tmp_b[{step_index - step_window}].data();
    float* channel_window_sum_data = tmp_a[{step_index}].data();
    for (size_t channel_index = 0; channel_index < num_channels;
         channel_index += Lanes(d)) {
      Store(Sub(Load(d, curr_window_sum_data + channel_index),
                Load(d, prev_window_sum_data + channel_index)),
            d, channel_window_sum_data + channel_index);
    }
  }

  for (size_t step_index = 0; step_index < num_steps; ++step_index) {
    // Populate tmp_b with prefix sums across the channel axis of the windowed
    // sums across the step axis in tmp_a.
    {
      const float* channel_window_sum_data = tmp_a[{step_index}].data();
      float* step_prefix_sum_data = tmp_b[{step_index}].data();
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
      const float* step_prefix_sum_data = tmp_b[{step_index}].data();
      float* step_window_sum_data = tmp_a[{step_index}].data();
      hwy::CopyBytes(step_prefix_sum_data, step_window_sum_data,
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
  const Vec reciprocal = Set(d, 1.0 / (step_window * channel_window));
  for (size_t step_index = 0; step_index < num_steps; ++step_index) {
    float* result_data = tmp_a[{step_index}].data();
    for (size_t channel_index = 0; channel_index < num_channels;
         channel_index += Lanes(d)) {
      Store(Mul(reciprocal, Load(d, result_data + channel_index)), d,
            result_data + channel_index);
    }
  }

  return tmp_a;
}

hwy::AlignedNDArray<float, 2> HwyWindowMeanArray(
    const hwy::AlignedNDArray<float, 2>& source, size_t step_window,
    size_t channel_window) {
  const float* source_data = source.data();
  size_t row_size = source.memory_shape()[1];
  return HwyWindowMean(
      source.shape()[0], source.shape()[1], step_window, channel_window,
      [source_data, row_size](size_t step_index, size_t channel_index) -> Vec {
        return Load(d, source_data + step_index * row_size + channel_index);
      });
}

float HwyNSIM(const hwy::AlignedNDArray<float, 2>& a,
              const hwy::AlignedNDArray<float, 2>& b,
              const std::vector<std::pair<size_t, size_t>>& time_pairs,
              size_t step_window, size_t channel_window) {
  const size_t num_channels = a.shape()[1];
  const size_t num_steps = time_pairs.size();

  hwy::AlignedNDArray<float, 2> mean_a =
      HwyWindowMean(num_steps, num_channels, step_window, channel_window,
                    [&](size_t step_index, size_t channel_index) -> Vec {
                      return Load(d, a[{time_pairs[step_index].first}].data() +
                                         channel_index);
                    });
  hwy::AlignedNDArray<float, 2> mean_b =
      HwyWindowMean(num_steps, num_channels, step_window, channel_window,
                    [&](size_t step_index, size_t channel_index) -> Vec {
                      return Load(d, b[{time_pairs[step_index].second}].data() +
                                         channel_index);
                    });
  // NB: This computes is (value - mean) using the mean computed for the window
  // at the same position as the value, so that each value gets a different mean
  // subtracted.
  hwy::AlignedNDArray<float, 2> var_a = HwyWindowMean(
      num_steps, num_channels, step_window, channel_window,
      [&](size_t step_index, size_t channel_index) -> Vec {
        const Vec delta = Sub(
            Load(d, a[{time_pairs[step_index].first}].data() + channel_index),
            Load(d, mean_a[{step_index}].data() + channel_index));
        return Mul(delta, delta);
      });
  hwy::AlignedNDArray<float, 2> var_b = HwyWindowMean(
      num_steps, num_channels, step_window, channel_window,
      [&](size_t step_index, size_t channel_index) -> Vec {
        const Vec delta = Sub(
            Load(d, b[{time_pairs[step_index].second}].data() + channel_index),
            Load(d, mean_b[{step_index}].data() + channel_index));
        return Mul(delta, delta);
      });
  hwy::AlignedNDArray<float, 2> cov = HwyWindowMean(
      num_steps, num_channels, step_window, channel_window,
      [&](size_t step_index, size_t channel_index) -> Vec {
        const Vec delta_a = Sub(
            Load(d, a[{time_pairs[step_index].first}].data() + channel_index),
            Load(d, mean_a[{step_index}].data() + channel_index));
        const Vec delta_b = Sub(
            Load(d, b[{time_pairs[step_index].second}].data() + channel_index),
            Load(d, mean_b[{step_index}].data() + channel_index));
        return Mul(delta_a, delta_b);
      });
  const Vec two = Set(d, 2.0);
  const Vec C1 = Set(d, 0.19863327786546683);
  const Vec C3 = Set(d, 0.17538360286546675);
  float nsim_sum = 0.0;
  const Vec num_channels_vec = Set(d, num_channels);
  const Vec zero = Zero(d);
  for (size_t step_index = 0; step_index < num_steps; ++step_index) {
    for (size_t channel_index = 0; channel_index < num_channels;
         channel_index += Lanes(d)) {
      const Vec mean_a_vec =
          Load(d, mean_a[{step_index}].data() + channel_index);
      const Vec mean_b_vec =
          Load(d, mean_b[{step_index}].data() + channel_index);
      const Vec std_a_vec =
          Sqrt(Load(d, var_a[{step_index}].data() + channel_index));
      const Vec std_b_vec =
          Sqrt(Load(d, var_b[{step_index}].data() + channel_index));
      const Vec cov_vec = Load(d, cov[{step_index}].data() + channel_index);
      const Vec intensity = Div(
          MulAdd(two, Mul(mean_a_vec, mean_b_vec), C1),
          MulAdd(mean_a_vec, mean_a_vec, MulAdd(mean_b_vec, mean_b_vec, C1)));
      const Vec structure =
          Div(Add(cov_vec, C3), MulAdd(std_a_vec, std_b_vec, C3));
      const Vec channel_index_vec = Iota(d, channel_index);
      const Vec nsim = IfThenElse(Lt(channel_index_vec, num_channels_vec),
                                  Mul(intensity, structure), zero);
      nsim_sum += ReduceSum(d, nsim);
    }
  }
  return nsim_sum / static_cast<float>(num_steps * num_channels);
}

}  // namespace HWY_NAMESPACE

}  // namespace zimtohrli
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace zimtohrli {

HWY_EXPORT(HwyWindowMeanArray);
HWY_EXPORT(HwyNSIM);

hwy::AlignedNDArray<float, 2> WindowMean(
    const hwy::AlignedNDArray<float, 2>& source, size_t step_window,
    size_t channel_window) {
  CHECK_GE(source.shape()[0], step_window);
  CHECK_GE(source.shape()[1], channel_window);
  return HWY_DYNAMIC_DISPATCH(HwyWindowMeanArray)(source, step_window,
                                                  channel_window);
}

float NSIM(const hwy::AlignedNDArray<float, 2>& a,
           const hwy::AlignedNDArray<float, 2>& b,
           const std::vector<std::pair<size_t, size_t>>& time_pairs,
           size_t step_window, size_t channel_window) {
  CHECK_GT(a.shape()[0], 0);
  CHECK_GT(b.shape()[0], 0);
  CHECK_GT(a.shape()[1], 0);
  CHECK_GT(b.shape()[1], 0);
  CHECK_GT(step_window, 0);
  CHECK_GT(channel_window, 0);
  return HWY_DYNAMIC_DISPATCH(HwyNSIM)(a, b, time_pairs, step_window,
                                       channel_window);
}

}  // namespace zimtohrli

#endif  // HWY_ONCE
