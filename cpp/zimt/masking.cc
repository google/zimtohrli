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

#include "zimt/masking.h"

#include <cmath>
#include <cstddef>

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
#define HWY_TARGET_INCLUDE "zimt/masking.cc"
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

void HwyToDb(const hwy::AlignedNDArray<float, 2>& energy_channels_linear,
             float full_scale_sine_db, float epsilon,
             hwy::AlignedNDArray<float, 2>& energy_channels_db) {
  const Vec epsilon_vec = Set(d, epsilon);
  const Vec ten_vec = Set(d, 10);
  const Vec full_scale_sine_db_vec = Set(d, full_scale_sine_db);
  const size_t num_samples = energy_channels_linear.shape()[0];
  const size_t num_channels = energy_channels_linear.shape()[1];
  for (size_t sample_index = 0; sample_index < num_samples; ++sample_index) {
    for (size_t channel_index = 0; channel_index < num_channels;
         channel_index += Lanes(d)) {
      Store(
          MulAdd(
              ten_vec,
              Log10(d,
                    Add(epsilon_vec,
                        Load(d, energy_channels_linear[{sample_index}].data() +
                                    channel_index))),
              full_scale_sine_db_vec),
          d, energy_channels_db[{sample_index}].data() + channel_index);
    }
  }
}

void HwyToLinear(const hwy::AlignedNDArray<float, 2>& energy_channels_db,
                 float full_scale_sine_db,
                 hwy::AlignedNDArray<float, 2>& energy_channels_linear) {
  const Vec log_ten_mul_point_1_vec = Set(d, log(10) * 0.1);
  const Vec full_scale_sine_db_vec = Set(d, full_scale_sine_db);
  const size_t num_samples = energy_channels_db.shape()[0];
  const size_t num_channels = energy_channels_db.shape()[1];
  for (size_t sample_index = 0; sample_index < num_samples; ++sample_index) {
    for (size_t channel_index = 0; channel_index < num_channels;
         channel_index += Lanes(d)) {
      // Since Highway doesn't have any pow equivalent, this is done via exp+log
      // instead.
      //
      // y = 10^(x / 10)
      // ln(y) = (x / 10) * ln(10)
      // ln(y) = x * ln(10) * 0.1
      // y = exp(x * ln(10) * 0.1)
      Store(Exp(d, Mul(Sub(Load(d, energy_channels_db[{sample_index}].data() +
                                       channel_index),
                           full_scale_sine_db_vec),
                       log_ten_mul_point_1_vec)),
            d, energy_channels_linear[{sample_index}].data() + channel_index);
    }
  }
}

// Reusable logic to efficiently compute full masking.
struct FullMaskingCalculator {
  FullMaskingCalculator(const Masking& m, float cam_delta_per_channel) {
    cam_delta_vec = Set(d, cam_delta_per_channel);
    decrementing_cam_delta_vec = Neg(Mul(cam_delta_vec, Iota(d, 0)));
    neg_point_one_vec = Set(d, -0.1);
    point_one_vec = Set(d, 0.1);
    max_mask_vec = Set(d, m.max_mask);
    lower_zero_at_20 = Set(d, m.lower_zero_at_20);
    lower_zero_at_80_minus_lower_zero_at_20_divided_by_60 =
        Set(d, (m.lower_zero_at_80 - m.lower_zero_at_20) / 60);
    upper_zero_at_20 = Set(d, m.upper_zero_at_20);
    upper_zero_at_80_minus_upper_zero_at_20_divided_by_60 =
        Set(d, (m.upper_zero_at_80 - m.upper_zero_at_20) / 60);
    zero = Zero(d);
  }

  Vec Calculate(const Vec& masker_level, float channel_offset) {
    const Vec masker_level_minus_max_mask = Sub(masker_level, max_mask_vec);
    const Vec lower_zero =
        Min(neg_point_one_vec,
            Add(lower_zero_at_20,
                Mul(masker_level_minus_max_mask,
                    lower_zero_at_80_minus_lower_zero_at_20_divided_by_60)));
    const Vec lower_slope = Div(masker_level_minus_max_mask, Neg(lower_zero));
    const Vec upper_zero =
        Max(point_one_vec,
            Add(upper_zero_at_20,
                Mul(masker_level_minus_max_mask,
                    upper_zero_at_80_minus_upper_zero_at_20_divided_by_60)));
    const Vec upper_slope = Div(masker_level_minus_max_mask, upper_zero);
    const Vec local_incrementing_cam_delta_vec = Add(
        decrementing_cam_delta_vec, Mul(Set(d, channel_offset), cam_delta_vec));
    return Max(
        zero,
        Min(Mul(lower_slope, Sub(local_incrementing_cam_delta_vec, lower_zero)),
            Mul(upper_slope,
                Sub(upper_zero, local_incrementing_cam_delta_vec))));
  }

  Vec cam_delta_vec;
  Vec decrementing_cam_delta_vec;
  Vec neg_point_one_vec;
  Vec point_one_vec;
  Vec max_mask_vec;
  Vec lower_zero_at_20;
  Vec lower_zero_at_80_minus_lower_zero_at_20_divided_by_60;
  Vec upper_zero_at_20;
  Vec upper_zero_at_80_minus_upper_zero_at_20_divided_by_60;
  Vec zero;
};

void HwyFullMasking(const Masking& m,
                    const hwy::AlignedNDArray<float, 2>& energy_channels_db,
                    float cam_delta,
                    hwy::AlignedNDArray<float, 3>& full_masking_db) {
  FullMaskingCalculator calculator(m, cam_delta);
  const size_t num_samples = energy_channels_db.shape()[0];
  const size_t num_channels = energy_channels_db.shape()[1];
  for (size_t sample_index = 0; sample_index < num_samples; ++sample_index) {
    for (size_t masked_channel_index = 0; masked_channel_index < num_channels;
         ++masked_channel_index) {
      for (size_t masker_channel_index = 0; masker_channel_index < num_channels;
           masker_channel_index += Lanes(d)) {
        Store(calculator.Calculate(
                  Load(d, energy_channels_db[{sample_index}].data() +
                              masker_channel_index),
                  static_cast<float>(masked_channel_index) -
                      static_cast<float>(masker_channel_index)),
              d,
              full_masking_db[{sample_index, masked_channel_index}].data() +
                  masker_channel_index);
      }
    }
  }
}

void HwyCutFullyMasked(const Masking& m,
                       const hwy::AlignedNDArray<float, 2>& energy_channels_db,
                       float cam_delta,
                       hwy::AlignedNDArray<float, 2>& non_masked_db) {
  FullMaskingCalculator full_masking_calculator(m, cam_delta);
  const size_t num_samples = energy_channels_db.shape()[0];
  const size_t num_channels = energy_channels_db.shape()[1];
  for (size_t sample_index = 0; sample_index < num_samples; ++sample_index) {
    const float* energy_channels_db_data =
        energy_channels_db[{sample_index}].data();
    for (size_t probe_channel_index = 0; probe_channel_index < num_channels;
         ++probe_channel_index) {
      float max_masked = std::numeric_limits<float>::min();
      for (size_t masker_channel_index = 0; masker_channel_index < num_channels;
           masker_channel_index += Lanes(d)) {
        const Vec masker_level_db =
            Load(d, energy_channels_db_data + masker_channel_index);
        const Vec full_masking_db = full_masking_calculator.Calculate(
            masker_level_db, static_cast<float>(probe_channel_index) -
                                 static_cast<float>(masker_channel_index));
        max_masked = std::max(max_masked, ReduceMax(d, full_masking_db));
      }
      const float probe_energy_db =
          energy_channels_db_data[probe_channel_index];
      non_masked_db[{sample_index}][probe_channel_index] =
          max_masked > probe_energy_db ? probe_energy_db - max_masked
                                       : probe_energy_db;
    }
  }
}

}  // namespace HWY_NAMESPACE
}  // namespace zimtohrli
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace zimtohrli {

HWY_EXPORT(HwyToDb);
HWY_EXPORT(HwyToLinear);
HWY_EXPORT(HwyFullMasking);
HWY_EXPORT(HwyCutFullyMasked);

void ToDb(const hwy::AlignedNDArray<float, 2>& energy_channels_linear,
          float full_scale_sine_db, float epsilon,
          hwy::AlignedNDArray<float, 2>& energy_channels_db) {
  CHECK_EQ(energy_channels_linear.shape()[0], energy_channels_db.shape()[0]);
  CHECK_EQ(energy_channels_linear.shape()[1], energy_channels_db.shape()[1]);
  HWY_DYNAMIC_DISPATCH(HwyToDb)
  (energy_channels_linear, full_scale_sine_db, epsilon, energy_channels_db);
}

void ToLinear(const hwy::AlignedNDArray<float, 2>& energy_channels_db,
              float full_scale_sine_db,
              hwy::AlignedNDArray<float, 2>& energy_channels_linear) {
  CHECK_EQ(energy_channels_linear.shape()[0], energy_channels_db.shape()[0]);
  CHECK_EQ(energy_channels_linear.shape()[1], energy_channels_db.shape()[1]);
  HWY_DYNAMIC_DISPATCH(HwyToLinear)
  (energy_channels_db, full_scale_sine_db, energy_channels_linear);
}

void Masking::FullMasking(
    const hwy::AlignedNDArray<float, 2>& energy_channels_db, float cam_delta,
    hwy::AlignedNDArray<float, 3>& full_masking_db) const {
  CHECK_GT(cam_delta, 0.0f);
  CHECK_EQ(energy_channels_db.shape()[0], full_masking_db.shape()[0]);
  CHECK_EQ(energy_channels_db.shape()[1], full_masking_db.shape()[2]);
  CHECK_EQ(full_masking_db.shape()[1], full_masking_db.shape()[2]);
  HWY_DYNAMIC_DISPATCH(HwyFullMasking)
  (*this, energy_channels_db, cam_delta, full_masking_db);
}

void Masking::CutFullyMasked(
    const hwy::AlignedNDArray<float, 2>& energy_channels_db, float cam_delta,
    hwy::AlignedNDArray<float, 2>& non_masked_db) const {
  CHECK_EQ(energy_channels_db.shape()[0], non_masked_db.shape()[0]);
  CHECK_EQ(energy_channels_db.shape()[1], non_masked_db.shape()[1]);
  CHECK_GT(cam_delta, 0.0f);

  HWY_DYNAMIC_DISPATCH(HwyCutFullyMasked)
  (*this, energy_channels_db, cam_delta, non_masked_db);
}

}  // namespace zimtohrli

#endif  // HWY_ONCE
