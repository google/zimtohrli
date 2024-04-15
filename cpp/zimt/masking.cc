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

void HwyComputeEnergy(const hwy::AlignedNDArray<float, 2>& sample_channels,
                      hwy::AlignedNDArray<float, 2>& energy_channels) {
  const size_t num_out_samples = energy_channels.shape()[0];
  const size_t downscaling = sample_channels.shape()[0] / num_out_samples;
  const size_t num_in_samples = num_out_samples * downscaling;
  const size_t num_channels = sample_channels.shape()[1];
  const auto downscaling_reciprocal_vec = Set(d, 1.0f / downscaling);
  Vec accumulator = Set(d, 0);
  Vec samples = Set(d, 0);
  size_t energy_sample_index = 0;

  for (size_t sample_index = 0; sample_index < num_in_samples;
       sample_index += downscaling) {
    for (size_t channel_index = 0; channel_index < num_channels;
         channel_index += Lanes(d)) {
      accumulator = Set(d, 0);
      for (size_t downscale_index = 0; downscale_index < downscaling;
           ++downscale_index) {
        samples =
            Load(d, sample_channels[{sample_index + downscale_index}].data() +
                        channel_index);
        accumulator = MulAdd(samples, samples, accumulator);
      }
      Store(Mul(downscaling_reciprocal_vec, accumulator), d,
            energy_channels[{energy_sample_index}].data() + channel_index);
    }
    ++energy_sample_index;
  }
}

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

// Reusable logic to efficiently compute masked amount.
struct MaskedAmountCalculator {
  MaskedAmountCalculator(const Masking& m) {
    // This epsilon shouldn't change the behavior of the function, so it's not a
    // parameter.
    epsilon_vec = Set(d, 1e6);
    zero = Zero(d);
    onset_peak = Set(d, m.onset_peak);
    onset_width = Set(d, m.onset_width);
    onset_width_reciprocal = Set(d, 1 / m.onset_width);
    neg_onset_width_reciprocal = Neg(onset_width_reciprocal);
    max_mask = Set(d, m.max_mask);
    neg_max_mask_reciprocal = Set(d, -1 / m.max_mask);
  }

  Vec Calculate(const Vec& full_mask_level, const Vec& probe_level) {
    const Vec onset_delta = Min(epsilon_vec, Sub(onset_peak, full_mask_level));
    const Vec onset_slope = Mul(onset_delta, onset_width_reciprocal);
    const Vec onset_offset =
        MulAdd(full_mask_level, neg_onset_width_reciprocal, full_mask_level);
    const Vec onset_crossing = Add(full_mask_level, onset_width);
    const Vec max_mask_slope =
        Div(Min(onset_peak, full_mask_level),
            Neg(Sub(Add(full_mask_level, max_mask), onset_crossing)));
    const Vec max_mask_offset = Add(full_mask_level, max_mask);
    return Min(
        full_mask_level,
        Max(zero,
            Min(Max(Mul(Sub(probe_level, onset_offset), onset_slope),
                    Mul(Sub(probe_level, max_mask_offset), max_mask_slope)),
                Mul(Sub(Sub(probe_level, full_mask_level), max_mask),
                    Mul(full_mask_level, neg_max_mask_reciprocal)))));
  }

  Vec epsilon_vec;
  Vec zero;
  Vec onset_peak;
  Vec onset_width;
  Vec onset_width_reciprocal;
  Vec neg_onset_width_reciprocal;
  Vec max_mask;
  Vec neg_max_mask_reciprocal;
};

void HwyMaskedAmount(const Masking& m,
                     const hwy::AlignedNDArray<float, 3>& full_masking_db,
                     const hwy::AlignedNDArray<float, 2>& probe_energy_db,
                     hwy::AlignedNDArray<float, 3>& masked_amount_db) {
  MaskedAmountCalculator calculator(m);
  const size_t num_samples = full_masking_db.shape()[0];
  const size_t num_channels = full_masking_db.shape()[2];
  for (size_t sample_index = 0; sample_index < num_samples; ++sample_index) {
    for (size_t masked_channel_index = 0; masked_channel_index < num_channels;
         ++masked_channel_index) {
      for (size_t masker_channel_index = 0; masker_channel_index < num_channels;
           masker_channel_index += Lanes(d)) {
        const Vec full_mask_level = Load(
            d, full_masking_db[{sample_index, masked_channel_index}].data() +
                   masker_channel_index);
        const Vec probe_level =
            Set(d, probe_energy_db[{sample_index}][masked_channel_index]);
        Store(calculator.Calculate(full_mask_level, probe_level), d,
              masked_amount_db[{sample_index, masked_channel_index}].data() +
                  masker_channel_index);
      }
    }
  }
}

void HwyPartialLoudness(const Masking& m,
                        const hwy::AlignedNDArray<float, 2>& energy_channels_db,
                        float cam_delta,
                        hwy::AlignedNDArray<float, 2>& partial_loudness_db) {
  FullMaskingCalculator full_masking_calculator(m, cam_delta);
  MaskedAmountCalculator masked_amount_calculator(m);
  const size_t num_samples = energy_channels_db.shape()[0];
  const size_t num_channels = energy_channels_db.shape()[1];
  const auto log_ten_mul_point_1_vec = Set(d, log(10) * 0.1);
  for (size_t sample_index = 0; sample_index < num_samples; ++sample_index) {
    for (size_t probe_channel_index = 0; probe_channel_index < num_channels;
         ++probe_channel_index) {
      float masked_amount_sum_with_offset = 0;
      const Vec probe_level_db =
          Set(d, energy_channels_db[{sample_index}][probe_channel_index]);
      for (size_t masker_channel_index = 0; masker_channel_index < num_channels;
           masker_channel_index += Lanes(d)) {
        const Vec masker_level_db =
            Load(d, energy_channels_db[{sample_index}].data() +
                        masker_channel_index);
        const Vec full_masking_db = full_masking_calculator.Calculate(
            masker_level_db, static_cast<float>(probe_channel_index) -
                                 static_cast<float>(masker_channel_index));
        const Vec masked_amount_db =
            masked_amount_calculator.Calculate(full_masking_db, probe_level_db);
        // y = 10^(x / 10)
        // ln(y) = (x / 10) * ln(10)
        // ln(y) = x * ln(10) * 0.1
        // y = exp(x * ln(10) * 0.1)
        const auto masked_amounts_linear_with_offset =
            Exp(d, Mul(masked_amount_db, log_ten_mul_point_1_vec));
        masked_amount_sum_with_offset +=
            ReduceSum(d, masked_amounts_linear_with_offset);
      }
      // Subtracting num_channels rounded up to Lanes(d) to compensate for
      // linear 0 being dB 1.
      partial_loudness_db[{sample_index}][probe_channel_index] =
          energy_channels_db[{sample_index}][probe_channel_index] -
          10 * log10(masked_amount_sum_with_offset + 1 -
                     hwy::RoundUpTo(num_channels, Lanes(d)));
    }
  }
}

}  // namespace HWY_NAMESPACE
}  // namespace zimtohrli
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace zimtohrli {

HWY_EXPORT(HwyComputeEnergy);
HWY_EXPORT(HwyToDb);
HWY_EXPORT(HwyToLinear);
HWY_EXPORT(HwyFullMasking);
HWY_EXPORT(HwyMaskedAmount);
HWY_EXPORT(HwyPartialLoudness);

void ComputeEnergy(const hwy::AlignedNDArray<float, 2>& sample_channels,
                   hwy::AlignedNDArray<float, 2>& energy_channels) {
  CHECK_GE(sample_channels.shape()[0], energy_channels.shape()[0]);
  CHECK_EQ(sample_channels.shape()[1], energy_channels.shape()[1]);
  HWY_DYNAMIC_DISPATCH(HwyComputeEnergy)(sample_channels, energy_channels);
}

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

void Masking::MaskedAmount(
    const hwy::AlignedNDArray<float, 3>& full_masking_db,
    const hwy::AlignedNDArray<float, 2>& probe_energy_db,
    hwy::AlignedNDArray<float, 3>& masked_amount_db) const {
  // Check same number of samples in all arguments.
  CHECK_EQ(full_masking_db.shape()[0], probe_energy_db.shape()[0]);
  CHECK_EQ(masked_amount_db.shape()[0], probe_energy_db.shape()[0]);

  // Check same number of channels in all arguments.
  CHECK_EQ(full_masking_db.shape()[1], masked_amount_db.shape()[1]);
  CHECK_EQ(full_masking_db.shape()[1], probe_energy_db.shape()[1]);

  // Check same number of channels in axis 1 and 2.
  CHECK_EQ(full_masking_db.shape()[1], full_masking_db.shape()[2]);
  CHECK_EQ(masked_amount_db.shape()[1], masked_amount_db.shape()[2]);

  // Check same number of channels allocated in the last axes.
  CHECK_EQ(full_masking_db.shape()[2], probe_energy_db.shape()[1]);
  CHECK_EQ(full_masking_db.shape()[2], masked_amount_db.shape()[2]);
  HWY_DYNAMIC_DISPATCH(HwyMaskedAmount)
  (*this, full_masking_db, probe_energy_db, masked_amount_db);
}

void Masking::PartialLoudness(
    const hwy::AlignedNDArray<float, 2>& energy_channels_db, float cam_delta,
    hwy::AlignedNDArray<float, 2>& partial_loudness_db) const {
  CHECK_EQ(energy_channels_db.shape()[0], partial_loudness_db.shape()[0]);
  CHECK_EQ(energy_channels_db.shape()[1], partial_loudness_db.shape()[1]);
  CHECK_GT(cam_delta, 0.0f);

  HWY_DYNAMIC_DISPATCH(HwyPartialLoudness)
  (*this, energy_channels_db, cam_delta, partial_loudness_db);
}

}  // namespace zimtohrli

#endif  // HWY_ONCE
