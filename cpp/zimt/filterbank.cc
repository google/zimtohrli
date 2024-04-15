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

#include "zimt/filterbank.h"

#include <cstddef>
#include <cstring>
#include <vector>

#include "absl/log/check.h"
#include "absl/numeric/bits.h"
#include "hwy/aligned_allocator.h"
#include "hwy/base.h"
#include "zimt/elliptic.h"

// This file uses a lot of magic from the SIMD library Highway.
// In simplified terms, it will compile the code for multiple architectures
// using the "foreach_target.h" header file, and use the special namespace
// convention HWY_NAMESPACE to find the code to adapt to the SIMD functions,
// which are then called via HWY_DYNAMIC_DISPATCH. This leads to a lot of
// hard-to-explain Highway-related conventions being followed, like this here
// #define that makes this entire file be included by Highway in the process of
// building.
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "zimt/filterbank.cc"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
// Must come after foreach_target.h to avoid redefinition errors.
#include "hwy/highway.h"

// This is Highway magic conventions.
HWY_BEFORE_NAMESPACE();
namespace zimtohrli {
namespace HWY_NAMESPACE {

const hwy::HWY_NAMESPACE::ScalableTag<float> d;
using Vec = hwy::HWY_NAMESPACE::Vec<decltype(d)>;

void HwyComputeReciprocal(hwy::Span<const float> coeffs,
                          hwy::Span<float> inverted_coeffs) {
  CHECK_EQ(coeffs.size(), inverted_coeffs.size());
  const auto one = Set(d, 1.0f);
  int index;
  for (index = 0; index + Lanes(d) <= coeffs.size(); index += Lanes(d)) {
    const auto coeff = Load(d, coeffs.data() + index);
    Store(Div(one, coeff), d, inverted_coeffs.data() + index);
  }
  if (index < coeffs.size()) {
    const size_t max_lanes = coeffs.size() - index;
    const auto scale = LoadN(d, coeffs.data() + index, max_lanes);
    StoreN(Div(one, scale), d, inverted_coeffs.data() + index, max_lanes);
  }
}

void HwyFilter(const hwy::AlignedNDArray<float, 3>& b_coeffs,
               const hwy::AlignedNDArray<float, 3>& a_coeffs,
               hwy::AlignedNDArray<float, 3>& x_buffer,
               hwy::AlignedNDArray<float, 3>& y_buffer,
               hwy::Span<const float> input,
               hwy::AlignedNDArray<float, 2>& output,
               size_t& global_sample_index) {
  const size_t num_sections = b_coeffs.shape()[0];
#if HWY_IS_DEBUG_BUILD
  CHECK_EQ(num_sections, a_coeffs.shape()[0]);
  CHECK_EQ(num_sections, x_buffer.shape()[0]);
  CHECK_EQ(num_sections, y_buffer.shape()[0]);
#endif

  const size_t num_b_coeffs = b_coeffs.shape()[1];
#if HWY_IS_DEBUG_BUILD
  CHECK_GT(num_b_coeffs, 0);
#endif

  const size_t num_a_coeffs = a_coeffs.shape()[1];
#if HWY_IS_DEBUG_BUILD
  CHECK_GT(num_a_coeffs, 0);
#endif

  const size_t num_filters = b_coeffs.shape()[2];
#if HWY_IS_DEBUG_BUILD
  CHECK_GT(num_filters, 0);
  CHECK_EQ(num_filters, a_coeffs.shape()[2]);
  CHECK_EQ(num_filters, x_buffer.shape()[2]);
  CHECK_EQ(num_filters, y_buffer.shape()[2]);
  CHECK_EQ(num_filters, output.shape()[1]);
#endif

  // Using memory_shape instead of shape to get the actual padded-to-lanes
  // shape.
  const size_t padded_num_filters = b_coeffs.memory_shape()[2];
#if HWY_IS_DEBUG_BUILD
  CHECK_GT(padded_num_filters, 0);
  CHECK_EQ(padded_num_filters, a_coeffs.memory_shape()[2]);
  CHECK_EQ(padded_num_filters, x_buffer.memory_shape()[2]);
  CHECK_EQ(padded_num_filters, y_buffer.memory_shape()[2]);
  CHECK_EQ(padded_num_filters, output.memory_shape()[1]);
#endif

  const size_t num_samples = output.shape()[0];
#if HWY_IS_DEBUG_BUILD
  CHECK_GT(num_samples, 0);
  CHECK_EQ(num_samples, input.size());
#endif

  const size_t x_buffer_length = x_buffer.shape()[1];
  const size_t y_buffer_length = y_buffer.shape()[1];
#if HWY_IS_DEBUG_BUILD
  CHECK_EQ(x_buffer_length, absl::bit_ceil(x_buffer_length));
  CHECK_EQ(y_buffer_length, absl::bit_ceil(y_buffer_length));
#endif

  const float* input_data = input.data();
  float* output_data = output.data();
  float* x_buffer_data = x_buffer.data();
  float* y_buffer_data = y_buffer.data();
  const float* a_coeffs_data = a_coeffs.data();
  const float* b_coeffs_data = b_coeffs.data();

  const size_t b_coeff_values_per_section = num_b_coeffs * padded_num_filters;
  const size_t a_coeff_values_per_section = num_a_coeffs * padded_num_filters;
  const size_t x_buffer_values_per_section =
      x_buffer_length * padded_num_filters;
  const size_t y_buffer_values_per_section =
      y_buffer_length * padded_num_filters;

  // Returns the address to the x_buffer at (b_coeff_index +
  // global_sample_index) % x_buffer.shape()[0].
  const auto circular_x = [&](size_t sample_index, size_t section_offset,
                              size_t b_coeff_index) HWY_ATTR {
    return x_buffer_data + section_offset +
           (padded_num_filters *
            ((global_sample_index + x_buffer_length - b_coeff_index) &
             (x_buffer_length - 1)));
  };

  // Returns the address to the y_buffer at (a_coeff_index +
  // global_sample_index) % y_buffer.size().
  const auto circular_y = [&](size_t sample_index, size_t section_offset,
                              size_t a_coeff_index) HWY_ATTR {
    return y_buffer_data + section_offset +
           (padded_num_filters *
            ((global_sample_index + y_buffer_length - a_coeff_index) &
             (y_buffer_length - 1)));
  };

  for (size_t sample_index = 0; sample_index < num_samples; ++sample_index) {
    for (size_t section_index = 0; section_index < num_sections;
         ++section_index) {
      const size_t x_section_offset =
          (x_buffer_values_per_section * section_index);
      const size_t y_section_offset =
          (y_buffer_values_per_section * section_index);
      const size_t b_section_offset =
          (b_coeff_values_per_section * section_index);
      const size_t a_section_offset =
          (a_coeff_values_per_section * section_index);
      for (size_t filter_index = 0; filter_index < num_filters;
           filter_index += Lanes(d)) {
        if (section_index == 0) {
          // Store this input sample in x_buffer as current input.
          Store(Set(d, *(input_data + sample_index)), d,
                circular_x(sample_index, 0, 0) + filter_index);
        }
        Vec numerator_out = Zero(d);
        for (size_t b_coeff_index = 0; b_coeff_index < num_b_coeffs;
             ++b_coeff_index) {
          const Vec b_coeff =
              Load(d, b_coeffs_data + b_section_offset +
                          (padded_num_filters * b_coeff_index) + filter_index);
          const Vec x = Load(
              d, circular_x(sample_index, x_section_offset, b_coeff_index) +
                     filter_index);
          numerator_out = MulAdd(b_coeff, x, numerator_out);
        }

        Vec denominator_out = Zero(d);
        for (size_t a_coeff_index = 1; a_coeff_index < num_a_coeffs;
             ++a_coeff_index) {
          const Vec a_coeff =
              Load(d, a_coeffs_data + a_section_offset +
                          (padded_num_filters * a_coeff_index) + filter_index);
          const Vec y = Load(
              d, circular_y(sample_index, y_section_offset, a_coeff_index) +
                     filter_index);
          denominator_out = MulAdd(a_coeff, y, denominator_out);
        }

        const Vec scale =
            Load(d, a_coeffs_data + a_section_offset + filter_index);
        const Vec result = Mul(scale, Sub(numerator_out, denominator_out));

        // Store result in output buffer for next sample-step.
        Store(result, d,
              circular_y(sample_index, y_section_offset, 0) + filter_index);
        if (section_index + 1 < num_sections) {
          // Store result in input buffer for next section-step.
          Store(
              result, d,
              circular_x(sample_index,
                         (section_index + 1) * x_buffer_values_per_section, 0) +
                  filter_index);
        } else {
          // This was the last section, and we have computed the final result
          // for this filter-step, write it to output.
          Store(
              result, d,
              output_data + (padded_num_filters * sample_index) + filter_index);
        }
      }
    }
    ++global_sample_index;
  }
}

}  // namespace HWY_NAMESPACE
}  // namespace zimtohrli
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace zimtohrli {

HWY_EXPORT(HwyComputeReciprocal);
HWY_EXPORT(HwyFilter);

Filterbank::Filterbank(const std::vector<std::vector<BACoeffs>>& filters)
    : b_coeffs_({filters.front().size(),
                 filters.front().front().b_coeffs.size(), filters.size()}),
      a_coeffs_({filters.front().size(),
                 filters.front().front().a_coeffs.size(), filters.size()}),
      x_buffer_shape_({filters.front().size(),
                       absl::bit_ceil(filters.front().front().b_coeffs.size()),
                       filters.size()}),
      y_buffer_shape_({filters.front().size(),
                       absl::bit_ceil(filters.front().front().a_coeffs.size()),
                       filters.size()}) {
#if HWY_IS_DEBUG_BUILD
  CHECK_GT(filters.size(), 0);
#endif
  int num_sections = -1;
  for (size_t filter_index = 0; filter_index < filters.size(); ++filter_index) {
    const std::vector<BACoeffs>& filter = filters[filter_index];
#if HWY_IS_DEBUG_BUILD
    CHECK_GT(filter.size(), 0);
#endif
    if (num_sections == -1) {
      num_sections = filter.size();
#if HWY_IS_DEBUG_BUILD
    } else {
      CHECK_EQ(num_sections, filter.size());
#endif
    }
    for (size_t section_index = 0; section_index < num_sections;
         ++section_index) {
      const BACoeffs& coeffs = filter[section_index];
#if HWY_IS_DEBUG_BUILD
      CHECK_EQ(coeffs.b_coeffs.size(), b_coeffs_.shape()[1]);
      CHECK_EQ(coeffs.a_coeffs.size(), a_coeffs_.shape()[1]);
#endif
      for (size_t coeff_index = 0; coeff_index < b_coeffs_.shape()[1];
           ++coeff_index) {
        b_coeffs_[{section_index, coeff_index}][filter_index] =
            coeffs.b_coeffs[coeff_index];
      }
      for (size_t coeff_index = 0; coeff_index < a_coeffs_.shape()[1];
           ++coeff_index) {
        a_coeffs_[{section_index, coeff_index}][filter_index] =
            coeffs.a_coeffs[coeff_index];
      }
    }
  }
  // Invert the a[0] coeff early so that we can multiply instead of divide by
  // it.
  for (size_t section_index = 0; section_index < num_sections;
       ++section_index) {
    HWY_DYNAMIC_DISPATCH(HwyComputeReciprocal)
    (a_coeffs_[{section_index, 0}], a_coeffs_[{section_index, 0}]);
  }
}

void Filterbank::Filter(hwy::Span<const float> input,
                        hwy::AlignedNDArray<float, 2>& output) const {
  FilterbankState new_state = NewState();
  Filter(input, new_state, output);
}

void Filterbank::Filter(hwy::Span<const float> input, FilterbankState& state,
                        hwy::AlignedNDArray<float, 2>& output) const {
#if HWY_IS_DEBUG_BUILD
  CHECK_EQ(input.size(), output.shape()[0]);
  CHECK_EQ(output.shape()[1], a_coeffs_.shape()[2]);
  // Checking actual buffer shape instead of intended number of filters, since
  // that's what will actually be processed.
  CHECK_EQ(output.memory_shape()[1], a_coeffs_.memory_shape()[2]);
#endif
  HWY_DYNAMIC_DISPATCH(HwyFilter)
  (b_coeffs_, a_coeffs_, state.x_buffer, state.y_buffer, input, output,
   state.global_sample_index);
}

size_t Filterbank::Size() const { return b_coeffs_.shape()[2]; }

FilterbankState Filterbank::NewState() const {
  return {.x_buffer = hwy::AlignedNDArray<float, 3>(x_buffer_shape_),
          .y_buffer = hwy::AlignedNDArray<float, 3>(y_buffer_shape_),
          .global_sample_index = 0};
}

}  // namespace zimtohrli

#endif  // HWY_ONCE
