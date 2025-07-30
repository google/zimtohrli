// Copyright 2025 The Zimtohrli Authors. All Rights Reserved.
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

#ifndef CPP_ZIMT_RESAMPLE_H_
#define CPP_ZIMT_RESAMPLE_H_

#include <cstddef>
#include <type_traits>
#include <vector>

#include "absl/log/check.h"
#include "soxr.h"
#include "zimt/zimtohrli.h"

namespace zimtohrli {

template <typename O, typename I>
std::vector<O> Convert(Span<const I> input) {
  if constexpr (std::is_same<O, I>::value) {
    std::vector<O> result(input.size);
    memcpy(result.data(), input.data, input.size * sizeof(I));
    return result;
  }
  std::vector<O> output(input.size);
  for (size_t sample_index = 0; sample_index < input.size; ++sample_index) {
    output[sample_index] = static_cast<O>(input[sample_index]);
  }
  return output;
}

template <typename T>
inline constexpr soxr_datatype_t SoxrType() {
  if constexpr (std::is_same_v<T, int16_t>) {
    return SOXR_INT16_I;
  } else if constexpr (std::is_same_v<T, int32_t>) {
    return SOXR_INT32_I;
  } else if constexpr (std::is_same_v<T, float>) {
    return SOXR_FLOAT32_I;
  } else if constexpr (std::is_same_v<T, double>) {
    return SOXR_FLOAT64_I;
  } else {
    // This can't be `static_assert(false)`, as explained here:
    // https://devblogs.microsoft.com/oldnewthing/20200311-00/?p=103553
    static_assert(sizeof(T) < 0, "Unsupported type for resampling");
  }
}

template <typename O, typename I>
std::vector<O> Resample(Span<const I> samples, float in_sample_rate,
                        float out_sample_rate) {
  if (in_sample_rate == out_sample_rate) {
    return Convert<O>(samples);
  }

  std::vector<O> result(
      static_cast<size_t>(samples.size * out_sample_rate / in_sample_rate));
  soxr_quality_spec_t quality = soxr_quality_spec(SOXR_VHQ, SOXR_LINEAR_PHASE);
  soxr_io_spec_t io_spec = soxr_io_spec(SoxrType<I>(), SoxrType<O>());
  const soxr_error_t error = soxr_oneshot(
      in_sample_rate, out_sample_rate, 1, samples.data, samples.size, nullptr,
      result.data(), result.size(), nullptr, &io_spec, &quality, nullptr);
  assert(error == 0);
  return result;
}

}  // namespace zimtohrli

#endif  // CPP_ZIMT_RESAMPLE_H_
