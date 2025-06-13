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

#ifndef CPP_ZIMT_RESAMPLE_H_
#define CPP_ZIMT_RESAMPLE_H_

#include <cstddef>
#include <type_traits>
#include <vector>

#include "absl/log/check.h"
#include "samplerate.h"
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

template <typename O, typename I>
std::vector<O> Resample(Span<const I> samples, float in_sample_rate,
                        float out_sample_rate) {
  if (in_sample_rate == out_sample_rate) {
    return Convert<O>(samples);
  }

  const std::vector<float> samples_as_floats = Convert<float>(samples);
  std::vector<float> result_as_floats(
      static_cast<size_t>(samples.size * out_sample_rate / in_sample_rate));
  SRC_DATA resample_data = {
      .data_in = samples_as_floats.data(),
      .data_out = result_as_floats.data(),
      .input_frames = static_cast<long>(samples_as_floats.size()),
      .output_frames = static_cast<long>(result_as_floats.size()),
      .src_ratio = out_sample_rate / in_sample_rate,
  };
  int src_result = src_simple(&resample_data, SRC_SINC_BEST_QUALITY, 1);
  assert_eq(src_result, 0);
  return Convert<O>(
      Span<const float>(result_as_floats.data(), result_as_floats.size()));
}

}  // namespace zimtohrli

#endif  // CPP_ZIMT_RESAMPLE_H_
