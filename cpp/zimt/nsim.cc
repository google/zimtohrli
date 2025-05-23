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

#include "zimt/spectrogram.h"

namespace zimtohrli {

hwy::AlignedNDArray<float, 2> WindowMeanHwy(
    const hwy::AlignedNDArray<float, 2>& source, size_t step_window,
    size_t channel_window) {
  Spectrogram mean =
      WindowMean(source.shape()[0], source.shape()[1], step_window,
                 channel_window, [&](size_t step_index, size_t channel_index) {
                   return source[{step_index}][channel_index];
                 });
  hwy::AlignedNDArray<float, 2> result({mean.num_steps, mean.num_dims});
  for (size_t step_index = 0; step_index < mean.num_steps; ++step_index) {
    hwy::Span<float> dst = result[{step_index}];
    std::memcpy(dst.data(), mean.step(step_index), dst.size() * sizeof(float));
  }
  return result;
}

float NSIMHwy(const hwy::AlignedNDArray<float, 2>& a,
              const hwy::AlignedNDArray<float, 2>& b,
              const std::vector<std::pair<size_t, size_t>>& time_pairs,
              size_t step_window, size_t channel_window) {
  Spectrogram spec_a(a.shape()[0], a.shape()[1]);
  for (size_t step_index = 0; step_index < spec_a.num_steps; ++step_index) {
    std::memcpy(spec_a.step(step_index), a[{step_index}].data(),
                spec_a.num_steps * sizeof(float));
  }
  Spectrogram spec_b(b.shape()[0], b.shape()[1]);
  for (size_t step_index = 0; step_index < spec_b.num_steps; ++step_index) {
    std::memcpy(spec_b.step(step_index), b[{step_index}].data(),
                spec_b.num_steps * sizeof(float));
  }
  return NSIM(spec_a, spec_b, time_pairs, step_window, channel_window);
}

}  // namespace zimtohrli
