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

#include "hwy/aligned_allocator.h"

namespace zimtohrli {

std::vector<std::pair<size_t, size_t>> ChainDTW(
    const hwy::AlignedNDArray<float, 2>& spec_a,
    const hwy::AlignedNDArray<float, 2>& spec_b, size_t window_size) {
  assert(spec_a.shape()[1] == spec_b.shape()[1]);
  Spectrogram spectrogram_a(spec_a.shape()[0], spec_a.shape()[1]);
  Spectrogram spectrogram_b(spec_b.shape()[0], spec_b.shape()[1]);
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

}  // namespace zimtohrli