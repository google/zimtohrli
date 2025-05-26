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

#ifndef CPP_ZIMT_SPECTROGRAM_H_
#define CPP_ZIMT_SPECTROGRAM_H_

#include <cstddef>
#include <memory>

namespace {

// A simple buffer of float samples describing a spectrogram with a given number
// of steps and feature dimensions.
// The values buffer is populated like:
// [
//   [sample0_dim0, sample0_dim1, ..., sample0_dimn],
//   [sample1_dim0, sample1_dim1, ..., sample1_dimn],
//   ...,
//   [samplem_dim0, samplem_dim1, ..., samplem_dimn],
// ]
struct Spectrogram {
  Spectrogram(size_t num_steps, size_t num_dims)
      : num_steps(num_steps),
        num_dims(num_dims),
        values(std::make_unique<float[]>(num_steps * num_dims)) {};
  const float* step(size_t n) const { return values.get() + n * num_dims; }
  float* step(size_t n) { return values.get() + n * num_dims; }
  size_t num_steps;
  size_t num_dims;
  std::unique_ptr<float[]> values;
};

}  // namespace

#endif  // CPP_ZIMT_SPECTROGRAM_H_
