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

#ifndef CPP_ZIMT_DTW_H_
#define CPP_ZIMT_DTW_H_

#include <cstddef>
#include <utility>
#include <vector>

#include "hwy/aligned_allocator.h"

namespace zimtohrli {

// Computes the DTW (https://en.wikipedia.org/wiki/Dynamic_time_warping)
// between two arrays.
//
// spec_a and spec_b are (num_steps, X)-shaped arrays.
std::vector<std::pair<size_t, size_t>> DTW(
    const hwy::AlignedNDArray<float, 2>& spec_a,
    const hwy::AlignedNDArray<float, 2>& spec_b);

// Computes DTW on window_size segments of spec_a and spec_b, with 50% overlap,
// to find the optimal DTW for the entire specs as long as the warp never
// exceeds window_size / 2.
//
// spec_a and spec_b are (num_steps, X)-shaped arrays.
std::vector<std::pair<size_t, size_t>> ChainDTW(
    const hwy::AlignedNDArray<float, 2>& spec_a,
    const hwy::AlignedNDArray<float, 2>& spec_b, size_t window_size);

}  // namespace zimtohrli

#endif  // CPP_ZIMT_DTW_H_
