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

#ifndef CPP_ZIMT_NSIM_H_
#define CPP_ZIMT_NSIM_H_

#include <utility>
#include <vector>

#include "hwy/aligned_allocator.h"

namespace zimtohrli {

// Returns an array shaped exactly like source, where each element is the mean
// of the zero-padded step_window x channel_window rectangle of preceding
// elements.
hwy::AlignedNDArray<float, 2> WindowMean(
    const hwy::AlignedNDArray<float, 2>& source, size_t step_window,
    size_t channel_window);

// Returns a slightly nonstandard version of the NSIM neural structural
// similarity metric between arrays a and b.
//
// step_window and channel_window are the number of time steps and channels in
// the array over which to window the mean, standard deviance, and covariance
// measures in NSIM.
//
// time_pairs is the dynamic time warp computed between array a and array b,
// i.e. pairs of time step indices where array a and array b are considered to
// match each other in time.
//
// See https://doi.org/10.1016/j.specom.2011.09.004 for details.
float NSIM(const hwy::AlignedNDArray<float, 2>& a,
           const hwy::AlignedNDArray<float, 2>& b,
           const std::vector<std::pair<size_t, size_t>>& time_pairs,
           size_t step_window, size_t channel_window);

}  // namespace zimtohrli

#endif  // CPP_ZIMT_NSIM_H_