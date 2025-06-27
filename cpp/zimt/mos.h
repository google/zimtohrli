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

#ifndef CPP_ZIMT_MOS_H_
#define CPP_ZIMT_MOS_H_

#include <array>
#include <cmath>

namespace zimtohrli {

namespace {

const std::array<float, 3> mos_params = {1.000e+00, -5.784e-09, 5.106e+01};

float sigmoid(float x) {
  return mos_params[0] / (mos_params[1] + std::exp(mos_params[2] * x));
}

const float zero_crossing_reciprocal = 1.0 / sigmoid(0);

}  // namespace

// Returns a _very_approximate_ mean opinion score based on the
// provided Zimtohrli distance.
// This is calibrated using default settings of v0.1.5, with a
// minimum channel bandwidth (zimtohrli::Cam.minimum_bandwidth_hz)
// of 5Hz and perceptual sample rate
// (zimtohrli::Distance(..., perceptual_sample_rate, ...) of 100Hz.
float MOSFromZimtohrli(float zimtohrli_distance) {
  return 1.0 + 4.0 * sigmoid(zimtohrli_distance) * zero_crossing_reciprocal;
}

}  // namespace zimtohrli

#endif  // CPP_ZIMT_MOS_H_