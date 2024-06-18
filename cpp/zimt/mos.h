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

#ifndef CPP_ZIMT_MOS_H_
#define CPP_ZIMT_MOS_H_

#include <array>

namespace zimtohrli {

// Maps from Zimtohrli distance to MOS.
struct MOSMapper {
  // Returns a _very_approximate_ mean opinion score based on the
  // provided Zimtohrli distance.
  //
  // Computed by:
  // s(x) = params[0] / (params[1] + e^(params[2] * x))
  // MOS = 1 + 4 * s(distance)) / s(0)
  //
  // This is calibrated using default settings of v0.1.5, with a
  // minimum channel bandwidth (zimtohrli::Cam.minimum_bandwidth_hz)
  // of 5Hz and perceptual sample rate
  // (zimtohrli::Distance(..., perceptual_sample_rate, ...) of 100Hz.
  float Map(float zimtohrli_distance) const;

  // Params used when mapping Zimtohrli distance to MOS.
  std::array<float, 3> params = {1.000e+00, -7.449e-09, 3.344e+00};
};

}  // namespace zimtohrli

#endif  // CPP_ZIMT_MOS_H_