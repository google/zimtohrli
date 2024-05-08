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

#include "zimt/mos.h"

#include <array>
#include <cmath>

namespace zimtohrli {

namespace {

const std::array<float, 3> params = {1.000e+00, 7.451e-09, 2.943e+00};

float sigmoid(float x) {
  return params[0] / (params[1] + std::exp(params[2] * x));
}

const float zero_crossing_reciprocal = 1.0 / sigmoid(0);

}  // namespace

// Optimized using `mos_mapping.ipynb`.
float MOSFromZimtohrli(float zimtohrli_distance) {
  return 1.0 + 4.0 * sigmoid(zimtohrli_distance) * zero_crossing_reciprocal;
}

}  // namespace zimtohrli