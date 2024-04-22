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

const std::array<float, 4> params = {3.439e+00, -4.138e-02, 3.008e+00,
                                     -1.354e-01};

namespace {

float sigmoid(float x) { return 1 / (1 + std::exp(-x)); }

}  // namespace

// Optimized using `mos_mapping.ipynb`.
float MOSFromZimtohrli(float zimtohrli_distance) {
  return 1 + 2 * (sigmoid(params[0] + params[1] * zimtohrli_distance) +
                  sigmoid(params[2] + params[3] * zimtohrli_distance));
}

}  // namespace zimtohrli