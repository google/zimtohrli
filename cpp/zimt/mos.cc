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

float sigmoid(const std::array<float, 3>& params, float x) {
  return params[0] / (params[1] + std::exp(params[2] * x));
}

}  // namespace

float MOSMapper::Map(float zimtohrli_distance) const {
  return 1.0 + 4.0 * sigmoid(params, zimtohrli_distance) / sigmoid(params, 0);
}

}  // namespace zimtohrli