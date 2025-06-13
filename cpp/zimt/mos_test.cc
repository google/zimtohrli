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

#include <cstddef>
#include <vector>

#include "gtest/gtest.h"

namespace zimtohrli {

namespace {

TEST(MOS, MOSFromZimtohrli) {
  const std::vector<float> zimt_scores = {0, 0.001, 0.01, 0.02, 0.03, 0.04};
  const std::vector<float> mos = {
      5.0,
      4.4273738861083984,
      1.8532474040985107,
      1.1820077896118164,
      1.0388244390487671,
      1.0082817077636719,
  };
  for (size_t index = 0; index < zimt_scores.size(); ++index) {
    ASSERT_NEAR(MOSFromZimtohrli(zimt_scores[index]), mos[index], 1e-2);
  }
}

}  // namespace

}  // namespace zimtohrli