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

#include "zimt/zimtohrli.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <optional>
#include <utility>
#include <vector>

#include "benchmark/benchmark.h"
#include "gtest/gtest.h"

namespace zimtohrli {

namespace {

TEST(Zimtohrli, NormalizeAmplitudeTest) {
  std::vector<float> reference = {0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5};
  std::vector<float> signal = {0.25, 0.25, 0.25, 0.25, 0.25};
  const EnergyAndMaxAbsAmplitude reference_measurements = Measure(reference);
  EXPECT_NEAR(reference_measurements.energy_db_fs, 20 * std::log10(0.5 * 0.5),
              1e-4);
  EXPECT_EQ(reference_measurements.max_abs_amplitude, 0.5);
  const EnergyAndMaxAbsAmplitude signal_measurements =
      NormalizeAmplitude(reference_measurements.max_abs_amplitude, signal);
  for (size_t index = 0; index < signal.size(); ++index) {
    EXPECT_EQ(signal[index], 0.5);
  }
  EXPECT_NEAR(signal_measurements.energy_db_fs, 20 * std::log10(0.5 * 0.5),
              1e-4);
  EXPECT_EQ(signal_measurements.max_abs_amplitude, 0.5);
}

}  // namespace

}  // namespace zimtohrli
