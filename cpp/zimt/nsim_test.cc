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

#include "zimt/nsim.h"

#include <cstddef>
#include <utility>
#include <vector>

#include "benchmark/benchmark.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "zimt/nsim.h"
#include "zimt/zimtohrli.h"

namespace zimtohrli {

namespace {

void CheckEqual(Span<float> span, std::vector<float> expected) {
  for (size_t i = 0; i < span.size; i++) {
    EXPECT_EQ(span[i], expected[i]);
  }
}

TEST(NSIM, WindowMeanTest) {
  Spectrogram spec(5, 5, {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                          13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
  Spectrogram mean_3x3 = WindowMean(
      5, 5, 3, 3, [&](size_t step, size_t dim) { return spec[step][dim]; });
  CheckEqual(mean_3x3[0], {0.0, 1.0 / 9.0, 3.0 / 9.0, 6.0 / 9.0, 1.0});
  CheckEqual(mean_3x3[1], {5.0 / 9.0, 12.0 / 9.0, 21.0 / 9.0, 3.0, 33.0 / 9.0});
  CheckEqual(mean_3x3[2], {15.0 / 9.0, 33.0 / 9.0, 6.0, 7.0, 8.0});
  CheckEqual(mean_3x3[3], {30.0 / 9.0, 7.0, 11.0, 12.0, 13.0});
  CheckEqual(mean_3x3[4], {5.0, 93.0 / 9.0, 16.0, 17.0, 18.0});
}

TEST(NSIM, NSIMTest) {
  Spectrogram spec_a(5, 5, {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                            13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
  Spectrogram spec_b(5, 5, {5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17,
                            18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29});
  EXPECT_THAT(
      NSIM(spec_a, spec_b, {{0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}}, 3, 3),
      0.745816);

  Spectrogram spec_c(5, 5, {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                            13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
  EXPECT_THAT(
      NSIM(spec_a, spec_c, {{0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}}, 3, 3), 1);
}

void BM_NSIM(benchmark::State& state) {
  Spectrogram spec_a(state.range(0) * 100, 1000);
  std::vector<std::pair<size_t, size_t>> time_pairs(spec_a.num_steps);
  for (size_t i = 0; i < time_pairs.size(); i++) {
    time_pairs[i] = {i, i};
  }
  for (auto s : state) {
    NSIM(spec_a, spec_a, time_pairs, 9, 9);
  }
  state.SetItemsProcessed(spec_a.values.size() * state.iterations());
}
BENCHMARK_RANGE(BM_NSIM, 1, 60);

}  // namespace

}  // namespace zimtohrli