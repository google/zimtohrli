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

#include <cstddef>
#include <utility>
#include <vector>

#include "benchmark/benchmark.h"
#include "gtest/gtest.h"
#include "zimt/zimtohrli.h"

namespace zimtohrli {

namespace {

TEST(DTW, DTWTest) {
  Spectrogram spec_a(10, 1, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
  Spectrogram spec_b(10, 1, {0, 1, 2, 3, 3, 4, 5, 6, 8, 9});

  const std::vector<std::pair<size_t, size_t>> got_dtw = DTW(spec_a, spec_b);
  const std::vector<std::pair<size_t, size_t>> expected_dtw = {
      {0, 0}, {1, 1}, {2, 2}, {3, 3}, {3, 4}, {4, 5},
      {5, 6}, {6, 7}, {7, 8}, {8, 8}, {9, 9}};
  EXPECT_EQ(got_dtw, expected_dtw);
}

void BM_DTW(benchmark::State& state) {
  Spectrogram spec_a(state.range(0), 1024);
  Spectrogram spec_b(state.range(0), 1024);
  for (size_t step_index = 0; step_index < spec_a.num_steps; ++step_index) {
    for (size_t channel_index = 0; channel_index < spec_a.num_dims;
         ++channel_index) {
      spec_a[step_index][channel_index] = 1.0;
    }
  }

  for (auto s : state) {
    DTW(spec_a, spec_b);
  }
  state.SetItemsProcessed(state.range(0) * state.iterations());
}
BENCHMARK_RANGE(BM_DTW, 100, 5000);

}  // namespace

}  // namespace zimtohrli
