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
#include "hwy/aligned_allocator.h"
#include "zimt/nsim.h"

namespace zimtohrli {

namespace {

void CheckEqual(hwy::Span<float> span, std::vector<float> expected) {
  EXPECT_THAT(std::vector<float>(span.begin(), span.end()), expected);
}

TEST(NSIM, WindowMeanTest) {
  hwy::AlignedNDArray<float, 2> ary({5, 5});
  ary[{0}] = {0, 1, 2, 3, 4};
  ary[{1}] = {5, 6, 7, 8, 9};
  ary[{2}] = {10, 11, 12, 13, 14};
  ary[{3}] = {15, 16, 17, 18, 19};
  ary[{4}] = {20, 21, 22, 23, 24};
  hwy::AlignedNDArray<float, 2> mean_3x3 = WindowMeanHwy(ary, 3, 3);
  CheckEqual(mean_3x3[{0}], {0.0, 1.0 / 9.0, 3.0 / 9.0, 6.0 / 9.0, 1.0});
  CheckEqual(mean_3x3[{1}],
             {5.0 / 9.0, 12.0 / 9.0, 21.0 / 9.0, 3.0, 33.0 / 9.0});
  CheckEqual(mean_3x3[{2}], {15.0 / 9.0, 33.0 / 9.0, 6.0, 7.0, 8.0});
  CheckEqual(mean_3x3[{3}], {30.0 / 9.0, 7.0, 11.0, 12.0, 13.0});
  CheckEqual(mean_3x3[{4}], {5.0, 93.0 / 9.0, 16.0, 17.0, 18.0});
}

TEST(NSIM, NSIMTest) {
  hwy::AlignedNDArray<float, 2> a({5, 5});
  a[{0}] = {0, 1, 2, 3, 4};
  a[{1}] = {5, 6, 7, 8, 9};
  a[{2}] = {10, 11, 12, 13, 14};
  a[{3}] = {15, 16, 17, 18, 19};
  a[{4}] = {20, 21, 22, 23, 24};
  hwy::AlignedNDArray<float, 2> b({5, 5});
  b[{0}] = {5, 6, 7, 8, 9};
  b[{1}] = {10, 11, 12, 13, 14};
  b[{2}] = {15, 16, 17, 18, 19};
  b[{3}] = {20, 21, 22, 23, 24};
  b[{4}] = {25, 26, 27, 28, 29};
  EXPECT_THAT(NSIMHwy(a, b, {{0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}}, 3, 3),
              0.745816);
  hwy::AlignedNDArray<float, 2> c({5, 5});
  c[{0}] = {0, 1, 2, 3, 4};
  c[{1}] = {5, 6, 7, 8, 9};
  c[{2}] = {10, 11, 12, 13, 14};
  c[{3}] = {15, 16, 17, 18, 19};
  c[{4}] = {20, 21, 22, 23, 24};
  EXPECT_THAT(NSIMHwy(a, c, {{0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}}, 3, 3), 1);
}

void BM_NSIM(benchmark::State& state) {
  hwy::AlignedNDArray<float, 2> a(
      {static_cast<size_t>(state.range(0)) * 100, 1000});
  std::vector<std::pair<size_t, size_t>> time_pairs(a.shape()[0]);
  for (size_t i = 0; i < time_pairs.size(); i++) {
    time_pairs[i] = {i, i};
  }
  for (auto s : state) {
    NSIMHwy(a, a, time_pairs, 9, 9);
  }
  state.SetItemsProcessed(a.size() * state.iterations());
}
BENCHMARK_RANGE(BM_NSIM, 1, 60);

}  // namespace

}  // namespace zimtohrli