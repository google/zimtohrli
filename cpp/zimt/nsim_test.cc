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

#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "benchmark/benchmark.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
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
  EXPECT_NEAR(
      NSIM(spec_a, spec_b, {{0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}}, 3, 3),
      0.97899121, 1e-7);

  Spectrogram spec_c(5, 5, {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                            13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
  EXPECT_THAT(
      NSIM(spec_a, spec_c, {{0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}}, 3, 3),
      1.0);
}

TEST(Zimtohrli, DistanceWithoutDtwReturnsZeroForIdenticalSpectrograms) {
  Spectrogram spec_a(10, 5);
  for (size_t step = 0; step < 10; ++step) {
    for (size_t dim = 0; dim < 5; ++dim) {
      spec_a[step][dim] = static_cast<float>(step * 5 + dim);
    }
  }
  Zimtohrli z;
  EXPECT_NEAR(z.DistanceWithoutDtw(spec_a, spec_a), 0.0f, 1e-5);
}

TEST(Zimtohrli, DistanceWithoutDtwReturnsPositiveForDifferentSpectrograms) {
  Spectrogram spec_a(10, 5);
  Spectrogram spec_b(10, 5);
  for (size_t step = 0; step < 10; ++step) {
    for (size_t dim = 0; dim < 5; ++dim) {
      spec_a[step][dim] = static_cast<float>(step * 5 + dim);
      spec_b[step][dim] = static_cast<float>(step * 5 + dim + 1);
    }
  }
  Zimtohrli z;
  EXPECT_GT(z.DistanceWithoutDtw(spec_a, spec_b), 0.0f);
}

TEST(Zimtohrli,
     DistanceWithoutDtwReturnsDistanceWithinBoundsForScaledSpectrograms) {
  Spectrogram spec_a(10, 5);
  Spectrogram spec_b(10, 5);
  for (size_t step = 0; step < 10; ++step) {
    for (size_t dim = 0; dim < 5; ++dim) {
      spec_a[step][dim] = static_cast<float>(step * 5 + dim);
      spec_b[step][dim] = static_cast<float>(step * 5 + dim) * 2.0f;
    }
  }
  Zimtohrli z;
  float dist = z.DistanceWithoutDtw(spec_a, spec_b);
  EXPECT_NEAR(dist, 0.000929296f, 1e-7);
}

TEST(Zimtohrli, DistanceWithoutDtwRespectsCustomStepWindow) {
  Spectrogram spec_a(10, 5);
  for (size_t step = 0; step < 10; ++step) {
    for (size_t dim = 0; dim < 5; ++dim) {
      spec_a[step][dim] = static_cast<float>(step * 5 + dim);
    }
  }
  Zimtohrli z;
  EXPECT_NEAR(z.DistanceWithoutDtw(spec_a, spec_a, 3), 0.0f, 1e-5);
}

TEST(Zimtohrli, DistanceWithoutDtwMatchesDistanceForAlignedSignals) {
  Spectrogram spec_a(10, 5);
  Spectrogram spec_b(10, 5);
  for (size_t step = 0; step < 10; ++step) {
    for (size_t dim = 0; dim < 5; ++dim) {
      spec_a[step][dim] = static_cast<float>(step * 5 + dim);
      spec_b[step][dim] = static_cast<float>(step * 5 + dim) + 0.1f;
    }
  }
  Zimtohrli z;
  float dist_dtw = z.Distance(spec_a, spec_b);

  // Re-initialize spec_a and spec_b because they were mutated by z.Distance.
  for (size_t step = 0; step < 10; ++step) {
    for (size_t dim = 0; dim < 5; ++dim) {
      spec_a[step][dim] = static_cast<float>(step * 5 + dim);
      spec_b[step][dim] = static_cast<float>(step * 5 + dim) + 0.1f;
    }
  }

  float dist_aligned = z.DistanceWithoutDtw(spec_a, spec_b);
  EXPECT_NEAR(dist_dtw, dist_aligned, 1e-5);
}

TEST(Zimtohrli, DistanceDoesNotCrashOnShortInputs) {
  Spectrogram spec_a(3, 5);
  Spectrogram spec_b(3, 5);
  for (size_t step = 0; step < 3; ++step) {
    for (size_t dim = 0; dim < 5; ++dim) {
      spec_a[step][dim] = static_cast<float>(step * 5 + dim);
      spec_b[step][dim] = static_cast<float>(step * 5 + dim) + 0.1f;
    }
  }
  Zimtohrli z;
  float dist = z.Distance(spec_a, spec_b);
  EXPECT_NEAR(dist, 4.22000885e-05f, 1e-9);
}

TEST(Zimtohrli, DistanceHandlesEmptySpectrogramsSafely) {
  Spectrogram spec_a(0, 5);
  Spectrogram spec_b(0, 5);
  Zimtohrli z;
  // Empty spectrograms should return maximal distance without crashing
  float dist = z.DistanceWithoutDtw(spec_a, spec_b);
  EXPECT_EQ(dist, 1.0f);
}

TEST(Zimtohrli, DistanceHandlesEmptySpectrogramsSafelyWithDtw) {
  Spectrogram spec_a(0, 5);
  Spectrogram spec_b(0, 5);
  Zimtohrli z;
  // Verifies that empty spectrograms are handled safely, returning maximal
  // distance without triggering out-of-bounds writes in the DTW CostMatrix.
  float dist = z.Distance(spec_a, spec_b);
  EXPECT_EQ(dist, 1.0f);
}

TEST(Zimtohrli, DistanceHandlesSilentSpectrogramsSafely) {
  Spectrogram spec_a(10, 5);
  Spectrogram spec_b(10, 5);
  // spec_a is silent (all zeros)
  for (size_t step = 0; step < 10; ++step) {
    for (size_t dim = 0; dim < 5; ++dim) {
      spec_a[step][dim] = 0.0f;
      spec_b[step][dim] = static_cast<float>(step * 5 + dim) + 1.0f;
    }
  }
  Zimtohrli z;
  // Verifies that comparing a silent spectrogram with a non-silent one
  // is handled safely without producing NaNs during energy rescaling.
  float dist = z.Distance(spec_a, spec_b);
  EXPECT_FALSE(std::isnan(dist));
  EXPECT_GT(dist, 0.0f);
}

TEST(Zimtohrli, DistanceHandlesZeroWindowSizeSafely) {
  Spectrogram spec_a(10, 5);
  Spectrogram spec_b(10, 5);
  for (size_t step = 0; step < 10; ++step) {
    for (size_t dim = 0; dim < 5; ++dim) {
      spec_a[step][dim] = static_cast<float>(step * 5 + dim);
      spec_b[step][dim] = static_cast<float>(step * 5 + dim);
    }
  }
  Zimtohrli z;
  // Custom window size = 0 should be handled safely without division-by-zero
  float dist = z.DistanceWithoutDtw(spec_a, spec_b, 0);
  EXPECT_EQ(dist, 1.0f);
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
  state.SetItemsProcessed(spec_a.size() * state.iterations());
}
BENCHMARK_RANGE(BM_NSIM, 1, 60);

}  // namespace

}  // namespace zimtohrli
