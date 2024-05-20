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

#include "zimt/masking.h"

#include <cstddef>
#include <vector>

#include "benchmark/benchmark.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "hwy/aligned_allocator.h"

namespace zimtohrli {

namespace {

void CheckNear(hwy::Span<float> span, std::vector<float> expected) {
  EXPECT_THAT(std::vector<float>(span.begin(), span.end()),
              testing::Pointwise(testing::FloatNear(1e-5), expected));
}

TEST(Masking, ComputeEnergyTest) {
  hwy::AlignedNDArray<float, 2> sample_channels({10, 2});
  for (size_t sample_index = 0; sample_index < sample_channels.shape()[0];
       ++sample_index) {
    sample_channels[{sample_index}] = {
        static_cast<float>(sample_index + 1),
        static_cast<float>((sample_index + 1) * 2)};
  }
  hwy::AlignedNDArray<float, 2> got_energy_downsampled_2x({5, 2});
  ComputeEnergy(sample_channels, got_energy_downsampled_2x);
  CheckNear(got_energy_downsampled_2x[{0}],
            {(1.0 + 4.0) / 2.0, (4.0 + 16.0) / 2.0});
  CheckNear(got_energy_downsampled_2x[{1}],
            {(9.0 + 16.0) / 2.0, (36.0 + 64.0) / 2.0});
  CheckNear(got_energy_downsampled_2x[{2}],
            {(25.0 + 36.0) / 2.0, (100.0 + 144.0) / 2.0});
  CheckNear(got_energy_downsampled_2x[{3}],
            {(49.0 + 64.0) / 2.0, (14.0 * 14.0 + 16.0 * 16.0) / 2.0});
  CheckNear(got_energy_downsampled_2x[{4}],
            {(81.0 + 100.0) / 2.0, (18.0 * 18.0 + 400.0) / 2.0});

  hwy::AlignedNDArray<float, 2> got_energy_downsampled_5x({2, 2});
  ComputeEnergy(sample_channels, got_energy_downsampled_5x);
  CheckNear(got_energy_downsampled_5x[{0}],
            {(1.0 + 4.0 + 9.0 + 16.0 + 25.0) / 5.0,
             (4.0 + 16.0 + 36.0 + 64.0 + 100.0) / 5.0});
  CheckNear(got_energy_downsampled_5x[{1}],
            {(36.0 + 49.0 + 64.0 + 81.0 + 100.0) / 5.0,
             (144.0 + 14.0 * 14.0 + 16.0 * 16.0 + 18.0 * 18.0 + 400.0) / 5.0});
}

void BM_ComputeEnergy(benchmark::State& state) {
  const size_t sample_rate = 48000;
  const hwy::AlignedNDArray<float, 2> sample_channels(
      {sample_rate * state.range(0), 1024});
  hwy::AlignedNDArray<float, 2> energy_channels(
      {100 * static_cast<size_t>(state.range(0)), 1024});
  for (auto s : state) {
    ComputeEnergy(sample_channels, energy_channels);
  }
  state.SetItemsProcessed(sample_channels.size() * state.iterations());
}
BENCHMARK_RANGE(BM_ComputeEnergy, 1, 64);

TEST(Masking, ToDbToLinear) {
  hwy::AlignedNDArray<float, 2> energy_channels_linear({2, 2});
  energy_channels_linear[{0}] = {0.1, 0.01};
  energy_channels_linear[{1}] = {1.0, 10.0};
  hwy::AlignedNDArray<float, 2> energy_channels_db({2, 2});
  ToDb(energy_channels_linear, 80, 0, energy_channels_db);
  CheckNear(energy_channels_db[{0}], {70.0, 60.0});
  CheckNear(energy_channels_db[{1}], {80.0, 90.0});
  ToLinear(energy_channels_db, 90, energy_channels_db);
  CheckNear(energy_channels_db[{0}], {0.01, 0.001});
  CheckNear(energy_channels_db[{1}], {0.1, 1.0});
}

TEST(Masking, FullMasking) {
  hwy::AlignedNDArray<float, 2> energy_channels({1, 2});
  hwy::AlignedNDArray<float, 3> full_masking({1, 2, 2});
  Masking m;

  // Testing masker with lower frequency than probe.

  // 2 Cam away

  energy_channels[{0}] = {20, 0};
  m.FullMasking(energy_channels, 2, full_masking);
  EXPECT_NEAR((full_masking[{0, 0}][0]), 0, 1e-2) << "No self masking at 20dB";
  EXPECT_NEAR((full_masking[{0, 0}][1]), 0, 1e-2) << "No masking from silence";
  EXPECT_NEAR((full_masking[{0, 1}][0]), 0, 1e-2) << "No masking from 20dB";
  EXPECT_NEAR((full_masking[{0, 1}][1]), 0, 1e-2)
      << "No self masking from silence";

  energy_channels[{0}] = {21, 0};
  m.FullMasking(energy_channels, 2, full_masking);
  EXPECT_NEAR((full_masking[{0, 0}][0]), 1, 1e-2) << "-20dB self masking";
  EXPECT_NEAR((full_masking[{0, 0}][1]), 0, 1e-2) << "No masking from silence";
  EXPECT_NEAR((full_masking[{0, 1}][0]), 0.0625, 1e-2)
      << "Some masking from 20dB @ -2 Cams";
  EXPECT_NEAR((full_masking[{0, 1}][1]), 0, 1e-2)
      << "No self masking from silence";

  energy_channels[{0}] = {60, 0};
  m.FullMasking(energy_channels, 2, full_masking);
  EXPECT_NEAR((full_masking[{0, 0}][0]), 40, 1e-2) << "-20dB self masking";
  EXPECT_NEAR((full_masking[{0, 0}][1]), 0, 1e-2) << "No masking from silence";
  EXPECT_NEAR((full_masking[{0, 1}][0]), 29.0909, 1e-2)
      << "Quite some masking from 60dB @ -2 Cams";
  EXPECT_NEAR((full_masking[{0, 1}][1]), 0, 1e-2)
      << "No self masking from silence";

  energy_channels[{0}] = {80, 0};
  m.FullMasking(energy_channels, 2, full_masking);
  EXPECT_NEAR((full_masking[{0, 0}][0]), 60, 1e-2) << "-20dB self masking";
  EXPECT_NEAR((full_masking[{0, 0}][1]), 0, 1e-2) << "No masking from silence";
  EXPECT_NEAR((full_masking[{0, 1}][0]), 48, 1e-2)
      << "A lot of masking from 80dB @ -2 Cams";
  EXPECT_NEAR((full_masking[{0, 1}][1]), 0, 1e-2)
      << "No self masking from silence";

  // 4 Cam away

  energy_channels[{0}] = {20, 0};
  m.FullMasking(energy_channels, 4, full_masking);
  EXPECT_NEAR((full_masking[{0, 0}][0]), 0, 1e-2) << "No self masking at 20dB";
  EXPECT_NEAR((full_masking[{0, 0}][1]), 0, 1e-2) << "No masking from silence";
  EXPECT_NEAR((full_masking[{0, 1}][0]), 0, 1e-2) << "No masking from 20dB";
  EXPECT_NEAR((full_masking[{0, 1}][1]), 0, 1e-2)
      << "No self masking from silence";

  energy_channels[{0}] = {21, 0};
  m.FullMasking(energy_channels, 4, full_masking);
  EXPECT_NEAR((full_masking[{0, 0}][0]), 1, 1e-2) << "-20dB self masking";
  EXPECT_NEAR((full_masking[{0, 0}][1]), 0, 1e-2) << "No masking from silence";
  EXPECT_NEAR((full_masking[{0, 1}][0]), 0, 1e-2)
      << "No masking from 21dB @ -4 Cams";
  EXPECT_NEAR((full_masking[{0, 1}][1]), 0, 1e-2)
      << "No self masking from silence";

  energy_channels[{0}] = {60, 0};
  m.FullMasking(energy_channels, 4, full_masking);
  EXPECT_NEAR((full_masking[{0, 0}][0]), 40, 1e-2) << "-20dB self masking";
  EXPECT_NEAR((full_masking[{0, 0}][1]), 0, 1e-2) << "No masking from silence";
  EXPECT_NEAR((full_masking[{0, 1}][0]), 18.182, 1e-2)
      << "Some masking from 60dB @ -4 Cams";
  EXPECT_NEAR((full_masking[{0, 1}][1]), 0, 1e-2)
      << "No self masking from silence";

  energy_channels[{0}] = {80, 0};
  m.FullMasking(energy_channels, 4, full_masking);
  EXPECT_NEAR((full_masking[{0, 0}][0]), 60, 1e-2) << "-20dB self masking";
  EXPECT_NEAR((full_masking[{0, 0}][1]), 0, 1e-2) << "No masking from silence";
  EXPECT_NEAR((full_masking[{0, 1}][0]), 36, 1e-2)
      << "A lot of masking from 80dB @ -4 Cams";
  EXPECT_NEAR((full_masking[{0, 1}][1]), 0, 1e-2)
      << "No self masking from silence";

  // Testing masker with higher frequency than probe.

  // 2 Cam away

  energy_channels[{0}] = {0, 20};
  m.FullMasking(energy_channels, 2, full_masking);
  EXPECT_NEAR((full_masking[{0, 0}][0]), 0, 1e-2)
      << "No self masking from silence";
  EXPECT_NEAR((full_masking[{0, 0}][1]), 0, 1e-2) << "No masking from 20dB";
  EXPECT_NEAR((full_masking[{0, 1}][0]), 0, 1e-2) << "No masking from silence";
  EXPECT_NEAR((full_masking[{0, 1}][1]), 0, 1e-2) << "No self masking at 20dB";

  energy_channels[{0}] = {0, 21};
  m.FullMasking(energy_channels, 2, full_masking);
  EXPECT_NEAR((full_masking[{0, 0}][0]), 0, 1e-2)
      << "No self masking from silence";
  EXPECT_NEAR((full_masking[{0, 0}][1]), 0.032258, 1e-2)
      << "A little masking from 20dB @ 2 Cams";
  EXPECT_NEAR((full_masking[{0, 1}][0]), 0, 1e-2) << "No masking from silence";
  EXPECT_NEAR((full_masking[{0, 1}][1]), 1, 1e-2) << "-20dB self masking";

  energy_channels[{0}] = {0, 60};
  m.FullMasking(energy_channels, 2, full_masking);
  EXPECT_NEAR((full_masking[{0, 0}][0]), 0, 1e-2)
      << "No self masking from silence";
  EXPECT_NEAR((full_masking[{0, 0}][1]), 22.857, 1e-2)
      << "Quite some masking from 60dB @ 2 cams";
  EXPECT_NEAR((full_masking[{0, 1}][0]), 0, 1e-2) << "No masking from silence";
  EXPECT_NEAR((full_masking[{0, 1}][1]), 40, 1e-2) << "-20dB self masking";

  energy_channels[{0}] = {0, 80};
  m.FullMasking(energy_channels, 2, full_masking);
  EXPECT_NEAR((full_masking[{0, 0}][0]), 0, 1e-2)
      << "No self masking from silence";
  EXPECT_NEAR((full_masking[{0, 0}][1]), 40, 1e-2)
      << "Quite some masking from 60dB @ 2 cams";
  EXPECT_NEAR((full_masking[{0, 1}][0]), 0, 1e-2) << "No masking from silence";
  EXPECT_NEAR((full_masking[{0, 1}][1]), 60, 1e-2) << "-20dB self masking";

  // 4 Cam away

  energy_channels[{0}] = {0, 20};
  m.FullMasking(energy_channels, 4, full_masking);
  EXPECT_NEAR((full_masking[{0, 0}][0]), 0, 1e-2)
      << "No self masking from silence";
  EXPECT_NEAR((full_masking[{0, 0}][1]), 0, 1e-2) << "No masking from 20dB";
  EXPECT_NEAR((full_masking[{0, 1}][0]), 0, 1e-2) << "No masking from silence";
  EXPECT_NEAR((full_masking[{0, 1}][1]), 0, 1e-2) << "No self masking at 20dB";

  energy_channels[{0}] = {0, 21};
  m.FullMasking(energy_channels, 4, full_masking);
  EXPECT_NEAR((full_masking[{0, 0}][0]), 0, 1e-2)
      << "No self masking from silence";
  EXPECT_NEAR((full_masking[{0, 0}][1]), 0, 1e-2)
      << "No masking from 20dB @ 4 Cams";
  EXPECT_NEAR((full_masking[{0, 1}][0]), 0, 1e-2) << "No masking from silence";
  EXPECT_NEAR((full_masking[{0, 1}][1]), 1, 1e-2) << "-20dB self masking";

  energy_channels[{0}] = {0, 60};
  m.FullMasking(energy_channels, 4, full_masking);
  EXPECT_NEAR((full_masking[{0, 0}][0]), 0, 1e-2)
      << "No self masking from silence";
  EXPECT_NEAR((full_masking[{0, 0}][1]), 5.714, 1e-2)
      << "A bit of some masking from 60dB @ 4 cams";
  EXPECT_NEAR((full_masking[{0, 1}][0]), 0, 1e-2) << "No masking from silence";
  EXPECT_NEAR((full_masking[{0, 1}][1]), 40, 1e-2) << "-20dB self masking";

  energy_channels[{0}] = {0, 80};
  m.FullMasking(energy_channels, 4, full_masking);
  EXPECT_NEAR((full_masking[{0, 0}][0]), 0, 1e-2)
      << "No self masking from silence";
  EXPECT_NEAR((full_masking[{0, 0}][1]), 20, 1e-2)
      << "Quite some masking from 60dB @ 4 cams";
  EXPECT_NEAR((full_masking[{0, 1}][0]), 0, 1e-2) << "No masking from silence";
  EXPECT_NEAR((full_masking[{0, 1}][1]), 60, 1e-2) << "-20dB self masking";
}

TEST(Masking, CutFullyMasked) {
  hwy::AlignedNDArray<float, 2> energy_channels({1, 2});
  hwy::AlignedNDArray<float, 2> non_masked({1, 2});
  Masking m;

  energy_channels[{0}] = {90, 20};
  m.CutFullyMasked(energy_channels, 1, non_masked);
  EXPECT_NEAR((non_masked[{0}][0]), 90, 1e-2) << "No self masking";
  EXPECT_NEAR((non_masked[{0}][1]), -43.82, 1e-2)
      << "20dB fully masked by 90dB";
}

void BM_FullMasking(benchmark::State& state) {
  const size_t sample_rate = 100;
  const hwy::AlignedNDArray<float, 2> energy_channels_db(
      {sample_rate * state.range(0), 1024});
  hwy::AlignedNDArray<float, 3> full_masking_db(
      {energy_channels_db.shape()[0], energy_channels_db.shape()[1],
       energy_channels_db.shape()[1]});
  Masking m;
  for (auto s : state) {
    m.FullMasking(energy_channels_db, 1, full_masking_db);
  }
  state.SetItemsProcessed(energy_channels_db.size() * state.iterations());
}
BENCHMARK_RANGE(BM_FullMasking, 1, 64);

}  // namespace

}  // namespace zimtohrli
