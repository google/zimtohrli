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

#include "zimt/cam.h"

#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "hwy/aligned_allocator.h"
#include "zimt/filterbank.h"

namespace zimtohrli {

namespace {

TEST(Cam, GoldenDataTest) {
  const Cam cam;

  const std::vector<std::pair<float, float>> cam_and_hz({
      {1, 25.995276471698844},
      {10, 442.29956714831576},
      {20, 1739.4974583218288},
      {30, 5543.983136917903},
  });
  for (const std::pair<float, float>& p : cam_and_hz) {
    EXPECT_NEAR(cam.CamFromHz(p.second), p.first, 1e-5);
  }

  const std::vector<std::pair<float, float>> hz_and_cam({
      {1, 0.04052563},
      {10, 0.3975194},
      {100, 3.3695753},
      {1000, 15.621448},
      {10000, 35.316578},
  });
  for (const std::pair<float, float>& p : hz_and_cam) {
    EXPECT_NEAR(cam.HzFromCam(p.second), p.first, 1e-2);
  }
}

TEST(Cam, CreateFilterbankTest) {
  CamFilterbank filterbank = Cam().CreateFilterbank(48000);
  hwy::AlignedNDArray<float, 1> signal({4800});
  signal[{}][0] = 1;
  hwy::AlignedNDArray<float, 2> filtered_signal(
      {4800, filterbank.filter.Size()});
  filterbank.filter.Filter(signal[{}], filtered_signal);
  for (size_t filter_index = 0; filter_index < filtered_signal.shape()[1];
       ++filter_index) {
    float energy = 0;
    for (size_t sample_index = 0; sample_index < filtered_signal.shape()[0];
         ++sample_index) {
      energy += pow(filtered_signal[{sample_index}][filter_index], 2);
    }
    EXPECT_GT(energy, 0);
  }
}

TEST(Cam, ProcessFilterbankTest) {
  const float sample_rate = 48000;
  Cam cam;
  cam.minimum_bandwidth_hz = 10;
  CamFilterbank filterbank = cam.CreateFilterbank(sample_rate);
  size_t signal_length = static_cast<size_t>(sample_rate * 0.2);
  hwy::AlignedNDArray<float, 1> signal({signal_length});
  hwy::AlignedNDArray<float, 2> filtered_signal(
      {signal_length, filterbank.filter.Size()});
  std::pair<float, float> previous_threshold_hz = {cam.low_threshold_hz - 1,
                                                   cam.low_threshold_hz};
  float previous_bandwidth_cam = -1;
  for (size_t tested_filter_index = 0;
       tested_filter_index < filterbank.filter.Size(); ++tested_filter_index) {
    const std::pair<float, float> threshold_hz = {
        filterbank.thresholds_hz[{0}][tested_filter_index],
        filterbank.thresholds_hz[{2}][tested_filter_index]};
    if (previous_bandwidth_cam == -1) {
      previous_bandwidth_cam = cam.CamFromHz(threshold_hz.second) -
                               cam.CamFromHz(threshold_hz.first);
    } else {
      float new_bandwidth_cam = cam.CamFromHz(threshold_hz.second) -
                                cam.CamFromHz(threshold_hz.first);
      EXPECT_NEAR(previous_bandwidth_cam, new_bandwidth_cam, 1e-3);
      previous_bandwidth_cam = new_bandwidth_cam;
    }
    EXPECT_NEAR(threshold_hz.first, previous_threshold_hz.second, 1e-3);
    EXPECT_GT(threshold_hz.second, threshold_hz.first);
    EXPECT_NEAR(filterbank.thresholds_hz[{1}][tested_filter_index],
                (threshold_hz.first + threshold_hz.second) * 0.5, 1e-3);
    float center_frequency = (threshold_hz.first + threshold_hz.second) * 0.5;
    for (size_t sample_index = 0; sample_index < signal.shape()[0];
         ++sample_index) {
      signal[{}][sample_index] =
          std::sin(sample_index * 2 * M_PI * center_frequency / sample_rate);
    }
    filterbank.filter.Filter(signal[{}], filtered_signal);
    size_t max_energy_filter_index = 0;
    float max_energy = -1;
    for (size_t measured_filter_index = 0;
         measured_filter_index < filtered_signal.shape()[1];
         ++measured_filter_index) {
      float energy = 0;
      for (size_t sample_index = 0; sample_index < filtered_signal.shape()[0];
           ++sample_index) {
        energy +=
            pow(filtered_signal[{sample_index}][measured_filter_index], 2);
      }
      if (energy > max_energy) {
        max_energy = energy;
        max_energy_filter_index = measured_filter_index;
      }
    }
    EXPECT_EQ(max_energy_filter_index, tested_filter_index)
        << "center_frequency=" << center_frequency
        << ", max_energy_filter_index=" << max_energy_filter_index << " ("
        << filterbank.thresholds_hz[{0}][max_energy_filter_index] << "Hz - "
        << filterbank.thresholds_hz[{2}][max_energy_filter_index] << "Hz) with "
        << max_energy << ", tested_filter_index=" << tested_filter_index << " ("
        << threshold_hz.first << "Hz - " << threshold_hz.second << "Hz)";
    previous_threshold_hz = threshold_hz;
  }
}

}  // namespace

}  // namespace zimtohrli
