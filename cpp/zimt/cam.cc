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
#include <iostream>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "hwy/aligned_allocator.h"
#include "zimt/elliptic.h"
#include "zimt/filterbank.h"

namespace zimtohrli {

float Cam::HzFromCam(float cam) const {
  return (pow(10, cam / erbs_scale_1) - erbs_offset) / erbs_scale_2;
}

float Cam::CamFromHz(float hz) const {
  return erbs_scale_1 * log10(erbs_offset + erbs_scale_2 * hz);
}

CamFilterbank Cam::CreateFilterbank(float sample_rate) const {
  const float low_threshold_cam = CamFromHz(low_threshold_hz);
  const float high_threshold_cam = CamFromHz(high_threshold_hz);
  const float cam_delta =
      CamFromHz(low_threshold_hz + minimum_bandwidth_hz) - low_threshold_cam;

  std::vector<std::vector<BACoeffs>> coeffs;
  std::vector<std::pair<float, float>> thresholds_vector;

  int sections = -1;
  for (float left_cam = low_threshold_cam;
       left_cam + cam_delta < high_threshold_cam; left_cam += cam_delta) {
    float left_hz = HzFromCam(left_cam);
    float right_hz = HzFromCam(left_cam + cam_delta);
    const std::vector<BACoeffs> filter_coeffs = DigitalSOSBandPass(
        filter_order, filter_pass_band_ripple, filter_stop_band_ripple, left_hz,
        right_hz, sample_rate);
    if (sections == -1) {
      sections = filter_coeffs.size();
    } else {
      CHECK_EQ(filter_coeffs.size(), sections);
    }
    coeffs.push_back(filter_coeffs);
    thresholds_vector.push_back(std::make_pair(left_hz, right_hz));
  }
  CHECK(!coeffs.empty());

  hwy::AlignedNDArray<float, 2> thresholds({3, coeffs.size()});
  for (size_t filter_index = 0; filter_index < coeffs.size(); ++filter_index) {
    thresholds[{0}][filter_index] = thresholds_vector[filter_index].first;
    thresholds[{1}][filter_index] = (thresholds_vector[filter_index].first +
                                     thresholds_vector[filter_index].second) *
                                    0.5;
    thresholds[{2}][filter_index] = thresholds_vector[filter_index].second;
  }

  return {.filter = Filterbank(coeffs),
          .thresholds_hz = std::move(thresholds),
          .cam_delta = cam_delta,
          .sample_rate = sample_rate,
          .filter_order = filter_order,
          .filter_pass_band_ripple = filter_pass_band_ripple,
          .filter_stop_band_ripple = filter_stop_band_ripple};
}

}  // namespace zimtohrli
