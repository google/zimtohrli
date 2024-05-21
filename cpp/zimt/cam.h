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

#ifndef CPP_ZIMT_CAM_H_
#define CPP_ZIMT_CAM_H_

#include "hwy/aligned_allocator.h"
#include "zimt/filterbank.h"

namespace zimtohrli {

// Contains a filterbank along with the threshold frequencies used to create it.
struct CamFilterbank {
  // A filterbank.
  Filterbank filter;

  // The low threshold/peak response/high thresholds of each filter in the bank,
  // shaped like (3, num_channels).
  hwy::AlignedNDArray<float, 2> thresholds_hz;

  // The Cam delta between channels.
  float cam_delta;

  // The sample rate this filterbank was designed for.
  float sample_rate;

  // The parameters used when creating this filterbank.
  int filter_order;
  float filter_pass_band_ripple;
  float filter_stop_band_ripple;
};

// Converts between Hz and Cam (see
// https://en.wikipedia.org/wiki/Equivalent_rectangular_bandwidth), and creates
// filterbanks to enable filtering sounds according to the frequency resolution
// of human hearing.
struct Cam {
  // Returns the Hz frequency for the given Cam frequency.
  float HzFromCam(float cam) const;

  // Returns the Cam frequency for the given Hz frequency.
  float CamFromHz(float hz) const;

  // Returns a filterbank with filters between low_threshold_hz and
  // high_threshold_hz, with the first filter having minimum_bandwidth_hz width,
  // and all filters having the same bandwidth in Cam.
  CamFilterbank CreateFilterbank(float sample_rate) const;

  // Scale constant for Hz/Cam conversion.
  float erbs_scale_1 = 21.4;
  // Scale constant for Hz/Cam conversion.
  float erbs_scale_2 = 0.00437;
  // Offset constant for Hz/Cam conversion.
  float erbs_offset = 1.0;

  // Low frequency threshold for hearing.
  float low_threshold_hz = 20;
  // High frequency threshold for hearing.
  float high_threshold_hz = 20000;
  // Frequency resolution at low threshold. Default is 5 since it has
  // provided the best correlation scores with tested datasets.
  float minimum_bandwidth_hz = 5;
  // Order (sharpness and slowness) of channel filters.
  int filter_order = 1;
  // Attenuation in dB in each filter where the filterbank filters meet.
  float filter_pass_band_ripple = 3;
  // Minimum attenuation outside the filters, interpreted as attempted
  // sharpness. Not used for order 1.
  float filter_stop_band_ripple = 80;
};

}  // namespace zimtohrli

#endif  // CPP_ZIMT_CAM_H_
