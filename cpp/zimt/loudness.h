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

#ifndef CPP_ZIMT_LOUDNESS_H_
#define CPP_ZIMT_LOUDNESS_H_

#include <array>

#include "hwy/aligned_allocator.h"

namespace zimtohrli {

// Contains parameters and functions to compute perceptual loudness according to
// ISO 226 (https://www.iso.org/standard/83117.html).
//
// Computes the a_f, L_U, and T_f parameters in the ISO 226 conversion formula
// using a parameterization described in `loudness_parameter_computation.ipynb`.
struct Loudness {
  // Populates channels_phons with the Phons intensity corresponding to the dB
  // SPL intensity of channels_db_spl.
  //
  // channels_db_spl is a (num_samples, num_channels)-shaped array of intensity
  // in dB SPL.
  //
  // thresholds_hz is a (3, num_channels)-shaped array of low threshold/peak
  // response/high threshold in Hz for each channel spectrum.
  //
  // channels_phons is a (num_samples, num_channels)-shaped array of intensity
  // in Phons.
  //
  // channels_db_spl and channels_phons can be the same array.
  void PhonsFromSPL(const hwy::AlignedNDArray<float, 2>& channels_db_spl,
                    const hwy::AlignedNDArray<float, 2>& thresholds_hz,
                    hwy::AlignedNDArray<float, 2>& channels_phons) const;

  // Parameters to create the ISO 226 a_f parameter for an arbitrary frequency.
  std::array<float, 10> a_f_params = {
      9.64075296e-01, 8.76031085e-02, 1.04933605e+00, 7.21105886e+00,
      1.02014870e+03, 1.58967888e-02, 5.06793363e+00, 1.33880326e+03,
      1.39332233e-01, -3.86752752e+00};
  // Parameters to create the ISO 226 L_U parameter for an arbitrary frequency.
  std::array<float, 16> l_u_params = {
      1.04895312e+04, 1.16834373e+01,  4.63216422e+02,  1.07277873e+01,
      9.33873976e-01, 5.79566363e-01,  1.06503907e+00,  1.93853475e+04,
      4.63762437e-02, -4.36163544e+00, 1.50737510e+00,  7.94185866e-01,
      8.70352919e-01, 2.12991220e+03,  -2.62054739e-03, 2.63009217e-01,
  };
  // Parameters to create the ISO 226 T_f parameter for an arbitrary frequency.
  std::array<float, 13> t_f_params = {
      -1.92510181e+02, -2.20827757e+01, 9.19748235e+01, 1.26594322e+01,
      6.97360326e+00,  2.99022584e-02,  9.50394539e-01, -3.71694403e+01,
      4.20769098e-02,  1.86764149e+00,  1.97954462e-07, -8.75703210e-03,
      -3.64426652e+01};
};

}  // namespace zimtohrli

#endif  // CPP_ZIMT_LOUDNESS_H_
