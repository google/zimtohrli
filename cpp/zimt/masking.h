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

#ifndef CPP_ZIMT_MASKING_H_
#define CPP_ZIMT_MASKING_H_

#include "hwy/aligned_allocator.h"

namespace zimtohrli {

// Populates the energy_channels with the (possibly downsampled) energy of the
// sample_channels.
//
// Input and output contains linear energy values.
//
// sample_channels is a (num_samples, num_channels)-shaped array with samples.
//
// energy_channels is a (downscaled_num_samples, num_channels)-shaped array with
// energy (mean square of samples).
//
// num_downscaled_samples must be less than num_samples, and is typically 100
// x duration of the sound for a perceptual intensity sample rate of 100Hz
// which has proven reasonable for human hearing time resolution.
void ComputeEnergy(const hwy::AlignedNDArray<float, 2>& sample_channels,
                   hwy::AlignedNDArray<float, 2>& energy_channels);

// Populates energy_channels_db with the dB energy value of
// energy_channels_linear.
//
// energy_channels_linear and energy_channels_db can be the same array.
//
// full_scale_sine_db is the reference dB SPL of a sine wave of amplitude 1.
//
// Equivalent to setting all values in energy_channels_db to
// `full_scale_sine_db + 10 * log10(energy_channels_linear + epsilon)`.
//
// energy_channels_linear and energy_channels_db can be the same array.
void ToDb(const hwy::AlignedNDArray<float, 2>& energy_channels_linear,
          float full_scale_sine_db, float epsilon,
          hwy::AlignedNDArray<float, 2>& energy_channels_db);

// Populates energy_channels_linear with the linear energy value of
// energy_channels_db.
//
// energy_channels_linear and energy_channels_db can be the same array.
//
// full_scale_sine_db is the reference dB SPL of a sine wave of amplitude 1.
//
// Equivalent to setting all values in energy_channels_linear to
// `10^((energy_channels_db - full_scale_sine_db) / 10)`.
//
// energy_channels_linear and energy_channels_db can be the same array.
void ToLinear(const hwy::AlignedNDArray<float, 2>& energy_channels_db,
              float full_scale_sine_db,
              hwy::AlignedNDArray<float, 2>& energy_channels_linear);

// Contains parameters and functions to compute auditory masking.
struct Masking {
  // Populates full_masking_db with the full masking levels of the channels in
  // energy_channels_db.
  //
  // energy_channels_db is a (num_samples, num_channels)-shaped array of energy
  // expressed in dB.
  //
  // cam_delta is the cam delta between each channel and the next.
  //
  // full_masking_db is a (num_samples, num_masked_channels,
  // num_masker_channels)-shaped array of full masking levels expressed in dB.
  // num_masker_channels and num_masked_channels are both identical to
  // num_channels.
  void FullMasking(const hwy::AlignedNDArray<float, 2>& energy_channels_db,
                   float cam_delta,
                   hwy::AlignedNDArray<float, 3>& full_masking_db) const;

  // Populates non_masked_db with the energy after any fully masked channels are
  // decimated.
  //
  // energy_channels_db is a (num_samples, num_channels)-shaped array of dB
  // energy values.
  //
  // cam_delta is the cam delta between each channel and the next.
  //
  // non_masked_db is a (num_samples, num_channels)-shaped array of dB energy
  // values.
  //
  // Assumes that any padding built into the energy_channels_db array (the
  // values between energy_channels_db.shape() and
  // energy_channels_db.memory_shape()) is populated with zeros.
  void CutFullyMasked(const hwy::AlignedNDArray<float, 2>& energy_channels_db,
                      float cam_delta,
                      hwy::AlignedNDArray<float, 2>& non_masked_db) const;

  // The negative distance in Cam at which a 20dB masker will no longer mask any
  // probe.
  float lower_zero_at_20 = -4.1;
  // The negative distance in Cam at which an 80dB masker will no longer mask
  // any probe.
  float lower_zero_at_80 = -6;
  // The positive distance in Cam at which a 20dB masker will no longer mask any
  // probe.
  float upper_zero_at_20 = 3.1;
  // The positive distance in Cam at which an 80dB masker will no longer mask
  // any probe.
  float upper_zero_at_80 = 9.6;

  // The dB that a masker masks in the same band.
  float max_mask = 17.7;
};

}  // namespace zimtohrli

#endif  // CPP_ZIMT_MASKING_H_
