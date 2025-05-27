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

#ifndef CPP_ZIMT_ZIMTOHRLI_H_
#define CPP_ZIMT_ZIMTOHRLI_H_

#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "hwy/aligned_allocator.h"

namespace zimtohrli {

// Expected signal sample rate.
constexpr float kSampleRate = 48000;

// Contains the energy in dB FS, and maximum absolute amplitude, of a signal.
struct EnergyAndMaxAbsAmplitude {
  float energy_db_fs;
  float max_abs_amplitude;
};

// Returns the energy and maximum absolute amplitude of a signal.
EnergyAndMaxAbsAmplitude Measure(hwy::Span<const float> signal);

// Normalizes the amplitude of the signal array to have the provided maximum
// absolute amplitude.
//
// Returns the energy in dB FS, and maximum absolute amplitude, of the result.
EnergyAndMaxAbsAmplitude NormalizeAmplitude(float max_abs_amplitude,
                                            hwy::Span<float> signal);

// Contains parameters and code to compute perceptual spectrograms of sounds.
struct Zimtohrli {
  // Populates the spectrogram with the perception of frequency channels over
  // time.
  //
  void Spectrogram(hwy::Span<const float> signal,
                   hwy::AlignedNDArray<float, 2>& spectrogram) const;

  float Distance(bool verbose,
                 const hwy::AlignedNDArray<float, 2>& spectrogram_a,
                 const hwy::AlignedNDArray<float, 2>& spectrogram_b) const;

  std::vector<std::vector<float>> Compare(
      const hwy::AlignedNDArray<float, 2>& frames_a,
      absl::Span<const hwy::AlignedNDArray<float, 2>* const> frames_b_span)
      const;

  // The window in perceptual_sample_rate time steps when compting the NSIM.
  size_t nsim_step_window = 6;
  // The window in channels when computing the NSIM.
  size_t nsim_channel_window = 5;
  // Sample rate corresponding to the human hearing sensitivity to timing
  // differences.
  float high_gamma_band = 84.0;  // The clock frequency of the brain?!
  int samples_per_perceptual_block = int(kSampleRate / high_gamma_band);
  float perceptual_sample_rate = kSampleRate / samples_per_perceptual_block;
  float unwarp_window_seconds = 2.0;
  // The reference dB SPL of a sine signal of amplitude 1.
  float full_scale_sine_db = 78.3;
  float epsilon = 1e-9;
};

}  // namespace zimtohrli

#endif  // CPP_ZIMT_ZIMTOHRLI_H_
