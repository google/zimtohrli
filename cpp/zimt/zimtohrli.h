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
#include "zimt/cam.h"
#include "zimt/loudness.h"
#include "zimt/masking.h"

namespace zimtohrli {

// Contains a delta for a particular sample and channel between two
// (num_samples, num_channels)-shaped arrays.
struct SpectrogramDelta {
  // The delta between the spectrograms.
  float value = 0;
  // The value in spectrogram A.
  float spectrogram_a_value = 0;
  // The value in spectrogram B.
  float spectrogram_b_value = 0;
  // The sample where the value occurred in sound A.
  size_t sample_a_index = 0;
  // The sample where the value occurred in sound B.
  size_t sample_b_index = 0;
  // The channel where the sample occurred.
  size_t channel_index = 0;
};

// Contains results from a distance computation.
struct Distance {
  // The NSIM (https://doi.org/10.1016/j.specom.2011.09.004) between the
  // Zimtohrli spectrogram of two sounds.
  float value = 0;
  // The maximum difference in Phons between the two spectrograms.
  //
  // Computes a delta with a value equivalent to:
  // max(abs(spectrogram_a - spectrogram_b)).
  SpectrogramDelta max_relative_delta;
  // The maximum dB of the added noise between the two spectrograms.
  //
  // Computes a delta with a value equivalent to:
  // max(20 * log10(abs(10^(spectrogram_a / 20) -
  //                    10^(spectrogram_b / 20))),
  //                + epsilon),
  SpectrogramDelta max_absolute_delta;
};

// Convenient container for the output of an audio analysis.
struct Analysis {
  // The energy of the band pass filtered channels, in dB SPL.
  hwy::AlignedNDArray<float, 2> energy_channels_db;
  // The partial energy, after applying masking, of the channels, in dB SPL.
  hwy::AlignedNDArray<float, 2> partial_energy_channels_db;
  // The partial energy of the channels after converting to Phons.
  hwy::AlignedNDArray<float, 2> spectrogram;
};

// Convenient container for the output of a DTW between two Analysis instances.
struct AnalysisDTW {
  // Constructs a fake AnalysisDTW that assumes two sequences of the given
  // length with no warping.
  AnalysisDTW(size_t length);
  // Constructs an AnalysisDTW from two Analysis instances by running ChainDTW
  // with the provided window_size.
  AnalysisDTW(const Analysis& analysis_a, const Analysis& analysis_b,
              size_t window_size);
  // The DTW between the energy_channels_db field of two Analysis instances.
  std::vector<std::pair<size_t, size_t>> energy_channels_db;
  // The DTW between the partial_energy_channels_db field of two Analysis
  // instances.
  std::vector<std::pair<size_t, size_t>> partial_energy_channels_db;
  // The DTW between the spectrogram field of two Analysis instances.
  std::vector<std::pair<size_t, size_t>> spectrogram;
};

// Convenient container for the output of an audio comparison between sound A
// and a number of "sound B".
struct Comparison {
  // Analysis of each channel of sound A.
  //
  // analysis_a[channel_index] is the output of calling Analyze on channel
  // 'channel_index' of sound A.
  std::vector<Analysis> analysis_a;
  // Analysis of each channel of an arbitrary number of sounds B.
  // analysis_b[sound_b_index][channel_index] is the output of calling Analyze
  // on channel 'channel_index' of sound 'sound_b_index'.
  std::vector<std::vector<Analysis>> analysis_b;
  // The dynamic time warp relationship between sounds A and B.
  //
  // dtw[sound_b_index][channel_index].X contains the outputs of calling
  // ChainDTW(analysis_a[channel_index].X,
  // analysis_b[sound_b_index][channel_index].X, perceptual_sample_rate).
  std::vector<std::vector<AnalysisDTW>> dtw;
  // The amplitude of analysis B subtracted from analysis A, in dB.
  //
  // analysis_absolute_delta[sound_b_index][channel_index] contains the delta
  // between channel 'channel_index' of sound A and sound 'sound_b_index'.
  //
  // All comparisons between time steps in analysis A and B are made between
  // DTW-matched time steps, for all DTW-steps of analysis A.
  //
  // analysis_absolute_delta.energy_channels_db is ~equivalent to:
  // 20 * log10(10^(analysis_a.energy_channels_db / 20) -
  //            10^(analysis_b.energy_channels_db / 20))
  //
  // analysis_absolute_delta.partial_energy_channels_db is ~equivalent to:
  // 20 * log10(10^(analysis_a.partial_energy_channels_db / 20) -
  //            10^(analysis_b.partial_energy_channels_db / 20))
  //
  // analysis_absolute_delta.partial_energy_channels_phons is ~equivalent to:
  // 20 * log10(10^(analysis_a.partial_energy_channels_phons / 20) -
  //            10^(analysis_b.partial_energy_channels_phons / 20))
  std::vector<std::vector<Analysis>> analysis_absolute_delta;
  // The absolute difference between the dB of analysis A and B.
  //
  // analysis_relative_delta[sound_b_index][channel_index] contains the delta
  // between channel 'channel_index' of sound A and sound 'sound_b_index'.
  //
  // All comparisons between time steps in analysis A and B are made between
  // DTW-matched time steps, for all DTW-steps of analysis A.
  //
  // analysis_relative_delta.energy_channels_db is ~equivalent to:
  // abs(analysis_a.energy_channels_db -
  //     analysis_b.energy_channels_db)
  //
  // analysis_relative_delta.partial_energy_channels_db is ~equivalent to:
  // abs(analysis_a.partial_energy_channels_db -
  //     analysis_b.partial_energy_channels_db)
  //
  // analysis_relative_delta.partial_energy_channels_phons is ~equivalent to:
  // abs(analysis_a.partial_energy_channels_phons -
  //     analysis_b.partial_energy_channels_phons)
  std::vector<std::vector<Analysis>> analysis_relative_delta;
  // The delta between the frames in audio A and audio B, i.e. frames_a -
  // frames_b.
  //
  // frames_delta[sound_b_index] contains the delta between sound A and sound
  // 'sound_b_index' matched up using the DTW-matched time steps.
  std::vector<hwy::AlignedNDArray<float, 2>> frames_delta;
};

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
  // Returns the number of channels used in this instance.
  size_t NumChannels() const { return cam_filterbank->filter.Size(); }

  // Populates the spectrogram with the perception of frequency channels over
  // time.
  //
  // When called multiple times, the internal filterbank will keep track of the
  // previous N values to properly continue the filter where the last call
  // returned. This means that each Zimtohrli instance must only be used for one
  // sound, and each call to Spectrogram must have the next window of the sound
  // in the samples argument.
  //
  // signal is a span of audio samples between -1 and 1.
  //
  // state is the state of the internal filterbank. Reusing the state between
  // calls allows processing of audio in chunks.
  //
  // channels is a (num_samples, num-channels)-shaped array that will be
  // populated with the audio samples in the individual channels.
  //
  // energy_channels_db is a (num_downscaled_samples, num_channels)-shaped array
  // that will be populated with the dB SPL in the individual channels.
  //
  // partial_energy_channels_db is a (num_downscaled_samples,
  // num_channels)-shaped array that will be populated with energy left in each
  // channel after masking, in dB SPL.
  //
  // spectrogram is a (num_downscaled_samples, num_channels)-shaped array of
  // Phons values representing the perceptual intensity of each channel.
  //
  // num_downscaled_samples must be less than num_samples, and is typically 100
  // x duration of the sound for a perceptual intensity sample rate of 100Hz
  // which has proven reasonable for human hearing time resolution.
  //
  // partial_energy_channels_db and spectrogram can be the same arrays, in which
  // case it will be populated with the spectrogram content after the function
  // returns.
  void Spectrogram(hwy::Span<const float> signal, FilterbankState& state,
                   hwy::AlignedNDArray<float, 2>& channels,
                   hwy::AlignedNDArray<float, 2>& energy_channels_db,
                   hwy::AlignedNDArray<float, 2>& partial_energy_channels_db,
                   hwy::AlignedNDArray<float, 2>& spectrogram) const;

  // Spectrogram without chunk processing.
  void Spectrogram(hwy::Span<const float> signal,
                   hwy::AlignedNDArray<float, 2>& channels,
                   hwy::AlignedNDArray<float, 2>& energy_channels_db,
                   hwy::AlignedNDArray<float, 2>& partial_energy_channels_db,
                   hwy::AlignedNDArray<float, 2>& spectrogram) const;

  // Returns the perceptual distance between the two spectrograms.
  //
  // spectrogram_a and spectrogram_b are (num_samples, num_channels)-shaped
  // arrays of Phons values, computed using the Spectrum method.
  //
  // If verbose is false only the `norm` field in the result will be populated.
  //
  // If unwarp_window_samples has a value, will run a ChainDTW with that window
  // and only compare matching time steps of the spectrograms.
  //
  // Assumes that any padding built into the spectrogram arrays (the
  // values between spectrogram.shape() and spectrogram.memory_shape()) is
  // populated with zeros.
  struct Distance Distance(bool verbose,
                           const hwy::AlignedNDArray<float, 2>& spectrogram_a,
                           const hwy::AlignedNDArray<float, 2>& spectrogram_b,
                           std::optional<size_t> unwarp_window_samples) const;

  // Convenience method to analyze a signal.
  //
  // Allocates an Analysis instance, and executes Spectrogram on it along with
  // the provided channels working memory array.
  //
  // signal is a span of audio samples between -1 and 1.
  //
  // state is the state of the internal filterbank. Reusing the state between
  // calls allows processing of audio in chunks.
  //
  // channels is a (num_samples, num_channels)-shaped array that will be
  // populated with the audio samples in the individual channels.
  Analysis Analyze(hwy::Span<const float> signal, FilterbankState& state,
                   hwy::AlignedNDArray<float, 2>& channels) const;

  // Analyze without chunk processing.
  Analysis Analyze(hwy::Span<const float> signal,
                   hwy::AlignedNDArray<float, 2>& channels) const;

  // Convenience method to compare multi channel audios.
  //
  // Allocates a Comparison instance and populates it with analyses of the
  // signals and their delta.
  //
  // Note that the channels of the input are audio channels, e.g. left and right
  // in a stereo setup, and not the perceptual channels mentioned in other parts
  // of the Zimtohrli API.
  //
  // To ensure all audio channels are treated equally, the filterbank will be
  // reset (input and output buffers cleared) before analyzing each audio
  // channel.
  //
  // frames_a is a (num_audio_channels, num_samples)-shaped array of samples
  // between -1 and 1.
  //
  // frames_b_span is a span of (num_audio_channels, num_samples)-shaped
  // arrays of samples between -1 and 1.
  //
  // If unwarp_window_samples has a value, will run a ChainDTW with that window
  // and only compare matching time steps of the spectrograms.
  Comparison Compare(
      const hwy::AlignedNDArray<float, 2>& frames_a,
      absl::Span<const hwy::AlignedNDArray<float, 2>* const> frames_b_span,
      std::optional<size_t> unwarp_window_samples);

  // Sample rate corresponding to the human hearing sensitivity to timing
  // differences.
  float perceptual_sample_rate = 100.0;

  // The filterbank used to separate the signal in frequency channels.
  std::optional<CamFilterbank> cam_filterbank;

  // The window in perceptual_sample_rate time steps when compting the NSIM.
  size_t nsim_step_window = 16;

  // The window in channels when computing the NSIM.
  size_t nsim_channel_window = 32;

  // The reference dB SPL of a sine signal of amplitude 1.
  float full_scale_sine_db = 80;

  // The epsilon added to linear energy before converting to dB to avoid
  // log-of-zero.
  float epsilon = 1e-9;

  // The masking model.
  Masking masking;

  // Perceptual intensity model.
  Loudness loudness;
};

}  // namespace zimtohrli

#endif  // CPP_ZIMT_ZIMTOHRLI_H_
