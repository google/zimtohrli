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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "sndfile.h"
#include "zimt/audio.h"
#include "zimt/mos.h"
#include "zimt/zimtohrli.h"

ABSL_FLAG(std::string, path_a, "", "file A to compare");
ABSL_FLAG(std::vector<std::string>, path_b, {}, "files B to compare to file A");
ABSL_FLAG(float, perceptual_sample_rate,
          zimtohrli::Zimtohrli{}.perceptual_sample_rate,
          "the frequency corresponding to the maximum time resolution, Hz");
ABSL_FLAG(float, full_scale_sine_db, 80,
          "reference dB SPL for a sine signal of amplitude 1");
ABSL_FLAG(bool, verbose, false, "verbose output");
ABSL_FLAG(bool, output_zimtohrli_distance, false,
          "Whether to output the raw Zimtohrli distance instead of a mapped "
          "mean opinion score.");
ABSL_FLAG(bool, per_channel, false,
          "Whether to output the produced metric per channel instead of a "
          "single value for all channels.");

namespace zimtohrli {

namespace {

void PrintLoadFileInfo(
    const std::string& path, const SF_INFO& file_info,
    const std::vector<EnergyAndMaxAbsAmplitude>& measurements) {
  std::cout << "Loaded " << path << " (" << file_info.channels << "x"
            << file_info.frames << "@" << file_info.samplerate << "Hz "
            << GetFormatName(file_info.format) << ", "
            << (static_cast<float>(file_info.frames) /
                static_cast<float>(file_info.samplerate))
            << "s)\n";
  for (size_t channel_index = 0; channel_index < measurements.size();
       ++channel_index) {
    std::cout << "  Channel " << channel_index
              << " energy = " << measurements[channel_index].energy_db_fs
              << "dB FS, max abs amplitude = "
              << measurements[channel_index].max_abs_amplitude << std::endl;
  }
}

float GetMetric(float zimtohrli_score) {
  if (absl::GetFlag(FLAGS_output_zimtohrli_distance)) {
    return zimtohrli_score;
  }
  return MOSFromZimtohrli(zimtohrli_score);
}

int Main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);
  const std::string path_a = absl::GetFlag(FLAGS_path_a);
  const std::vector<std::string> path_b = absl::GetFlag(FLAGS_path_b);
  if (path_a.empty() || path_b.empty()) {
    std::cerr << "Both path_a and path_b have to be specified." << std::endl;
    return 1;
  }
  const float full_scale_sine_db = absl::GetFlag(FLAGS_full_scale_sine_db);
  if (full_scale_sine_db < 1) {
    std::cerr << "Full scale sine dB must be >= 1." << std::endl;
    return 3;
  }

  absl::StatusOr<AudioFile> file_a = AudioFile::Load(path_a);
  if (!file_a.ok()) {
    std::cerr << file_a.status().message();
    return 4;
  }
  if (file_a->Info().samplerate != kSampleRate) {
    std::cerr << path_a << " has sample rate " << file_a->Info().samplerate
              << ", only " << kSampleRate << " supported" << std::endl;
    return 5;
  }
  std::vector<EnergyAndMaxAbsAmplitude> file_a_measurements;
  float file_a_max_abs_amplitude = 0;
  for (size_t channel_index = 0; channel_index < file_a->Info().channels;
       ++channel_index) {
    EnergyAndMaxAbsAmplitude measurements = Measure((*file_a)[channel_index]);
    file_a_max_abs_amplitude =
        std::max(file_a_max_abs_amplitude, measurements.max_abs_amplitude);
    file_a_measurements.push_back(measurements);
  }
  const bool verbose = absl::GetFlag(FLAGS_verbose);
  if (verbose) {
    PrintLoadFileInfo(path_a, file_a->Info(), file_a_measurements);
  }
  size_t min_length = file_a->Info().frames;

  std::vector<AudioFile> file_b_vector;
  file_b_vector.reserve(path_b.size());
  for (const std::string& path : path_b) {
    absl::StatusOr<AudioFile> file_b = AudioFile::Load(path);
    if (!file_b.ok()) {
      std::cerr << file_b.status().message();
      return 4;
    }
    if (file_b->Info().samplerate != kSampleRate) {
      std::cerr << path << " has sample rate " << file_b->Info().samplerate
                << ", only " << kSampleRate << " supported";
      return 5;
    }
    std::vector<EnergyAndMaxAbsAmplitude> measurements;
    for (size_t channel_index = 0; channel_index < file_b->Info().channels;
         ++channel_index) {
      measurements.push_back(Measure((*file_b)[channel_index]));
    }
    if (verbose) {
      PrintLoadFileInfo(file_b->Path(), file_b->Info(), measurements);
    }
    CHECK_EQ(file_a->Info().channels, file_b->Info().channels);
    CHECK_EQ(file_a->Info().samplerate, file_b->Info().samplerate);
    min_length =
        std::min(min_length, static_cast<size_t>(file_b->Info().frames));
    for (size_t channel_index = 0; channel_index < file_b->Info().channels;
         ++channel_index) {
      const EnergyAndMaxAbsAmplitude new_energy_and_max_abs_amplitude =
          NormalizeAmplitude(file_a_max_abs_amplitude,
                             (*file_b)[channel_index]);
      if (verbose) {
        std::cerr << "  Normalized channel " << channel_index << " energy = "
                  << new_energy_and_max_abs_amplitude.energy_db_fs
                  << "dB FS, max abs amplitude = "
                  << new_energy_and_max_abs_amplitude.max_abs_amplitude
                  << std::endl;
      }
      if (std::abs(new_energy_and_max_abs_amplitude.energy_db_fs -
                   file_a_measurements[channel_index].energy_db_fs) > 2.0f) {
        std::cerr << "WARNING: Energies differ more than 2dB FS after "
                     "normalizing max amplitude!"
                  << std::endl;
      }
    }
    file_b_vector.push_back(*std::move(file_b));
  }

  Zimtohrli z = {
      .perceptual_sample_rate = absl::GetFlag(FLAGS_perceptual_sample_rate),
      .full_scale_sine_db = absl::GetFlag(FLAGS_full_scale_sine_db),
  };

  const bool per_channel = absl::GetFlag(FLAGS_per_channel);
  const size_t num_downscaled_samples_a =
      static_cast<size_t>(std::ceil(static_cast<float>(file_a->Info().frames) *
                                    z.perceptual_sample_rate / kSampleRate));
  std::vector<Spectrogram> file_a_spectrograms;
  for (size_t channel_index = 0; channel_index < file_a->Info().channels;
       ++channel_index) {
    Spectrogram spectrogram(num_downscaled_samples_a, kNumRotators);
    z.Analyze((*file_a)[channel_index], spectrogram);
    file_a_spectrograms.push_back(std::move(spectrogram));
  }
  for (int file_b_index = 0; file_b_index < file_b_vector.size();
       ++file_b_index) {
    const AudioFile& file_b = file_b_vector[file_b_index];
    const size_t num_downscaled_samples_b =
        static_cast<size_t>(std::ceil(static_cast<float>(file_b.Info().frames) *
                                      z.perceptual_sample_rate / kSampleRate));
    Spectrogram spectrogram_b(num_downscaled_samples_b, kNumRotators);
    float sum_of_squares = 0;
    for (size_t channel_index = 0; channel_index < file_a->Info().channels;
         ++channel_index) {
      z.Analyze(file_b[channel_index], spectrogram_b);
      const float distance =
          z.Distance(file_a_spectrograms[channel_index], spectrogram_b);
      if (per_channel) {
        std::cout << GetMetric(distance) << std::endl;
      } else {
        sum_of_squares += distance * distance;
      }
    }
    if (!per_channel) {
      for (int file_b_index = 0; file_b_index < file_b_vector.size();
           ++file_b_index) {
        std::cout << GetMetric(std::sqrt(sum_of_squares /
                                         float(file_a->Info().channels)))
                  << std::endl;
      }
    }
  }
  return 0;
}

}  // namespace

}  // namespace zimtohrli

int main(int argc, char* argv[]) { return zimtohrli::Main(argc, argv); }
