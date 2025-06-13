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
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
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
  std::vector<std::vector<float>> channels_a;
  channels_a.reserve(file_a->Info().channels);
  for (size_t channel_idx = 0; channel_idx < file_a->Info().channels;
       ++channel_idx) {
    channels_a.push_back(file_a->AtRate(channel_idx, kSampleRate));
  }
  std::vector<EnergyAndMaxAbsAmplitude> file_a_measurements;
  float file_a_max_abs_amplitude = 0;
  for (size_t channel_index = 0; channel_index < file_a->Info().channels;
       ++channel_index) {
    EnergyAndMaxAbsAmplitude measurements = Measure(channels_a[channel_index]);
    file_a_max_abs_amplitude =
        std::max(file_a_max_abs_amplitude, measurements.max_abs_amplitude);
    file_a_measurements.push_back(measurements);
  }
  const bool verbose = absl::GetFlag(FLAGS_verbose);
  if (verbose) {
    PrintLoadFileInfo(path_a, file_a->Info(), file_a_measurements);
  }

  std::vector<AudioFile> file_b_vector;
  file_b_vector.reserve(path_b.size());
  std::vector<std::vector<std::vector<float>>> channels_b_vector;
  channels_b_vector.reserve(path_b.size());
  for (const std::string& path : path_b) {
    absl::StatusOr<AudioFile> file_b = AudioFile::Load(path);
    if (!file_b.ok()) {
      std::cerr << file_b.status().message();
      return 4;
    }
    std::vector<std::vector<float>> channels_b;
    channels_b.reserve(file_b->Info().channels);
    for (size_t channel_idx = 0; channel_idx < file_b->Info().channels;
         ++channel_idx) {
      channels_b.push_back(file_b->AtRate(channel_idx, kSampleRate));
    }
    std::vector<EnergyAndMaxAbsAmplitude> measurements;
    for (size_t channel_index = 0; channel_index < file_b->Info().channels;
         ++channel_index) {
      measurements.push_back(Measure(channels_b[channel_index]));
    }
    if (verbose) {
      PrintLoadFileInfo(file_b->Path(), file_b->Info(), measurements);
    }
    CHECK_EQ(file_a->Info().channels, file_b->Info().channels);
    CHECK_EQ(file_a->Info().samplerate, file_b->Info().samplerate);
    for (size_t channel_index = 0; channel_index < file_b->Info().channels;
         ++channel_index) {
      const EnergyAndMaxAbsAmplitude new_energy_and_max_abs_amplitude =
          NormalizeAmplitude(file_a_max_abs_amplitude,
                             channels_b[channel_index]);
      if (verbose) {
        std::cerr << "  Normalized channel " << channel_index << " energy = "
                  << new_energy_and_max_abs_amplitude.energy_db_fs
                  << "dB FS, max abs amplitude = "
                  << new_energy_and_max_abs_amplitude.max_abs_amplitude
                  << std::endl;
      }
    }
    file_b_vector.push_back(*std::move(file_b));
    channels_b_vector.push_back(channels_b);
  }

  Zimtohrli z = {
      .perceptual_sample_rate = absl::GetFlag(FLAGS_perceptual_sample_rate),
      .full_scale_sine_db = absl::GetFlag(FLAGS_full_scale_sine_db),
  };

  const bool per_channel = absl::GetFlag(FLAGS_per_channel);
  std::vector<Spectrogram> file_a_spectrograms;
  for (size_t channel_index = 0; channel_index < file_a->Info().channels;
       ++channel_index) {
    Spectrogram spectrogram = z.Analyze(channels_a[channel_index]);
    file_a_spectrograms.push_back(std::move(spectrogram));
  }
  for (int file_b_index = 0; file_b_index < file_b_vector.size();
       ++file_b_index) {
    const AudioFile& file_b = file_b_vector[file_b_index];
    const std::vector<std::vector<float>>& channels_b =
        channels_b_vector[file_b_index];
    std::optional<Spectrogram> spectrogram_b;
    float sum_of_squares = 0;
    for (size_t channel_index = 0; channel_index < file_a->Info().channels;
         ++channel_index) {
      if (spectrogram_b.has_value()) {
        z.Analyze(channels_b[channel_index], *spectrogram_b);
      } else {
        spectrogram_b = z.Analyze(channels_b[channel_index]);
      }
      const float distance =
          z.Distance(file_a_spectrograms[channel_index], *spectrogram_b);
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
