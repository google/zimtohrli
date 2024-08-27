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

//
// To run:
//
// $ blaze run third_party/zimtohrli/cpp:compare -- --file_a=reference.wav \
//     --file_b=distortion1.wav,distortion2.wav
//
// This will output a single number per channel, which is the Zimtohrli distance
// between the channels.
//
// For more verbose output, explaining how the different layers of Zimtohrli
// measures the differences, and where they occur:
//
// $ compare --verbose --path_a=reference.wav \
// --path_b=distortion1.wav,distortion2.wav
//
// To see a user interface displaying the various spectrograms Zimtohrli
// generates to compute the distance, their relative and absolute differences,
// the dynamic time warp between them, and allowing the user to select
// time+frequency ranges to listen to:
//
// $ compare --ux --path_a=reference.wav \
// --path_b=distortion1.wav,distortion2.wav
//

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
#include "hwy/aligned_allocator.h"
#include "sndfile.h"
#include "zimt/audio.h"
#include "zimt/cam.h"
#include "zimt/mos.h"
#include "zimt/ux.h"
#include "zimt/zimtohrli.h"

ABSL_FLAG(std::string, path_a, "", "file A to compare");
ABSL_FLAG(std::vector<std::string>, path_b, {}, "files B to compare to file A");
ABSL_FLAG(float, frequency_resolution, zimtohrli::Cam{}.minimum_bandwidth_hz,
          "maximum frequency resolution, Hz");
ABSL_FLAG(float, perceptual_sample_rate,
          zimtohrli::Zimtohrli{}.perceptual_sample_rate,
          "the frequency corresponding to the maximum time resolution, Hz");
ABSL_FLAG(float, full_scale_sine_db, 80,
          "reference dB SPL for a sine signal of amplitude 1");
ABSL_FLAG(bool, verbose, false, "verbose output");
ABSL_FLAG(bool, ux, false, "create graphical UX");
ABSL_FLAG(bool, truncate, false,
          "if the files are of different lengths, truncate the longer one");
ABSL_FLAG(float, unwarp_window, 2.0f,
          "unwarp window length in seconds, must be greater than 0 if truncate "
          "is false and the files are of different lengths");
ABSL_FLAG(bool, normalize_amplitude, true,
          "whether to normalize the amplitude of all B sounds to the same max "
          "amplitude as the A sound");
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

struct DistanceData {
  DistanceData(const Zimtohrli& z, const DistanceData* previous_step,
               const hwy::AlignedNDArray<float, 2>& thresholds_hz,
               const hwy::AlignedNDArray<float, 2>& a,
               const hwy::AlignedNDArray<float, 2>& b, const std::string& unit,
               float perceptual_sample_rate)
      : previous_step(previous_step),
        a(&a),
        b(&b),
        distance(z.Distance(true, a, b)),
        unit(unit),
        thresholds_hz(&thresholds_hz),
        perceptual_sample_rate(perceptual_sample_rate) {}
  const DistanceData* previous_step;
  const hwy::AlignedNDArray<float, 2>* a;
  const hwy::AlignedNDArray<float, 2>* b;
  const Distance distance;
  const std::string unit;
  const hwy::AlignedNDArray<float, 2>* thresholds_hz;
  const float perceptual_sample_rate;
};

std::string Fix(float f, const std::string& unit, size_t width) {
  const std::string unpadded = absl::StrCat(f, " ", unit);
  if (width <= unpadded.size()) {
    return unpadded;
  }
  return absl::StrCat(unpadded, std::string(width - unpadded.size(), ' '));
}

void PrintDistanceInfo(std::ostream& outs, const DistanceData& data,
                       const absl::string_view label,
                       const SpectrogramDelta& delta) {
  outs << "    " << label << " of " << delta.value << " " << data.unit
       << " at sample A:" << delta.sample_a_index
       << "/B:" << delta.sample_b_index << " (A:"
       << (static_cast<float>(delta.sample_a_index) /
           static_cast<float>(data.perceptual_sample_rate))
       << " s/B:"
       << (static_cast<float>(delta.sample_b_index) /
           static_cast<float>(data.perceptual_sample_rate))
       << " s) and channel " << delta.channel_index << " ("
       << (*data.thresholds_hz)[{0}][delta.channel_index] << "-"
       << (*data.thresholds_hz)[{2}][delta.channel_index] << " Hz)" << std::endl
       << "                     A                B" << std::endl
       << "        This stage   "
       << Fix(delta.spectrogram_a_value, data.unit, 17)
       << Fix(delta.spectrogram_b_value, data.unit, 0) << std::endl;
  if (data.previous_step) {
    outs << "        Prev stage   "
         << Fix((*data.previous_step
                      ->a)[{delta.sample_a_index}][delta.channel_index],
                data.previous_step->unit, 17)
         << Fix((*data.previous_step
                      ->b)[{delta.sample_b_index}][delta.channel_index],
                data.previous_step->unit, 0)
         << std::endl;
  }
}

std::ostream& operator<<(std::ostream& outs, const DistanceData& data) {
  outs << data.distance.value << " " << data.unit << std::endl;
  PrintDistanceInfo(outs, data, "Peak absolute noise",
                    data.distance.max_absolute_delta);
  PrintDistanceInfo(outs, data, "Peak relative noise",
                    data.distance.max_relative_delta);
  return outs;
}

float GetMetric(const zimtohrli::Zimtohrli& z, float zimtohrli_score) {
  if (absl::GetFlag(FLAGS_output_zimtohrli_distance)) {
    return zimtohrli_score;
  }
  return z.mos_mapper.Map(zimtohrli_score);
}

int Main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);
  const std::string path_a = absl::GetFlag(FLAGS_path_a);
  const std::vector<std::string> path_b = absl::GetFlag(FLAGS_path_b);
  if (path_a.empty() || path_b.empty()) {
    std::cerr << "Both path_a and path_b have to be specified." << std::endl;
    return 1;
  }
  const float frequency_resolution = absl::GetFlag(FLAGS_frequency_resolution);
  if (frequency_resolution < 1) {
    std::cerr << "Maximum frequency resolution must be >= 1." << std::endl;
    return 2;
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
  std::vector<EnergyAndMaxAbsAmplitude> file_a_measurements;
  for (size_t channel_index = 0; channel_index < file_a->Info().channels;
       ++channel_index) {
    EnergyAndMaxAbsAmplitude measurements =
        Measure(file_a->Frames()[{channel_index}]);
    file_a_measurements.push_back(measurements);
  }
  const bool verbose = absl::GetFlag(FLAGS_verbose);
  if (verbose) {
    PrintLoadFileInfo(path_a, file_a->Info(), file_a_measurements);
  }
  size_t min_length = file_a->Frames().shape()[1];

  std::vector<AudioFile> file_b_vector;
  file_b_vector.reserve(path_b.size());
  for (const std::string& path : path_b) {
    absl::StatusOr<AudioFile> file_b = AudioFile::Load(path);
    if (!file_b.ok()) {
      std::cerr << file_b.status().message();
      return 4;
    }
    std::vector<EnergyAndMaxAbsAmplitude> measurements;
    for (size_t channel_index = 0; channel_index < file_b->Info().channels;
         ++channel_index) {
      measurements.push_back(Measure(file_b->Frames()[{channel_index}]));
    }
    if (verbose) {
      PrintLoadFileInfo(file_b->Path(), file_b->Info(), measurements);
    }
    CHECK_EQ(file_a->Info().channels, file_b->Info().channels);
    CHECK_EQ(file_a->Info().samplerate, file_b->Info().samplerate);
    min_length = std::min(min_length, file_b->Frames().shape()[1]);
    if (absl::GetFlag(FLAGS_normalize_amplitude)) {
      for (size_t channel_index = 0; channel_index < file_b->Info().channels;
           ++channel_index) {
        const EnergyAndMaxAbsAmplitude new_energy_and_max_abs_amplitude =
            NormalizeAmplitude(file_a_measurements[channel_index].max_abs_amplitude,
                               file_b->Frames()[{channel_index}]);
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
    }
    file_b_vector.push_back(*std::move(file_b));
  }

  Cam cam{.minimum_bandwidth_hz = frequency_resolution};
  cam.high_threshold_hz =
      std::min(cam.high_threshold_hz, file_a->Info().samplerate / 2.0f);
  Zimtohrli z = {
      .perceptual_sample_rate = absl::GetFlag(FLAGS_perceptual_sample_rate),
      .cam_filterbank =
          cam.CreateFilterbank(static_cast<float>(file_a->Info().samplerate)),
      .unwarp_window_seconds = absl::GetFlag(FLAGS_unwarp_window),
      .full_scale_sine_db = absl::GetFlag(FLAGS_full_scale_sine_db),
  };

  // Run a more optimized code path if the user doesn't want either UX or
  // verbose output.
  const bool truncate = absl::GetFlag(FLAGS_truncate);
  if (truncate && file_a->Frames().shape()[1] != min_length) {
    file_a->Frames().truncate({file_a->Frames().shape()[0], min_length});
    file_a->Info().frames = min_length;
  }
  for (AudioFile& file_b : file_b_vector) {
    if (truncate) {
      file_b.Frames().truncate({file_b.Frames().shape()[0], min_length});
      file_b.Info().frames = min_length;
    } else if (absl::GetFlag(FLAGS_unwarp_window) == 0) {
      CHECK_EQ(file_a->Info().frames, file_b.Info().frames)
          << "use --truncate=true or --unwarp_window=[something > 0]";
    }
  }

  const bool ux = absl::GetFlag(FLAGS_ux);
  const bool per_channel = absl::GetFlag(FLAGS_per_channel);
  if (!ux && !verbose) {
    const size_t num_downscaled_samples_a = static_cast<size_t>(
        std::ceil(static_cast<float>(file_a->Frames().shape()[1]) *
                  z.perceptual_sample_rate / z.cam_filterbank->sample_rate));
    hwy::AlignedNDArray<float, 2> channels_a(
        {file_a->Frames().shape()[1], z.cam_filterbank->filter.Size()});
    hwy::AlignedNDArray<float, 2> energy_channels_db_a(
        {num_downscaled_samples_a, z.cam_filterbank->filter.Size()});
    hwy::AlignedNDArray<float, 2> partial_energy_channels_db_a(
        {num_downscaled_samples_a, z.cam_filterbank->filter.Size()});
    std::vector<hwy::AlignedNDArray<float, 2>> file_a_spectrograms;
    for (size_t channel_index = 0; channel_index < file_a->Info().channels;
         ++channel_index) {
      hwy::AlignedNDArray<float, 2> spectrogram(
          {num_downscaled_samples_a, z.cam_filterbank->filter.Size()});
      z.Spectrogram(file_a->Frames()[{channel_index}], channels_a,
                    energy_channels_db_a, partial_energy_channels_db_a,
                    spectrogram);
      file_a_spectrograms.push_back(std::move(spectrogram));
    }
    for (int file_b_index = 0; file_b_index < file_b_vector.size();
         ++file_b_index) {
      const AudioFile& file_b = file_b_vector[file_b_index];
      const size_t num_downscaled_samples_b = static_cast<size_t>(
          std::ceil(static_cast<float>(file_b.Frames().shape()[1]) *
                    z.perceptual_sample_rate / z.cam_filterbank->sample_rate));
      hwy::AlignedNDArray<float, 2> channels_b(
          {file_b.Frames().shape()[1], z.cam_filterbank->filter.Size()});
      hwy::AlignedNDArray<float, 2> energy_channels_db_b(
          {num_downscaled_samples_b, z.cam_filterbank->filter.Size()});
      hwy::AlignedNDArray<float, 2> partial_energy_channels_db_b(
          {num_downscaled_samples_b, z.cam_filterbank->filter.Size()});
      hwy::AlignedNDArray<float, 2> spectrogram_b(
          {num_downscaled_samples_b, z.cam_filterbank->filter.Size()});
      float sum_of_squares = 0;
      for (size_t channel_index = 0; channel_index < file_a->Info().channels;
           ++channel_index) {
        z.Spectrogram(file_b.Frames()[{channel_index}], channels_b,
                      energy_channels_db_b, partial_energy_channels_db_b,
                      spectrogram_b);
        const float distance =
            z.Distance(false, file_a_spectrograms[channel_index], spectrogram_b)
                .value;
        if (per_channel) {
          std::cout << GetMetric(z, distance) << std::endl;
        } else {
          sum_of_squares += distance * distance;
        }
      }
      if (!per_channel) {
        for (int file_b_index = 0; file_b_index < file_b_vector.size();
             ++file_b_index) {
          std::cout << GetMetric(z, std::sqrt(sum_of_squares /
                                              float(file_a->Info().channels)))
                    << std::endl;
        }
      }
    }
    return 0;
  }

  std::vector<const hwy::AlignedNDArray<float, 2>*> frames_b;
  frames_b.reserve(file_b_vector.size());
  for (const AudioFile& file_b : file_b_vector) {
    frames_b.push_back(&file_b.Frames());
  }
  Comparison comparison = z.Compare(file_a->Frames(), frames_b);

  if (ux) {
    UX ux;
    ux.Paint({.file_a = std::move(file_a.value()),
              .file_b = std::move(file_b_vector),
              .comparison = std::move(comparison),
              .thresholds_hz = std::move(z.cam_filterbank->thresholds_hz),
              .full_scale_sine_db = full_scale_sine_db,
              .perceptual_sample_rate = z.perceptual_sample_rate,
              .unwarp_window = absl::GetFlag(FLAGS_unwarp_window)});
    return 0;
  }

  if (verbose) {
    for (size_t b_index = 0; b_index < file_b_vector.size(); ++b_index) {
      const AudioFile& file_b = file_b_vector[b_index];
      std::cout << "A (" << file_a->Path() << ") vs B (" << file_b.Path() << ")"
                << std::endl;
      float sum_of_squares = 0;
      for (size_t channel_index = 0;
           channel_index < comparison.analysis_a.size(); ++channel_index) {
        std::cout << "  Channel " << channel_index << std::endl;
        const DistanceData raw_channel_distance = DistanceData(
            z, nullptr, z.cam_filterbank->thresholds_hz,
            comparison.analysis_a[channel_index].energy_channels_db,
            comparison.analysis_b[b_index][channel_index].energy_channels_db,
            "dB SPL", z.perceptual_sample_rate);
        std::cout << "    Raw channel distance: " << raw_channel_distance
                  << std::endl;

        const DistanceData masked_channel_distance = DistanceData(
            z, &raw_channel_distance, z.cam_filterbank->thresholds_hz,
            comparison.analysis_a[channel_index].partial_energy_channels_db,
            comparison.analysis_b[b_index][channel_index]
                .partial_energy_channels_db,
            "dB SPL", z.perceptual_sample_rate);
        std::cout << "    Masked channel distance: " << masked_channel_distance
                  << std::endl;

        const DistanceData phons_channel_distance = DistanceData(
            z, &masked_channel_distance, z.cam_filterbank->thresholds_hz,
            comparison.analysis_a[channel_index].spectrogram,
            comparison.analysis_b[b_index][channel_index].spectrogram, "Phons",
            z.perceptual_sample_rate);
        std::cout << "    Phons channel distance: " << phons_channel_distance
                  << std::endl;

        const float distance = phons_channel_distance.distance.value;
        sum_of_squares += distance * distance;

        std::cout << "    Channel MOS: " << z.mos_mapper.Map(distance)
                  << std::endl;
      }
      const float zimtohrli_file_distance =
          std::sqrt(sum_of_squares / float(comparison.analysis_a.size()));
      std::cout << "  File distance: " << zimtohrli_file_distance << std::endl;
      std::cout << "  File MOS: " << z.mos_mapper.Map(zimtohrli_file_distance)
                << std::endl;
    }
    return 0;
  }

  return 0;
}

}  // namespace

}  // namespace zimtohrli

int main(int argc, char* argv[]) { return zimtohrli::Main(argc, argv); }
