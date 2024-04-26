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

#include "goohrli.h"

#include <algorithm>
#include <cstddef>
#include <utility>

#include "absl/types/span.h"
#include "hwy/aligned_allocator.h"
#include "hwy/base.h"
#include "zimt/cam.h"
#include "zimt/mos.h"
#include "zimt/visqol.h"
#include "zimt/zimtohrli.h"

float DefaultFrequencyResolution() {
  return zimtohrli::Cam{}.minimum_bandwidth_hz;
}

float DefaultPerceptualSampleRate() {
  return zimtohrli::Zimtohrli{}.perceptual_sample_rate;
}

EnergyAndMaxAbsAmplitude Measure(const float* signal, int size) {
  hwy::AlignedNDArray<float, 1> signal_array({static_cast<size_t>(size)});
  hwy::CopyBytes(signal, signal_array.data(), size * sizeof(float));
  const zimtohrli::EnergyAndMaxAbsAmplitude measurements =
      zimtohrli::Measure(signal_array[{}]);
  return EnergyAndMaxAbsAmplitude{
      .EnergyDBFS = measurements.energy_db_fs,
      .MaxAbsAmplitude = measurements.max_abs_amplitude};
}

EnergyAndMaxAbsAmplitude NormalizeAmplitude(float max_abs_amplitude,
                                            float* signal, int size) {
  hwy::AlignedNDArray<float, 1> signal_array({static_cast<size_t>(size)});
  hwy::CopyBytes(signal, signal_array.data(), size * sizeof(float));
  const zimtohrli::EnergyAndMaxAbsAmplitude measurements =
      zimtohrli::NormalizeAmplitude(max_abs_amplitude, signal_array[{}]);
  hwy::CopyBytes(signal_array.data(), signal, size * sizeof(float));
  return EnergyAndMaxAbsAmplitude{
      .EnergyDBFS = measurements.energy_db_fs,
      .MaxAbsAmplitude = measurements.max_abs_amplitude};
}

float MOSFromZimtohrli(float zimtohrli_distance) {
  return zimtohrli::MOSFromZimtohrli(zimtohrli_distance);
}

Zimtohrli CreateZimtohrli(float sample_rate, float frequency_resolution) {
  zimtohrli::Cam cam{.minimum_bandwidth_hz = frequency_resolution};
  cam.high_threshold_hz = std::min(cam.high_threshold_hz, sample_rate);
  return new zimtohrli::Zimtohrli{.cam_filterbank =
                                      cam.CreateFilterbank(sample_rate)};
}

void FreeZimtohrli(Zimtohrli zimtohrli) {
  delete static_cast<zimtohrli::Zimtohrli*>(zimtohrli);
}

Analysis Analyze(Zimtohrli zimtohrli, float* data, int size) {
  zimtohrli::Zimtohrli* z = static_cast<zimtohrli::Zimtohrli*>(zimtohrli);
  hwy::AlignedNDArray<float, 1> signal({static_cast<size_t>(size)});
  hwy::CopyBytes(data, signal.data(), size * sizeof(float));
  hwy::AlignedNDArray<float, 2> channels(
      {signal.shape()[0], z->cam_filterbank->filter.Size()});
  zimtohrli::Analysis analysis = z->Analyze(signal[{}], channels);
  return new zimtohrli::Analysis{
      .energy_channels_db = std::move(analysis.energy_channels_db),
      .partial_energy_channels_db =
          std::move(analysis.partial_energy_channels_db),
      .spectrogram = std::move(analysis.spectrogram)};
}

void FreeAnalysis(Analysis a) { delete static_cast<zimtohrli::Analysis*>(a); }

float AnalysisDistance(Zimtohrli zimtohrli, Analysis a, Analysis b,
                       int unwarp_window_samples) {
  zimtohrli::Zimtohrli* z = static_cast<zimtohrli::Zimtohrli*>(zimtohrli);
  zimtohrli::Analysis* analysis_a = static_cast<zimtohrli::Analysis*>(a);
  zimtohrli::Analysis* analysis_b = static_cast<zimtohrli::Analysis*>(b);
  return z
      ->Distance(false, analysis_a->spectrogram, analysis_b->spectrogram,
                 static_cast<size_t>(unwarp_window_samples))
      .value;
}

float GetTimeNormOrder(Zimtohrli zimtohrli) {
  return static_cast<zimtohrli::Zimtohrli*>(zimtohrli)->time_norm_order;
}

void SetTimeNormOrder(Zimtohrli zimtohrli, float f) {
  static_cast<zimtohrli::Zimtohrli*>(zimtohrli)->time_norm_order = f;
}

float GetFreqNormOrder(Zimtohrli zimtohrli) {
  return static_cast<zimtohrli::Zimtohrli*>(zimtohrli)->freq_norm_order;
}

void SetFreqNormOrder(Zimtohrli zimtohrli, float f) {
  static_cast<zimtohrli::Zimtohrli*>(zimtohrli)->freq_norm_order = f;
}

float GetPerceptualSampleRate(Zimtohrli zimtohrli) {
  return static_cast<zimtohrli::Zimtohrli*>(zimtohrli)->perceptual_sample_rate;
}

void SetPerceptualSampleRate(Zimtohrli zimtohrli, float f) {
  static_cast<zimtohrli::Zimtohrli*>(zimtohrli)->perceptual_sample_rate = f;
}

ViSQOL CreateViSQOL() { return new zimtohrli::ViSQOL(); }

void FreeViSQOL(ViSQOL v) { delete (zimtohrli::ViSQOL*)(v); }

MOSResult MOS(const ViSQOL v, float sample_rate, const float* reference,
              int reference_size, const float* distorted, int distorted_size) {
  const zimtohrli::ViSQOL* visqol = static_cast<const zimtohrli::ViSQOL*>(v);
  const absl::StatusOr<float> result = visqol->MOS(
      absl::Span<const float>(reference, reference_size),
      absl::Span<const float>(distorted, distorted_size), sample_rate);
  if (result.ok()) {
    return MOSResult{.MOS = result.value(), .Status = 0};
  } else {
    return MOSResult{.MOS = 0.0,
                     .Status = static_cast<int>(result.status().code())};
  }
}