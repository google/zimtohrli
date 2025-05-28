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
#include <cstring>
#include <utility>

#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "zimt/fourier_bank.h"
#include "zimt/mos.h"
#include "zimt/spectrogram.h"
#include "zimt/visqol.h"
#include "zimt/zimtohrli.h"

float SampleRate() { return zimtohrli::kSampleRate; }

EnergyAndMaxAbsAmplitude Measure(const float* signal, int size) {
  const zimtohrli::EnergyAndMaxAbsAmplitude measurements =
      zimtohrli::Measure(zimtohrli::Span(size, signal));
  return EnergyAndMaxAbsAmplitude{
      .EnergyDBFS = measurements.energy_db_fs,
      .MaxAbsAmplitude = measurements.max_abs_amplitude};
}

EnergyAndMaxAbsAmplitude NormalizeAmplitude(float max_abs_amplitude,
                                            float* signal, int size) {
  const zimtohrli::EnergyAndMaxAbsAmplitude measurements =
      zimtohrli::NormalizeAmplitude(max_abs_amplitude,
                                    zimtohrli::Span(size, signal));
  return EnergyAndMaxAbsAmplitude{
      .EnergyDBFS = measurements.energy_db_fs,
      .MaxAbsAmplitude = measurements.max_abs_amplitude};
}

float MOSFromZimtohrli(float zimtohrli_distance) {
  return zimtohrli::MOSFromZimtohrli(zimtohrli_distance);
}

Zimtohrli CreateZimtohrli(ZimtohrliParameters params) {
  zimtohrli::Zimtohrli* result = new zimtohrli::Zimtohrli{};
  SetZimtohrliParameters(result, params);
  return result;
}

void FreeZimtohrli(Zimtohrli zimtohrli) {
  delete static_cast<zimtohrli::Zimtohrli*>(zimtohrli);
}

GoSpectrogram Analyze(Zimtohrli zimtohrli, float* data, int size) {
  zimtohrli::Zimtohrli* z = static_cast<zimtohrli::Zimtohrli*>(zimtohrli);
  const size_t num_downscaled_samples = static_cast<size_t>(std::max(
      1.0f, std::ceil(static_cast<float>(size) * z->perceptual_sample_rate /
                      zimtohrli::kSampleRate)));
  zimtohrli::Spectrogram* spec = new zimtohrli::Spectrogram(
      num_downscaled_samples, zimtohrli::kNumRotators);
  z->Analyze(zimtohrli::Span(size, data), *spec);
  return spec;
}

void FreeSpec(GoSpectrogram a) {
  delete static_cast<zimtohrli::Spectrogram*>(a);
}

float Distance(Zimtohrli zimtohrli, GoSpectrogram a, GoSpectrogram b) {
  zimtohrli::Zimtohrli* z = static_cast<zimtohrli::Zimtohrli*>(zimtohrli);
  zimtohrli::Spectrogram* spec_a = static_cast<zimtohrli::Spectrogram*>(a);
  zimtohrli::Spectrogram* spec_b = static_cast<zimtohrli::Spectrogram*>(b);
  return z->Distance(*spec_a, *spec_b);
}

ZimtohrliParameters GetZimtohrliParameters(const Zimtohrli zimtohrli) {
  zimtohrli::Zimtohrli* z = static_cast<zimtohrli::Zimtohrli*>(zimtohrli);
  ZimtohrliParameters result;
  result.PerceptualSampleRate = z->perceptual_sample_rate;
  result.FullScaleSineDB = z->full_scale_sine_db;
  result.NSIMStepWindow = z->nsim_step_window;
  result.NSIMChannelWindow = z->nsim_channel_window;
  return result;
}

void SetZimtohrliParameters(Zimtohrli zimtohrli,
                            const ZimtohrliParameters parameters) {
  zimtohrli::Zimtohrli* z = static_cast<zimtohrli::Zimtohrli*>(zimtohrli);
  z->perceptual_sample_rate = parameters.PerceptualSampleRate;
  z->full_scale_sine_db = parameters.FullScaleSineDB;
  z->nsim_step_window = parameters.NSIMStepWindow;
  z->nsim_channel_window = parameters.NSIMChannelWindow;
}

ZimtohrliParameters DefaultZimtohrliParameters() {
  zimtohrli::Zimtohrli default_zimtohrli{};
  return GetZimtohrliParameters(&default_zimtohrli);
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
