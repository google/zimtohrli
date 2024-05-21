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

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "hwy/aligned_allocator.h"
#include "hwy/base.h"
#include "zimt/cam.h"
#include "zimt/fourier_bank.h"
#include "zimt/masking.h"
#include "zimt/mos.h"
#include "zimt/visqol.h"
#include "zimt/zimtohrli.h"

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

Zimtohrli CreateZimtohrli(ZimtohrliParameters params) {
  // Overriding lots of settings to ensure we have exactly tabuli::kNumRotators
  // channels.
  const zimtohrli::Cam def_cam;
  const float min_cam = def_cam.CamFromHz(def_cam.low_threshold_hz);
  const float max_cam = def_cam.CamFromHz(def_cam.high_threshold_hz);
  const float cam_step = (max_cam - min_cam) / tabuli::kNumRotators;
  const float hz_resolution =
      def_cam.HzFromCam(min_cam + cam_step) - def_cam.low_threshold_hz;
  zimtohrli::Cam cam{.minimum_bandwidth_hz = hz_resolution,
                     .filter_order = params.FilterOrder,
                     .filter_pass_band_ripple = params.FilterPassBandRipple,
                     .filter_stop_band_ripple = params.FilterStopBandRipple};
  CHECK_EQ(cam.CreateFilterbank(params.SampleRate).filter.Size(),
           tabuli::kNumRotators);
  cam.high_threshold_hz =
      std::min(cam.high_threshold_hz, params.SampleRate * 0.5f);
  zimtohrli::Zimtohrli* result = new zimtohrli::Zimtohrli{
      .cam_filterbank = cam.CreateFilterbank(params.SampleRate)};
  SetZimtohrliParameters(result, params);
  return result;
}

void FreeZimtohrli(Zimtohrli zimtohrli) {
  delete static_cast<zimtohrli::Zimtohrli*>(zimtohrli);
}

Analysis Analyze(Zimtohrli zimtohrli, float* data, int size) {
  zimtohrli::Zimtohrli* z = static_cast<zimtohrli::Zimtohrli*>(zimtohrli);
  hwy::AlignedNDArray<float, 1> signal({static_cast<size_t>(size)});
  hwy::CopyBytes(data, signal.data(), size * sizeof(float));
  hwy::AlignedNDArray<float, 2> channels(
      {signal.shape()[0], tabuli::kNumRotators});
  zimtohrli::Analysis analysis = z->Analyze(signal[{}], channels);
  return new zimtohrli::Analysis{
      .energy_channels_db = std::move(analysis.energy_channels_db),
      .partial_energy_channels_db =
          std::move(analysis.partial_energy_channels_db),
      .spectrogram = std::move(analysis.spectrogram)};
}

void FreeAnalysis(Analysis a) { delete static_cast<zimtohrli::Analysis*>(a); }

float AnalysisDistance(Zimtohrli zimtohrli, Analysis a, Analysis b) {
  zimtohrli::Zimtohrli* z = static_cast<zimtohrli::Zimtohrli*>(zimtohrli);
  zimtohrli::Analysis* analysis_a = static_cast<zimtohrli::Analysis*>(a);
  zimtohrli::Analysis* analysis_b = static_cast<zimtohrli::Analysis*>(b);
  return z->Distance(false, analysis_a->spectrogram, analysis_b->spectrogram)
      .value;
}

ZimtohrliParameters GetZimtohrliParameters(Zimtohrli zimtohrli) {
  zimtohrli::Zimtohrli* z = static_cast<zimtohrli::Zimtohrli*>(zimtohrli);
  ZimtohrliParameters result;
  result.SampleRate = z->cam_filterbank->sample_rate;
  const hwy::AlignedNDArray<float, 2>& thresholds =
      z->cam_filterbank->thresholds_hz;
  result.FrequencyResolution = thresholds[{2}][0] - thresholds[{0}][0];
  result.PerceptualSampleRate = z->perceptual_sample_rate;
  result.ApplyMasking = z->apply_masking;
  result.FullScaleSineDB = z->full_scale_sine_db;
  result.ApplyLoudness = z->apply_loudness;
  result.UnwarpWindowSeconds = z->unwarp_window_seconds;
  result.NSIMStepWindow = z->nsim_step_window;
  result.NSIMChannelWindow = z->nsim_channel_window;
  const zimtohrli::Masking& m = z->masking;
  result.MaskingLowerZeroAt20 = m.lower_zero_at_20;
  result.MaskingLowerZeroAt80 = m.lower_zero_at_80;
  result.MaskingUpperZeroAt20 = m.upper_zero_at_20;
  result.MaskingUpperZeroAt80 = m.upper_zero_at_80;
  result.MaskingMaxMask = m.max_mask;
  result.FilterOrder = z->cam_filterbank->filter_order;
  result.FilterPassBandRipple = z->cam_filterbank->filter_pass_band_ripple;
  result.FilterStopBandRipple = z->cam_filterbank->filter_stop_band_ripple;
  return result;
}

void SetZimtohrliParameters(Zimtohrli zimtohrli,
                            ZimtohrliParameters parameters) {
  zimtohrli::Zimtohrli* z = static_cast<zimtohrli::Zimtohrli*>(zimtohrli);
  z->perceptual_sample_rate = parameters.PerceptualSampleRate;
  z->apply_masking = parameters.ApplyMasking != 0;
  z->full_scale_sine_db = parameters.FullScaleSineDB;
  z->apply_loudness = parameters.ApplyLoudness != 0;
  z->nsim_step_window = parameters.NSIMStepWindow;
  z->nsim_channel_window = parameters.NSIMChannelWindow;
  z->unwarp_window_seconds = parameters.UnwarpWindowSeconds;
  z->masking.lower_zero_at_20 = parameters.MaskingLowerZeroAt20;
  z->masking.lower_zero_at_80 = parameters.MaskingLowerZeroAt80;
  z->masking.upper_zero_at_20 = parameters.MaskingUpperZeroAt20;
  z->masking.upper_zero_at_80 = parameters.MaskingUpperZeroAt80;
  z->masking.max_mask = parameters.MaskingMaxMask;
}

ZimtohrliParameters DefaultZimtohrliParameters() {
  const zimtohrli::Zimtohrli default_zimtohrli;
  ZimtohrliParameters result;
  result.SampleRate = -1;
  result.FrequencyResolution = zimtohrli::Cam{}.minimum_bandwidth_hz;
  result.PerceptualSampleRate = default_zimtohrli.perceptual_sample_rate;
  result.ApplyMasking = default_zimtohrli.apply_masking;
  result.FullScaleSineDB = default_zimtohrli.full_scale_sine_db;
  result.ApplyLoudness = default_zimtohrli.apply_loudness;
  result.UnwarpWindowSeconds = default_zimtohrli.unwarp_window_seconds;
  result.NSIMStepWindow = default_zimtohrli.nsim_step_window;
  result.NSIMChannelWindow = default_zimtohrli.nsim_channel_window;
  const zimtohrli::Masking m;
  result.MaskingLowerZeroAt20 = m.lower_zero_at_20;
  result.MaskingLowerZeroAt80 = m.lower_zero_at_80;
  result.MaskingUpperZeroAt20 = m.upper_zero_at_20;
  result.MaskingUpperZeroAt80 = m.upper_zero_at_80;
  result.MaskingMaxMask = m.max_mask;
  const zimtohrli::Cam c;
  result.FilterOrder = c.filter_order;
  result.FilterPassBandRipple = c.filter_pass_band_ripple;
  result.FilterStopBandRipple = c.filter_stop_band_ripple;
  return result;
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