// Copyright 2025 The Zimtohrli Authors. All Rights Reserved.
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
#include "zimt/mos.h"
#include "zimt/visqol.h"
#include "zimt/zimtohrli.h"

float SampleRate() { return zimtohrli::kSampleRate; }

int NumRotators() { return zimtohrli::kNumRotators; }

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

int SpectrogramSteps(Zimtohrli zimtohrli, int samples) {
  zimtohrli::Zimtohrli* z = static_cast<zimtohrli::Zimtohrli*>(zimtohrli);
  return z->SpectrogramSteps(static_cast<size_t>(samples));
}

void Analyze(Zimtohrli zimtohrli, const GoSpan* signal, GoSpectrogram* result) {
  zimtohrli::Zimtohrli* z = static_cast<zimtohrli::Zimtohrli*>(zimtohrli);
  zimtohrli::Spectrogram spec =
      zimtohrli::Spectrogram(result->steps, result->dims, result->values);
  z->Analyze(zimtohrli::Span(signal->data, signal->size), spec);
  spec.values.release();
}

float Distance(Zimtohrli zimtohrli, const GoSpectrogram* a, GoSpectrogram* b) {
  zimtohrli::Zimtohrli* z = static_cast<zimtohrli::Zimtohrli*>(zimtohrli);
  zimtohrli::Spectrogram spec_a =
      zimtohrli::Spectrogram(a->steps, a->dims, a->values);
  zimtohrli::Spectrogram spec_b =
      zimtohrli::Spectrogram(b->steps, b->dims, b->values);
  const float result = z->Distance(spec_a, spec_b);
  spec_a.values.release();
  spec_b.values.release();
  return result;
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

MOSResult ViSQOLMOS(const ViSQOL v, float sample_rate, const float* reference,
                    int reference_size, const float* distorted,
                    int distorted_size) {
  const zimtohrli::ViSQOL* visqol = static_cast<const zimtohrli::ViSQOL*>(v);
  const absl::StatusOr<float> result = visqol->MOS(
      zimtohrli::Span<const float>(reference, reference_size),
      zimtohrli::Span<const float>(distorted, distorted_size), sample_rate);
  if (result.ok()) {
    return MOSResult{.MOS = result.value(), .Status = 0};
  } else {
    return MOSResult{.MOS = 0.0,
                     .Status = static_cast<int>(result.status().code())};
  }
}
