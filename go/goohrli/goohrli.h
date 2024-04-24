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
// This file contains a C-compatible API layer simple to integrate in Go via
// cgo.
//
// All C++ classes used are aliased as void*, and all plumbing is done in
// C-style function calls.
//

#ifndef GO_LIB_GOOHRLI_H_
#define GO_LIB_GOOHRLI_H_

#ifdef __cplusplus
extern "C" {
#endif

// void* representation of zimtohrli::Zimtohrli.
typedef void* Zimtohrli;

// Returns the default frequency resolution.
float DefaultFrequencyResolution();

// Returns the default perceptual sample rate.
float DefaultPerceptualSampleRate();

// Returns a zimtohrli::Zimtohrli for the given parameters.
Zimtohrli CreateZimtohrli(float sample_rate, float frequency_resolution);

// Deletes a zimtohrli::Zimtohrli.
void FreeZimtohrli(Zimtohrli z);

// void* representation of zimtohrli::Analysis.
typedef void* Analysis;

// Returns a zimtohrli::Analysis produced by the provided zimtohrli::Zimtohrli
// and using the provided perceptual_sample_rate and data.
Analysis Analyze(Zimtohrli zimtohrli, float* data, int size);

// Plain C version of zimtohrli::EnergyAndMaxAbsAmplitude.
typedef struct {
  float EnergyDBFS;
  float MaxAbsAmplitude;
} EnergyAndMaxAbsAmplitude;

// Returns the energy in dB FS, and maximum absolute amplitude, of the signal.
EnergyAndMaxAbsAmplitude Measure(const float* signal, int size);

// Normalizes the amplitudes of the signal so that it has the provided max
// amplitude, and returns the new energ in dB FS, and the new maximum absolute
// amplitude.
EnergyAndMaxAbsAmplitude NormalizeAmplitude(float max_abs_amplitude,
                                            float* signal_data, int size);

// Returns a _very_approximate_ mean opinion score based on the
// provided Zimtohrli distance.
// This is calibrated using default settings of v0.1.5, with a
// minimum channel bandwidth (zimtohrli::Cam.minimum_bandwidth_hz)
// of 5Hz and perceptual sample rate
// (zimtohrli::Distance(..., perceptual_sample_rate, ...) of 100Hz.
float MOSFromZimtohrli(float zimtohrli_distance);

// Deletes a zimtohrli::Analysis.
void FreeAnalysis(Analysis a);

// Returns the Zimtohrli distance between two analyses using the provided
// zimtohrli::Zimtohrli and the unwarp window length.
float AnalysisDistance(Zimtohrli zimtohrli, Analysis a, Analysis b,
                       int unwarp_window_samples);

// Returns the order of the norm across time steps when computing Zimtohrli
// distance.
float GetTimeNormOrder(Zimtohrli zimtohrli);

// Sets the order of the norm across time steps when computing Zimtohrli
// distance.
void SetTimeNormOrder(Zimtohrli zimtohrli, float f);

// Returns the order of the norm across frequency channels when computing
// Zimtohrli distance.
float GetFreqNormOrder(Zimtohrli zimtohrli);

// Sets the order of the norm across frequency channels when computing Zimtohrli
// distance.
void SetFreqNormOrder(Zimtohrli zimtohrli, float f);

// Returns the perceptual sample rate used, corresponding to human hearing
// sensitivity to timing differences.
float GetPerceptualSampleRate(Zimtohrli zimtohrli);

// Sets the perceptual sample rate used.
void SetPerceptualSampleRate(Zimtohrli zimtohrli, float f);

// void* representation of zimtohrli::ViSQOL.
typedef void* ViSQOL;

// Returns a zimtohrli::ViSQOL.
ViSQOL CreateViSQOL();

// Deletes a zimtohrli::ViSQOL.
void FreeViSQOL(ViSQOL v);

// MOS returns a ViSQOL MOS between reference and distorted.
float MOS(ViSQOL v, float sample_rate, const float* reference,
          int reference_size, const float* distorted, int distorted_size);

#ifdef __cplusplus
}
#endif

#endif  // GO_LIB_GOOHRLI_H_
