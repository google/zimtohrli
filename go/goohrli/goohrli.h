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

#define NUM_LOUDNESS_A_F_PARAMS 10
#define NUM_LOUDNESS_L_U_PARAMS 16
#define NUM_LOUDNESS_T_F_PARAMS 13

// The supported sample rate for Zimtohrli.
float SampleRate();

// The number of rotators used by Zimtohrli, i.e. the number
// of dimensions in the spectrograms.
int NumRotators();

// Contains the parameters controlling Zimtohrli behavior.
typedef struct ZimtohrliParameters {
  float PerceptualSampleRate;
  float FullScaleSineDB;
  int NSIMStepWindow;
  int NSIMChannelWindow;
} ZimtohrliParameters;

// Returns the default parameters.
ZimtohrliParameters DefaultZimtohrliParameters();

// void* representation of zimtohrli::Zimtohrli.
typedef void* Zimtohrli;

// Returns a zimtohrli::Zimtohrli for the given parameters.
Zimtohrli CreateZimtohrli(ZimtohrliParameters params);

// Deletes a zimtohrli::Zimtohrli.
void FreeZimtohrli(Zimtohrli z);

// Returns the number of steps a spectrogram of the given number
// of samples requires.
int SpectrogramSteps(Zimtohrli zimtohrli, int samples);

// Represents a zimtohrli::Span.
typedef struct {
  float* data;
  int size;
} GoSpan;

// Represents a zimtohrli::Spectrogram.
typedef struct {
  float* values;
  int steps;
  int dims;
} GoSpectrogram;

// Returns a spectrogram by the provided zimtohrli::Zimtohrli using the provided
// data.
void Analyze(Zimtohrli zimtohrli, const GoSpan* signal, GoSpectrogram* spec);

// Returns an approximate mean opinion score based on the
// provided Zimtohrli distance.
// This is calibrated using default settings of v0.1.5, with a
// minimum channel bandwidth (zimtohrli::Cam.minimum_bandwidth_hz)
// of 5Hz and perceptual sample rate
// (zimtohrli::Distance(..., perceptual_sample_rate, ...) of 100Hz.
float MOSFromZimtohrli(float zimtohrli_distance);

// Returns the Zimtohrli distance between two analyses using the provided
// zimtohrli::Zimtohrli.
float Distance(Zimtohrli zimtohrli, const GoSpectrogram* a, GoSpectrogram* b);

// Sets the parameters.
//
// Sample rate, frequency resolution, and filter parameters can only be set when
// an instance is created and will be ignored in this function.
void SetZimtohrliParameters(Zimtohrli zimtohrli,
                            ZimtohrliParameters parameters);

// Returns the parameters.
ZimtohrliParameters GetZimtohrliParameters(Zimtohrli zimtohrli);

// void* representation of zimtohrli::ViSQOL.
typedef void* ViSQOL;

// Returns a zimtohrli::ViSQOL.
ViSQOL CreateViSQOL();

// Deletes a zimtohrli::ViSQOL.
void FreeViSQOL(ViSQOL v);

// MOSResult contains a MOS value and a status code.
typedef struct {
  float MOS;
  int Status;
} MOSResult;

// MOS returns a ViSQOL MOS between reference and distorted.
MOSResult ViSQOLMOS(ViSQOL v, float sample_rate, const float* reference,
                    int reference_size, const float* distorted,
                    int distorted_size);

#ifdef __cplusplus
}
#endif

#endif  // GO_LIB_GOOHRLI_H_
