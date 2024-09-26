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

#define NUM_LOUDNESS_A_F_PARAMS 10
#define NUM_LOUDNESS_L_U_PARAMS 16
#define NUM_LOUDNESS_T_F_PARAMS 13
#define NUM_MOS_MAPPER_PARAMS 3

// Returns the number of LoudnessAFParams in ZimtohrliParameters.
int NumLoudnessAFParams();

// Returns the number of LoudnessLUParams in ZimtohrliParameters.
int NumLoudnessLUParams();

// Returns the number of LoudnessTFParams in ZimtohrliParameters.
int NumLoudnessTFParams();

// Returns the number of MOSMapperParams in ZimtohrliParameters.
int NumMOSMapperParams();

// Contains the parameters controlling Zimtohrli behavior.
typedef struct ZimtohrliParameters {
  float SampleRate;
  float FrequencyResolution;
  float PerceptualSampleRate;
  int ApplyMasking;
  float FullScaleSineDB;
  int ApplyLoudness;
  float UnwarpWindowSeconds;
  int NSIMStepWindow;
  int NSIMChannelWindow;
  float MaskingLowerZeroAt20;
  float MaskingLowerZeroAt80;
  float MaskingUpperZeroAt20;
  float MaskingUpperZeroAt80;
  float MaskingMaxMask;
  int FilterOrder;
  float FilterStopBandRipple;
  float FilterPassBandRipple;
  float LoudnessAFParams[NUM_LOUDNESS_A_F_PARAMS];
  float LoudnessLUParams[NUM_LOUDNESS_L_U_PARAMS];
  float LoudnessTFParams[NUM_LOUDNESS_T_F_PARAMS];
  float MOSMapperParams[NUM_MOS_MAPPER_PARAMS];
} ZimtohrliParameters;

// Returns the default parameters.
ZimtohrliParameters DefaultZimtohrliParameters(float sample_rate);

// void* representation of zimtohrli::Zimtohrli.
typedef void* Zimtohrli;

// Returns a zimtohrli::Zimtohrli for the given parameters.
Zimtohrli CreateZimtohrli(const ZimtohrliParameters params);

// Deletes a zimtohrli::Zimtohrli.
void FreeZimtohrli(Zimtohrli z);

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
float MOSFromZimtohrli(const Zimtohrli zimtohrli, float zimtohrli_distance);

// Returns the Zimtohrli distance between data_b and data_b.
float Distance(const Zimtohrli zimtohrli, float* data_a, int size_a,
               float* data_b, int size_b);

// Sets the parameters.
//
// Sample rate, frequency resolution, and filter parameters can only be set when
// an instance is created and will be ignored in this function.
void SetZimtohrliParameters(Zimtohrli zimtohrli,
                            ZimtohrliParameters parameters);

// Returns the parameters.
ZimtohrliParameters GetZimtohrliParameters(const Zimtohrli zimtohrli);

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
MOSResult MOS(const ViSQOL v, float sample_rate, const float* reference,
              int reference_size, const float* distorted, int distorted_size);

#ifdef __cplusplus
}
#endif

#endif  // GO_LIB_GOOHRLI_H_
