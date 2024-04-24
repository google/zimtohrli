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

#ifndef GO_LIB_GOSQOL_H_
#define GO_LIB_GOSQOL_H_

#ifdef __cplusplus
extern "C" {
#endif

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

#endif  // GO_LIB_GOSQOL_H_
