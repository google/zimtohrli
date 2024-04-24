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

#include "gosqol.h"

#include "zimt/visqol.h"

// void* representation of zimtohrli::ViSQOL.
typedef void* ViSQOL;

// Returns a zimtohrli::ViSQOL.
ViSQOL CreateViSQOL() { return new zimtohrli::ViSQOL(); }

// Deletes a zimtohrli::ViSQOL.
void FreeViSQOL(ViSQOL v) { delete (zimtohrli::ViSQOL*)(v); }

// Returns the ViSQOL MOS between the two sounds.
float MOS(const ViSQOL v, float sample_rate, const float* reference,
          int reference_size, const float* distorted, int distorted_size) {
  const zimtohrli::ViSQOL* visqol = static_cast<const zimtohrli::ViSQOL*>(v);
  return visqol->MOS(absl::Span<const float>(reference, reference_size),
                     absl::Span<const float>(distorted, distorted_size),
                     sample_rate);
}