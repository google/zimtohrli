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

#ifndef CPP_ZIMT_UX_H_
#define CPP_ZIMT_UX_H_

#include <cstddef>
#include <optional>
#include <vector>

#include "GLFW/glfw3.h"
#include "hwy/aligned_allocator.h"
#include "imgui.h"
#include "zimt/audio.h"
#include "zimt/zimtohrli.h"

namespace zimtohrli {

// A zimtohrli::Comparison with information about the compared files, and some
// metadata about how the comparison was performed.
struct FileComparison {
  // A sound A to compare to other sounds.
  AudioFile file_a;
  // An arbitrary number of other sounds B to compare to sound A.
  std::vector<AudioFile> file_b;
  // Comparison between file A and B.
  Comparison comparison;
  // min/peak/max frequencies for the filterbank when the analyses were
  // performed.
  hwy::AlignedNDArray<float, 2> thresholds_hz;
  // Reference dB SPL for a sine signal of amplitude 1.
  float full_scale_sine_db;
  // The frequency corresponding to the maximum time resolution, Hz.
  float perceptual_sample_rate;
  // The size of the unwarp window, if DTW is used.
  std::optional<size_t> unwarp_window;
};

// Initializes an ImGui UX for a comparison of audio files.
class UX {
 public:
  UX();
  ~UX();

  // Starts the painting + event loop for this UX.
  void Paint(FileComparison comparisons);

 private:
  GLFWwindow* window_;
  ImGuiIO* io_;
};

}  // namespace zimtohrli

#endif  // CPP_ZIMT_UX_H_
