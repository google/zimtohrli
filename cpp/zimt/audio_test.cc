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

#include "zimt/audio.h"

#include <cstddef>
#include <filesystem>

#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "gtest/gtest.h"
#include "zimt/test_file_paths.h"

namespace zimtohrli {

namespace {

TEST(AudioFile, LoadAudioFileTest) {
  const std::filesystem::path test_wav_path =
      GetTestFilePath("cpp/zimt/test.wav");
  absl::StatusOr<AudioFile> audio_file = AudioFile::Load(test_wav_path);
  CHECK_OK(audio_file.status());
  EXPECT_EQ(audio_file->Info().channels, 2);
  EXPECT_EQ(audio_file->Info().frames, 10);
  for (size_t frame_index = 0; frame_index < audio_file->Info().frames;
       ++frame_index) {
    for (size_t channel_index = 0; channel_index < audio_file->Info().channels;
         ++channel_index) {
      switch (channel_index) {
        case 0:
          switch (frame_index % 2) {
            case 0:
              EXPECT_EQ((*audio_file)[channel_index][frame_index], 0.5);
              break;
            case 1:
              EXPECT_EQ((*audio_file)[channel_index][frame_index], -0.5);
              break;
          }
          break;
        case 1:
          switch (frame_index % 2) {
            case 0:
              EXPECT_EQ((*audio_file)[channel_index][frame_index], 0.25);
              break;
            case 1:
              EXPECT_EQ((*audio_file)[channel_index][frame_index], -0.25);
              break;
          }
          break;
      }
    }
  }
}

}  // namespace

}  // namespace zimtohrli
