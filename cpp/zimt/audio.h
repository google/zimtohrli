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

#ifndef CPP_ZIMT_AUDIO_FILE_H_
#define CPP_ZIMT_AUDIO_FILE_H_

#include <cstddef>
#include <functional>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "sndfile.h"
#include "zimt/spectrogram.h"

namespace zimtohrli {

// Returns a string representation of the libsndfile format ID.
std::string GetFormatName(size_t format_id);

// An audio file.
class AudioFile {
 public:
  // Reads from the path and returns an audio file.
  static absl::StatusOr<AudioFile> Load(const std::string& path);

  // Returns the sample rate of this audio file.
  float SampleRate() const { return buffer_.sample_rate; }

  // Returns the path this audio file was loaded from.
  const std::string& Path() const { return path_; }

  // Returns the metadata about this audio file.
  const SF_INFO& Info() const { return info_; }

  // Returns the metadata about this audio file.
  SF_INFO& Info() { return info_; }

  // Returns the frames in this audio file.
  const AudioBuffer& Buffer() const { return buffer_; }

  // Returns the frames in this audio file.
  AudioBuffer& Buffer() { return buffer_; }

 private:
  AudioFile(const std::string& path, const SF_INFO& info, AudioBuffer buffer)
      : path_(path), info_(info), buffer_(std::move(buffer)) {}
  std::string path_;
  SF_INFO info_;
  AudioBuffer buffer_;
};

}  // namespace zimtohrli

#endif  // CPP_ZIMT_AUDIO_FILE_H_
