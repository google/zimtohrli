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

#ifndef CPP_ZIMT_AUDIO_FILE_H_
#define CPP_ZIMT_AUDIO_FILE_H_

#include <cstddef>
#include <functional>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "sndfile.h"
#include "zimt/resample.h"
#include "zimt/zimtohrli.h"

namespace zimtohrli {

// Returns a string representation of the libsndfile format ID.
std::string GetFormatName(size_t format_id);

// An audio file.
class AudioFile {
 public:
  // Reads from the path and returns an audio file.
  static absl::StatusOr<AudioFile> Load(const std::string& path);

  // Returns the path this audio file was loaded from.
  const std::string& Path() const { return path_; }

  // Returns the metadata about this audio file.
  const SF_INFO& Info() const { return info_; }

  // Returns a channel of this audio file.
  Span<const float> operator[](size_t n) const {
    return Span<const float>(buffer_.data() + info_.frames * n, info_.frames);
  }

  // Returns a channel of this audio file.
  Span<float> operator[](size_t n) {
    return Span<float>(buffer_.data() + info_.frames * n, info_.frames);
  }

  std::vector<float> AtRate(size_t channel_id, float want_rate) {
    return Resample<float, float>(operator[](channel_id),
                                  static_cast<float>(info_.samplerate),
                                  want_rate);
  }

 private:
  AudioFile(const std::string& path, const SF_INFO& info,
            std::vector<float> buffer)
      : path_(path), info_(info), buffer_(buffer) {}
  std::string path_;
  SF_INFO info_;
  std::vector<float> buffer_;
};

}  // namespace zimtohrli

#endif  // CPP_ZIMT_AUDIO_FILE_H_
