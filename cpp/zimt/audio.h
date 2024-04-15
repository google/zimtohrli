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
#include "hwy/aligned_allocator.h"
#include "portaudio.h"
#include "sndfile.h"

namespace zimtohrli {

// Returns a string representation of the libsndfile format ID.
std::string GetFormatName(size_t format_id);

// Opens the audio system.
absl::Status OpenAudio();

// Closes the audio system.
absl::Status CloseAudio();

// Called when the sound starts playing, stops playing, and progresses in frame
// index.
//
// Guaranteed to be called on start and stop, but not on frame index progress.
using ProgressFunction = std::function<void(bool playing, size_t frame_index,
                                            const PaStreamInfo& info)>;

// An audio buffer.
struct AudioBuffer {
  // Plays the audio on the default audio device.
  //
  // If the progress function is provided, it will be called when the file
  // starts playing, stops playing, and (sometimes) when it progresses in
  // frame index.
  absl::StatusOr<const PaStreamInfo> Play(
      ProgressFunction progress = nullptr) const;

  // Returns the sample rate of this audio buffer.
  float SampleRate() const { return sample_rate; }

  // Returns the frames in this audio buffer.
  const hwy::AlignedNDArray<float, 2>& Frames() const { return frames; }

  // The sample rate the frames are expected to play at.
  float sample_rate;

  // A (num_channels, num_frames)-shaped array with samples between -1 and 1.
  hwy::AlignedNDArray<float, 2> frames;
};

// An audio file.
class AudioFile {
 public:
  // Reads from the path and returns an audio file.
  static absl::StatusOr<AudioFile> Load(const std::string& path);

  // Returns an audio buffer containing the data of this audio file.
  //
  // Destroys this audio file by moving the frames to the new buffer.
  AudioBuffer ToBuffer() && { return std::move(buffer_); }

  // Plays the audio on the default audio device.
  //
  // If the progress function is provided, it will be called when the file
  // starts playing, stops playing, and (sometimes) when it progresses in frame
  // index.
  absl::StatusOr<const PaStreamInfo> Play(
      ProgressFunction progress = nullptr) const;

  // Returns the sample rate of this audio file.
  float SampleRate() const { return buffer_.sample_rate; }

  // Returns the path this audio file was loaded from.
  const std::string& Path() const { return path_; }

  // Returns the metadata about this audio file.
  const SF_INFO& Info() const { return info_; }

  // Returns the metadata about this audio file.
  SF_INFO& Info() { return info_; }

  // Returns the frames in this audio file.
  const hwy::AlignedNDArray<float, 2>& Frames() const { return buffer_.frames; }

  // Returns the frames in this audio file.
  hwy::AlignedNDArray<float, 2>& Frames() { return buffer_.frames; }

 private:
  AudioFile(const std::string& path, const SF_INFO& info, AudioBuffer buffer)
      : path_(path), info_(info), buffer_(std::move(buffer)) {}
  std::string path_;
  SF_INFO info_;
  AudioBuffer buffer_;
};

}  // namespace zimtohrli

#endif  // CPP_ZIMT_AUDIO_FILE_H_
