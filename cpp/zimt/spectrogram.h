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

#ifndef CPP_ZIMT_SPECTROGRAM_H_
#define CPP_ZIMT_SPECTROGRAM_H_

#include <cassert>
#include <cstddef>
#include <memory>
#include <vector>

namespace zimtohrli {

template <typename T>
struct Span {
  Span(std::vector<T>& vec) : size(vec.size()), data(vec.data()) {}
  explicit Span(size_t size, T* data) : size(size), data(data) {}
  template <typename U>
  Span(const std::vector<U>& vec) noexcept
      : data(vec.data()), size(vec.size()) {
    static_assert(std::is_convertible_v<U(*)[], T(*)[]>,
                  "Cannot construct Span from vector of incompatible type.");
  }
  template <typename U>
  Span(const Span<U>& other) noexcept : data(other.data), size(other.size) {
    static_assert(std::is_convertible_v<U(*)[], T(*)[]>,
                  "Cannot construct Span from Span of incompatible type.");
  }
  const T& operator[](size_t index) const { return data[index]; }
  T& operator[](size_t index) { return data[index]; }
  size_t size;
  T* data;
};

// A simple buffer of float samples describing a spectrogram with a given number
// of steps and feature dimensions.
// The values buffer is populated like:
// [
//   [sample0_dim0, sample0_dim1, ..., sample0_dimn],
//   [sample1_dim0, sample1_dim1, ..., sample1_dimn],
//   ...,
//   [samplem_dim0, samplem_dim1, ..., samplem_dimn],
// ]
struct Spectrogram {
  Spectrogram(size_t num_steps, size_t num_dims)
      : num_steps(num_steps),
        num_dims(num_dims),
        values(num_steps * num_dims) {};
  Spectrogram(size_t num_steps, size_t num_dims, std::vector<float> values)
      : num_steps(num_steps), num_dims(num_dims), values(values) {
    assert(num_steps * num_dims == values.size());
  }
  Span<const float> operator[](size_t n) const {
    return Span<const float>(num_dims, values.data() + n * num_dims);
  }
  Span<float> operator[](size_t n) {
    return Span(num_dims, values.data() + n * num_dims);
  }
  size_t num_steps;
  size_t num_dims;
  std::vector<float> values;
};

// A simple buffer of float samples describing an audio file with a given number
// of frames and channels.
// The frames buffer is populated like:
// [
//   [sample0_channel_0, sample1_channel0, ... samplen_channel0],
//   [sample0_channel_1, sample1_channel1, ... samplen_channel1],
//   ...,
//   [sample0_channel_m, sample1_channelm, ... samplen_channelm],
// ]
struct AudioBuffer {
  AudioBuffer(const AudioBuffer& other)
      : sample_rate(other.sample_rate),
        num_frames(other.num_frames),
        num_channels(other.num_channels) {
    frames = other.frames;
  }
  AudioBuffer(float sample_rate, size_t num_frames, size_t num_channels)
      : sample_rate(sample_rate),
        num_frames(num_frames),
        num_channels(num_channels) {
    frames = std::vector<float>(num_frames * num_channels);
  }
  Span<const float> operator[](size_t n) const {
    return Span<const float>(num_frames, frames.data() + num_frames * n);
  }
  Span<float> operator[](size_t n) {
    return Span<float>(num_frames, frames.data() + num_frames * n);
  }
  float sample_rate;
  size_t num_frames;
  size_t num_channels;
  std::vector<float> frames;
};

}  // namespace zimtohrli

#endif  // CPP_ZIMT_SPECTROGRAM_H_
