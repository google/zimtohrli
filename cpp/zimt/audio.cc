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
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "hwy/aligned_allocator.h"
#include "hwy/base.h"
#include "portaudio.h"
#include "sndfile.h"

namespace zimtohrli {

std::string GetFormatName(size_t format_id) {
  if (format_id & SF_FORMAT_WAV) {
    return "wav";
  } else if (format_id & SF_FORMAT_AIFF) {
    return "aiff";
  } else if (format_id & SF_FORMAT_AU) {
    return "au";
  } else if (format_id & SF_FORMAT_RAW) {
    return "raw";
  } else if (format_id & SF_FORMAT_PAF) {
    return "paf";
  } else if (format_id & SF_FORMAT_SVX) {
    return "svx";
  } else if (format_id & SF_FORMAT_NIST) {
    return "nist";
  } else if (format_id & SF_FORMAT_VOC) {
    return "voc";
  } else if (format_id & SF_FORMAT_IRCAM) {
    return "ircam";
  } else if (format_id & SF_FORMAT_W64) {
    return "w64";
  } else if (format_id & SF_FORMAT_MAT4) {
    return "mat4";
  } else if (format_id & SF_FORMAT_MAT5) {
    return "mat5";
  } else if (format_id & SF_FORMAT_PVF) {
    return "pvf";
  } else if (format_id & SF_FORMAT_XI) {
    return "xi";
  } else if (format_id & SF_FORMAT_HTK) {
    return "htk";
  } else if (format_id & SF_FORMAT_SDS) {
    return "sds";
  } else if (format_id & SF_FORMAT_AVR) {
    return "avr";
  } else if (format_id & SF_FORMAT_WAVEX) {
    return "wavex";
  } else if (format_id & SF_FORMAT_SD2) {
    return "sd2";
  } else if (format_id & SF_FORMAT_FLAC) {
    return "flac";
  } else if (format_id & SF_FORMAT_CAF) {
    return "caf";
  } else if (format_id & SF_FORMAT_WVE) {
    return "wve";
  } else if (format_id & SF_FORMAT_OGG) {
    return "ogg";
  } else if (format_id & SF_FORMAT_MPC2K) {
    return "mpc2k";
  } else if (format_id & SF_FORMAT_RF64) {
    return "rf64";
  } else if (format_id & SF_FORMAT_MPEG) {
    return "mpeg";
  }
  return "unknown";
}

absl::Status OpenAudio() {
  PaError err = Pa_Initialize();
  return err == paNoError ? absl::OkStatus()
                          : absl::InternalError(Pa_GetErrorText(err));
}

absl::Status CloseAudio() {
  PaError err = Pa_Terminate();
  return err == paNoError ? absl::OkStatus()
                          : absl::InternalError(Pa_GetErrorText(err));
}

namespace {

struct StreamState {
  hwy::AlignedNDArray<float, 2> frames;
  PaStreamInfo stream_info;
  ProgressFunction progress;
  size_t next_frame;
};

int AudioCallback(ABSL_ATTRIBUTE_UNUSED const void* input_buffer,
                  void* output_buffer, uint64_t frames_per_buffer,
                  const PaStreamCallbackTimeInfo* time_info,
                  PaStreamCallbackFlags status_flags, void* user_data) {
  // Cast data passed through stream to our structure.
  StreamState& state = *static_cast<StreamState*>(user_data);
  const size_t num_channels = state.frames.shape()[0];
  const size_t num_frames = state.frames.shape()[1];

  float* out = static_cast<float*>(output_buffer);

  uint64_t frame_index;
  for (frame_index = 0; frame_index < frames_per_buffer &&
                        state.next_frame + frame_index < num_frames;
       ++frame_index) {
    for (size_t channel_index = 0; channel_index < num_channels;
         ++channel_index) {
      out[frame_index * num_channels + channel_index] =
          state.frames[{channel_index}][state.next_frame + frame_index];
    }
  }
  state.next_frame += frames_per_buffer;
  if (state.next_frame >= num_frames) {
    if (state.progress != nullptr)
      state.progress(false, state.next_frame, state.stream_info);
    // The stream is finished, remove the state.
    delete &state;
    return paComplete;
  }
  if (state.progress) state.progress(true, state.next_frame, state.stream_info);
  return paContinue;
}

absl::StatusOr<const PaStreamInfo> PlayFrames(
    const hwy::AlignedNDArray<float, 2>& frames, float sample_rate,
    ProgressFunction progress) {
  hwy::AlignedNDArray<float, 2> frames_copy(frames.memory_shape());
  hwy::CopyBytes(frames.data(), frames_copy.data(),
                 sizeof(float) * frames.memory_size());
  frames_copy.truncate(frames.shape());
  PaStream* stream;
  PaError err;
  StreamState* state = new StreamState({.frames = std::move(frames_copy),
                                        .progress = std::move(progress),
                                        .next_frame = 0});
  err = Pa_OpenDefaultStream(
      &stream, /* inputChannelCount */ 0, frames.shape()[0],
      /* sampleFormat */ paFloat32, sample_rate, paFramesPerBufferUnspecified,
      AudioCallback, state);

  if (err == paNoError) {
    err = Pa_StartStream(stream);
  }
  if (err == paNoError) {
    // state will be deleted by AudioCallback.
  } else {
    delete state;
    return absl::InternalError(Pa_GetErrorText(err));
  }
  state->stream_info = *Pa_GetStreamInfo(stream);
  if (state->progress != nullptr)
    state->progress(true, state->next_frame, state->stream_info);
  return state->stream_info;
}

}  // namespace

absl::StatusOr<const PaStreamInfo> AudioBuffer::Play(
    ProgressFunction progress) const {
  return zimtohrli::PlayFrames(frames, sample_rate, progress);
}

absl::StatusOr<const PaStreamInfo> AudioFile::Play(
    ProgressFunction progress) const {
  return zimtohrli::PlayFrames(buffer_.frames, info_.samplerate, progress);
}

absl::StatusOr<AudioFile> AudioFile::Load(const std::string& path) {
  SF_INFO info{};
  SNDFILE* file = sf_open(path.c_str(), SFM_READ, &info);
  if (sf_error(file)) {
    return absl::InternalError(sf_strerror(file));
  }
  std::vector<float> samples(info.channels * info.frames);
  CHECK_EQ(sf_readf_float(file, samples.data(), info.frames), info.frames);
  hwy::AlignedNDArray<float, 2> frames(
      {static_cast<size_t>(info.channels), static_cast<size_t>(info.frames)});
  for (size_t frame_index = 0; frame_index < info.frames; ++frame_index) {
    for (size_t channel_index = 0; channel_index < info.channels;
         ++channel_index) {
      frames[{channel_index}][frame_index] =
          samples[frame_index * info.channels + channel_index];
    }
  }
  sf_close(file);
  return AudioFile(path, info,
                   {.sample_rate = static_cast<float>(info.samplerate),
                    .frames = std::move(frames)});
}

}  // namespace zimtohrli
