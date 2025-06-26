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
#include "sndfile.h"
#include "zimt/zimtohrli.h"

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

absl::StatusOr<AudioFile> AudioFile::Load(const std::string& path) {
  SF_INFO info{};
  SNDFILE* file = sf_open(path.c_str(), SFM_READ, &info);
  if (sf_error(file)) {
    return absl::InternalError(sf_strerror(file));
  }
  std::vector<float> samples(info.channels * info.frames);
  CHECK_EQ(sf_readf_float(file, samples.data(), info.frames), info.frames);
  std::vector<float> buffer(info.frames * info.channels);
  for (size_t frame_index = 0; frame_index < info.frames; ++frame_index) {
    for (size_t channel_index = 0; channel_index < info.channels;
         ++channel_index) {
      buffer[channel_index * info.frames + frame_index] =
          samples[frame_index * info.channels + channel_index];
    }
  }
  sf_close(file);
  return AudioFile(path, info, std::move(buffer));
}

}  // namespace zimtohrli
