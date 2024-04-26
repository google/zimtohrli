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

#include "zimt/visqol.h"

#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/types/span.h"
#include "visqol_api.h"
#include "zimt/resample.h"
#include "zimt/visqol_model.h"

constexpr size_t SAMPLE_RATE = 48000;

namespace zimtohrli {

ViSQOL::ViSQOL() {
  std::string path_template = (std::filesystem::temp_directory_path() /
                               "zimtohrli_cpp_zimt_visqol_model_XXXXXX")
                                  .string();
  std::vector<char> populated_path_template(path_template.begin(),
                                            path_template.end());
  populated_path_template.push_back('\0');
  const int model_path_file = mkstemp(populated_path_template.data());
  CHECK_GT(model_path_file, 0) << strerror(errno);
  CHECK_EQ(close(model_path_file), 0);
  model_path_ = std::filesystem::path(std::string(
      populated_path_template.data(), populated_path_template.size()));
  std::ofstream output_stream(model_path_);
  CHECK(output_stream.good());
  absl::Span<const char> model = ViSQOLModel();
  output_stream.write(model.data(), model.size());
  CHECK(output_stream.good());
  output_stream.close();
  CHECK(output_stream.good());
}

ViSQOL::~ViSQOL() { std::filesystem::remove(model_path_); }

float ViSQOL::MOS(absl::Span<const float> reference,
                  absl::Span<const float> degraded, float sample_rate) const {
  std::vector<double> resampled_reference =
      Resample<double>(reference, sample_rate, SAMPLE_RATE);
  std::vector<double> resampled_degraded =
      Resample<double>(degraded, sample_rate, SAMPLE_RATE);

  Visqol::VisqolConfig config;
  config.mutable_options()->set_svr_model_path(model_path_);
  config.mutable_audio()->set_sample_rate(SAMPLE_RATE);

  // When running in audio mode, sample rates of 48k is recommended for
  // the input signals. Using non-48k input will very likely negatively
  // affect the comparison result. If, however, API users wish to run with
  // non-48k input, set this to true.
  config.mutable_options()->set_allow_unsupported_sample_rates(false);

  // ViSQOL will run in audio mode comparison by default.
  // If speech mode comparison is desired, set to true.
  config.mutable_options()->set_use_speech_scoring(false);

  // Speech mode will scale the MOS mapping by default. This means that a
  // perfect NSIM score of 1.0 will be mapped to a perfect MOS-LQO of 5.0.
  // Set to true to use unscaled speech mode. This means that a perfect
  // NSIM score will instead be mapped to a MOS-LQO of ~4.x.
  config.mutable_options()->set_use_unscaled_speech_mos_mapping(false);

  Visqol::VisqolApi visqol;
  CHECK_OK(visqol.Create(config));

  absl::StatusOr<Visqol::SimilarityResultMsg> comparison_status_or =
      visqol.Measure(absl::Span<double>(resampled_reference.data(),
                                        resampled_reference.size()),
                     absl::Span<double>(resampled_degraded.data(),
                                        resampled_degraded.size()));
  CHECK_OK(comparison_status_or);

  Visqol::SimilarityResultMsg similarity_result = comparison_status_or.value();

  return similarity_result.moslqo();
}

}  // namespace zimtohrli