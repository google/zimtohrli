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

#ifndef CPP_ZIMT_VISQOL_H_
#define CPP_ZIMT_VISQOL_H_

#include <filesystem>

#include "absl/status/statusor.h"
#include "zimt/zimtohrli.h"

namespace zimtohrli {

class ViSQOL {
 public:
  ViSQOL();
  ~ViSQOL();
  absl::StatusOr<float> MOS(Span<const float> reference,
                            Span<const float> degraded,
                            float sample_rate) const;

 private:
  std::filesystem::path model_path_;
};

}  // namespace zimtohrli

#endif  // CPP_ZIMT_VISQOL_H_
