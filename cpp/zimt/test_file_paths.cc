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

#include "zimt/test_file_paths.h"

#include <filesystem>

namespace zimtohrli {

std::filesystem::path GetTestFilePath(
    const std::filesystem::path& relative_path) {
  return std::filesystem::path(_xstr(CMAKE_CURRENT_SOURCE_DIR)) / relative_path;
}

}  // namespace zimtohrli