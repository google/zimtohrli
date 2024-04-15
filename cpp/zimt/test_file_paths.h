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

#ifndef CPP_ZIMT_TEST_FILE_PATHS_H_
#define CPP_ZIMT_TEST_FILE_PATHS_H_

#include <filesystem>

#ifndef CMAKE_CURRENT_SOURCE_DIR
#error "CMAKE_CURRENT_SOURCE_DIR must be #defined in the analysis test!"
#endif
#define _xstr(a) _str(a)
#define _str(a) #a

namespace zimtohrli {

std::filesystem::path GetTestFilePath(
    const std::filesystem::path& relative_path);

}  // namespace zimtohrli

#endif  // CPP_ZIMT_TEST_FILE_PATHS_H_
