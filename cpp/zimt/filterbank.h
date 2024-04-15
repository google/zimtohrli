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

#ifndef CPP_ZIMT_FILTERBANK_H_
#define CPP_ZIMT_FILTERBANK_H_

#include <stddef.h>

#include <array>
#include <optional>
#include <vector>

#include "hwy/aligned_allocator.h"
#include "zimt/elliptic.h"

namespace zimtohrli {

// State for the filterbank.
struct FilterbankState {
  hwy::AlignedNDArray<float, 3> x_buffer;
  hwy::AlignedNDArray<float, 3> y_buffer;
  size_t global_sample_index;
};

// Handles filtering signals with a continuous filter.
//
// Uses filter coefficients exactly like
// https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfilt.html
class Filterbank {
 public:
  // Constructs a filter with the provided coefficients in sections of second
  // order filters.
  //
  // Each element in filters is an individual filter defined as sections of
  // second order filters, i.e. the return value of DigitalSOSBandPass.
  explicit Filterbank(const std::vector<std::vector<BACoeffs>>& filters);

  // Filters the input signal into the output signals buffer.
  //
  // Returns a (num_samples, num_channels)-shaped array.
  //
  // The state is used to keep track of previous inputs and outputs to enable
  // processing a signal in chunks.
  void Filter(hwy::Span<const float> input, FilterbankState& state,
              hwy::AlignedNDArray<float, 2>& output) const;

  // Filter without chunk processing.
  void Filter(hwy::Span<const float> input,
              hwy::AlignedNDArray<float, 2>& output) const;

  // Returns the number of filters in the bank.
  size_t Size() const;

  // Returns state for a filterbank starting from scratch.
  FilterbankState NewState() const;

 private:
  // (num_sections, num_coeffs, num_filters)
  hwy::AlignedNDArray<float, 3> b_coeffs_;
  hwy::AlignedNDArray<float, 3> a_coeffs_;
  std::array<size_t, 3> x_buffer_shape_;
  std::array<size_t, 3> y_buffer_shape_;
};

}  // namespace zimtohrli

#endif  // CPP_ZIMT_FILTERBANK_H_
