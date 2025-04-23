#ifndef _TABULI_FOURIER_BANK_H
#define _TABULI_FOURIER_BANK_H

#include <algorithm>
#include <atomic>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <future>  // NOLINT
#include <memory>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/strings/str_split.h"
#include "hwy/aligned_allocator.h"
#include "sndfile.hh"

namespace tabuli {

constexpr int64_t kNumRotators = 128;

struct PerChannel {
  // [0..1] is for real and imag of 1st leaking accumulation
  // [2..3] is for real and imag of 2nd leaking accumulation
  // [4..5] is for real and imag of 3rd leaking accumulation
  float accu[6][kNumRotators] = {0};
};

struct Rotators {
  // Four arrays of rotators.
  // [0..1] is real and imag for rotation speed
  // [2..3] is real and image for a frequency rotator of length sqrt(gain[i])
  float rot[4][kNumRotators];
  PerChannel ch;
  // Accu has the channel related data, everything else the same between
  // channels.
  float window[kNumRotators];
  float gain[kNumRotators];

  void FilterAndDownsample(hwy::Span<const float> signal,
                           hwy::AlignedNDArray<float, 2>& channels,
                           int downsample);

  void OccasionallyRenormalize();
  void IncrementAll(float signal);
};

}  // namespace tabuli

#endif  // _TABULI_FOURIER_BANK_H
