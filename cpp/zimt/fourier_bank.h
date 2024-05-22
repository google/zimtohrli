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

float GetRotatorGains(int i);

enum FilterMode {
  IDENTITY,
  AMPLITUDE,
  PHASE,
};

float BarkFreq(float v);

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
  // Values inserted into the rotators are multiplied with this rotator in both
  // input and output, leading to a total gain multiplication if the length is
  // at sqrt(gain).
  float rot[4][kNumRotators] = {0};
  std::vector<PerChannel> channel;
  // Accu has the channel related data, everything else the same between
  // channels.
  float window[kNumRotators];
  float gain[kNumRotators];
  int16_t delay[kNumRotators] = {0};
  int16_t advance[kNumRotators] = {0};
  int16_t max_delay_ = 0;

  int FindMedian3xLeaker(float window);

  Rotators() = default;
  Rotators(int num_channels, std::vector<float> frequency,
           std::vector<float> filter_gains, const float sample_rate,
           float global_gain);

  void Filter(hwy::Span<const float> signal,
              hwy::AlignedNDArray<float, 2>& channels);

  void Increment(int c, int i, float audio);

  void AddAudio(int c, int i, float audio);
  void OccasionallyRenormalize();
  void IncrementAll();
  float GetSampleAll(int c);
  float GetSample(int c, int i, FilterMode mode = IDENTITY) const;
  std::vector<float> rotator_frequency;
};

static constexpr int64_t kBlockSize = 1 << 15;
static const int kHistorySize = (1 << 18);
static const int kHistoryMask = kHistorySize - 1;

float HardClip(float v);

struct RotatorFilterBank {
  RotatorFilterBank(size_t num_rotators, size_t num_channels, size_t samplerate,
                    size_t num_threads, const std::vector<float>& filter_gains,
                    float global_gain);
  ~RotatorFilterBank() = default;

  // TODO(jyrki): filter all at once in the generic case, filtering one
  // is not memory friendly in this memory tabulation.
  void FilterOne(size_t f_ix, const float* history, int64_t total_in,
                 int64_t len, FilterMode mode, float* output);

  int64_t FilterAllSingleThreaded(const float* history, int64_t total_in,
                                  int64_t len, FilterMode mode, float* output,
                                  size_t output_size);

  int64_t FilterAll(const float* history, int64_t total_in, int64_t len,
                    FilterMode mode, float* output, size_t output_size);

  size_t num_rotators_;
  size_t num_channels_;
  size_t num_threads_;
  std::unique_ptr<Rotators> rotators_;
  int64_t max_delay_;
  std::vector<std::vector<float>> filter_outputs_;
  std::atomic<size_t> next_task_{0};
};

}  // namespace tabuli

#endif  // _TABULI_FOURIER_BANK_H