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

#ifndef CPP_ZIMT_ZIMTOHRLI_H_
#define CPP_ZIMT_ZIMTOHRLI_H_

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <optional>
#include <utility>
#include <vector>

namespace zimtohrli {

namespace {

constexpr int64_t kNumRotators = 128;

inline void LoudnessDb(float* channels) {
  static const float kMul[128] = {
      0.69022, 0.68908, 0.69206, 0.68780, 0.68780, 0.68780, 0.68780, 0.68780,
      0.68780, 0.68780, 0.68780, 0.68913, 0.69045, 0.69310, 0.69575, 0.69565,
      0.69697, 0.70122, 0.72878, 0.79911, 0.85713, 0.88063, 0.88563, 0.87561,
      0.81948, 0.70435, 0.63479, 0.58382, 0.52065, 0.48390, 0.46452, 0.47952,
      0.52686, 0.63677, 0.75972, 0.89449, 0.97411, 1.01874, 1.01105, 0.99306,
      0.93613, 0.92825, 0.93149, 0.98687, 1.05782, 1.16461, 1.25028, 1.30768,
      1.31484, 1.28574, 1.23002, 1.15336, 1.08800, 1.01472, 0.94610, 0.91856,
      0.87797, 0.85825, 0.82836, 0.82198, 0.81394, 0.82724, 0.84235, 0.86009,
      0.88276, 0.89349, 0.92543, 0.94822, 0.98526, 0.99730, 1.02097, 1.04071,
      1.05254, 1.06462, 1.06872, 1.07382, 1.06739, 1.06331, 1.05118, 1.05002,
      1.04803, 1.06729, 1.09680, 1.15208, 1.22492, 1.32630, 1.42049, 1.50444,
      1.58735, 1.65199, 1.69488, 1.70748, 1.74525, 1.68760, 1.66818, 1.63401,
      1.55136, 1.49170, 1.42649, 1.33453, 1.28618, 1.26523, 1.24900, 1.24898,
      1.27864, 1.28723, 1.28455, 1.29777, 1.29637, 1.29687, 1.29853, 1.30319,
      1.30207, 1.26835, 1.25100, 1.24664, 1.24041, 1.17874, 1.07116, 0.97917,
  };
  static const float kBaseNoise = 886753.16118050041;
  for (int k = 0; k < kNumRotators; ++k) {
    channels[k] = log(channels[k] + kBaseNoise) * kMul[k];
  }
}

// Ear drum and other receiving mass-spring objects are
// modeled through the Resonator. Resonator is a non-linear process
// and does complex spectral shifting of energy.
struct Resonator {
  float acc0 = 0;
  float acc1 = 0;
  float Update(float signal) {  // Resonate and attenuate.
    // These parameters relate to a population of ear drums.
    static const float kMul0 = 0.93913835617233998;
    static const float kMul1 = -0.040539506065308289;
    acc0 = kMul0 * acc0 + kMul1 * acc1 + signal;
    acc1 += acc0;
    return acc0;
  }
};

inline float Dot32(const float* a, const float* b) {
  // -ffast-math is helpful here, and clang can simdify this.
  float sum = 0;
  for (int i = 0; i < 32; ++i) sum += a[i] * b[i];
  return sum;
}

float Freq(int i) {
  // Center frequencies of the filter bank, plus one frequency in both ends.
  static const float kFreq[130] = {
      17.858,  24.349,  33.199,  42.359,  51.839,  61.651,  71.805,  82.315,
      93.192,  104.449, 116.099, 128.157, 140.636, 153.552, 166.919, 180.754,
      195.072, 209.890, 225.227, 241.099, 257.527, 274.528, 292.124, 310.336,
      329.183, 348.690, 368.879, 389.773, 411.398, 433.778, 456.941, 480.914,
      505.725, 531.403, 557.979, 585.484, 613.950, 643.411, 673.902, 705.459,
      738.119, 771.921, 806.905, 843.111, 880.584, 919.366, 959.503, 1001.04,
      1044.03, 1088.53, 1134.58, 1182.24, 1231.57, 1282.62, 1335.46, 1390.14,
      1446.73, 1505.31, 1565.93, 1628.67, 1693.60, 1760.80, 1830.35, 1902.34,
      1976.84, 2053.94, 2133.74, 2216.33, 2301.81, 2390.27, 2481.83, 2576.58,
      2674.65, 2776.15, 2881.19, 2989.91, 3102.43, 3218.88, 3339.40, 3464.14,
      3593.23, 3726.84, 3865.12, 4008.23, 4156.35, 4309.64, 4468.30, 4632.49,
      4802.43, 4978.31, 5160.34, 5348.72, 5543.70, 5745.49, 5954.34, 6170.48,
      6394.18, 6625.70, 6865.32, 7113.31, 7369.97, 7635.61, 7910.53, 8195.06,
      8489.53, 8794.30, 9109.73, 9436.18, 9774.04, 10123.7, 10485.6, 10860.1,
      11247.8, 11648.9, 12064.2, 12493.9, 12938.7, 13399.0, 13875.3, 14368.4,
      14878.7, 15406.8, 15953.4, 16519.1, 17104.5, 17710.4, 18337.6, 18986.6,
      19658.3, 20352.7,
  };
  return kFreq[i + 1];
}

double CalculateBandwidthInHz(int i) {
  return std::sqrt(Freq(i + 1) * Freq(i)) - std::sqrt(Freq(i - 1) * Freq(i));
}

class Rotators {
 private:
  // Four arrays of rotators, with memory layout for up to 128-way
  // simd-parallel. [0..1] is real and imag for rotation speed [2..3] is real
  // and image for a frequency rotator of length sqrt(gain[i])
  float rot[4][kNumRotators];
  // [0..1] is for real and imag of 1st leaking accumulation
  // [2..3] is for real and imag of 2nd leaking accumulation
  // [4..5] is for real and imag of 3rd leaking accumulation
  float accu[6][kNumRotators] = {0};
  float window[kNumRotators];
  float gain[kNumRotators];

  void OccasionallyRenormalize() {
    for (int i = 0; i < kNumRotators; ++i) {
      float norm =
          gain[i] / sqrt(rot[2][i] * rot[2][i] + rot[3][i] * rot[3][i]);
      rot[2][i] *= norm;
      rot[3][i] *= norm;
    }
  }
  void IncrementAll(float signal) {
    for (int i = 0; i < kNumRotators; i++) {  // clang simdifies this.
      const float w = window[i];
      for (int k = 0; k < 6; ++k) accu[k][i] *= w;
      accu[2][i] += accu[0][i];  // For an unknown reason this
      accu[3][i] += accu[1][i];  // update order works best.
      accu[4][i] += accu[2][i];  // i.e., 2, 3, 4, 5, and finally 0, 1.
      accu[5][i] += accu[3][i];
      accu[0][i] += rot[2][i] * signal;
      accu[1][i] += rot[3][i] * signal;
      const float a = rot[2][i], b = rot[3][i];
      rot[2][i] = rot[0][i] * a - rot[1][i] * b;
      rot[3][i] = rot[0][i] * b + rot[1][i] * a;
    }
  }

 public:
  void FilterAndDownsample(const float* in, size_t in_size, float* out,
                           size_t out_shape0, size_t out_stride,
                           int downsample) {
    static const float kSampleRate = 48000.0;
    static const float kHzToRad = 2.0f * M_PI / kSampleRate;
    static const double kWindow = 0.9996028710680265;
    static const double kBandwidthMagic = 0.7328516996032982;
    // A big value for normalization. Ideally 1.0, but this works better.
    static const double kScale = 928170036864.07068;
    const float gainer = sqrt(kScale / downsample);
    for (int i = 0; i < kNumRotators; ++i) {
      float bandwidth = CalculateBandwidthInHz(i);  // bandwidth per bucket.
      window[i] = std::pow(kWindow, bandwidth * kBandwidthMagic);
      float windowM1 = 1.0f - window[i];
      const float f = Freq(i) * kHzToRad;
      gain[i] = gainer * pow(windowM1, 3.0) * Freq(i) / bandwidth;
      rot[0][i] = float(std::cos(f));
      rot[1][i] = float(-std::sin(f));
      rot[2][i] = gain[i];
      rot[3][i] = 0.0f;
    }
    for (size_t zz = 0; zz < out_shape0; zz++) {
      for (int k = 0; k < kNumRotators; ++k) {
        out[zz * out_stride + k] = 0;
      }
    }
    std::vector<float> window(downsample);
    for (int i = 0; i < downsample; ++i) {
      window[i] =
          1.0 / (1.0 + exp(7.9446 * ((2.0 / downsample) * (i + 0.5) - 1)));
    }
    Resonator resonator;
    size_t out_ix = 0;
    constexpr size_t kKernelSize = 32;
    static const float reso_kernel[kKernelSize] = {
        -0.00756885973, 0.00413482141,  -0.00000236200, 0.00619875373,
        -0.00283612301, -0.00000418032, -0.00653942799, -0.00697059266,
        0.00344293224,  0.00329933933,  -0.00298496041, 0.00350131041,
        0.00171017251,  -0.00154158276, 0.00404768079,  0.00127457555,
        -0.01171138281, -0.00010813847, -0.00152608046, -0.00838915828,
        -0.00640430929, -0.00086448874, -0.00720815920, 0.00344734180,
        -0.00294620320, 0.00079453551,  0.00067657883,  0.00185866424,
        0.00615985137,  -0.00236233239, -0.00680980952, 0.01082403830,
    };
    static const float linear_kernel[kKernelSize] = {
        -0.10104347418, -0.11826972031, -0.06180710258, 0.07855591921,
        0.03670823911,  -0.01840452136, 0.10859856308,  0.16449286025,
        0.06054576192,  0.08362268315,  -0.00320242077, 0.17410886426,
        -0.13348931125, 0.12798560564,  0.02840772721,  0.01655141242,
        0.00565097497,  -0.39669214512, 0.25126981719,  0.29050002107,
        -0.34990576312, 0.13135342797,  1.09071850579,  -0.97998963695,
        -0.97386487573, 0.30687938104,  0.52811340907,  1.35094332106,
        0.35339301883,  -0.17657465769, 0.36698233014,  -0.39494225991,
    };
    for (size_t in_ix = 0, dix = 0; in_ix + kKernelSize < in_size; ++in_ix) {
      const float weight = window[dix];
      IncrementAll(resonator.Update(Dot32(&in[in_ix], &reso_kernel[0])) +
                   Dot32(&in[in_ix], &linear_kernel[0]));
      if (out_ix + 1 < out_shape0) {
        for (int k = 0; k < kNumRotators; ++k) {
          float energy = accu[4][k] * accu[4][k] + accu[5][k] * accu[5][k];
          out[(out_ix + 1) * out_stride + k] += (1.0 - weight) * energy;
          out[out_ix * out_stride + k] += weight * energy;
        }
      } else {
        for (int k = 0; k < kNumRotators; ++k) {
          float energy = accu[4][k] * accu[4][k] + accu[5][k] * accu[5][k];
          out[out_ix * out_stride + k] += energy;
        }
      }
      if (++dix == downsample || in_ix + kKernelSize + 1 == in_size) {
        LoudnessDb(&out[out_stride * out_ix]);
        if (++out_ix >= out_shape0) {
          break;
        }
        dix = 0;
        OccasionallyRenormalize();
      }
    }
  }
};

template <typename T>
struct Span {
  Span(const Span& other) = default;
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
  Span& operator=(const Span& other) = default;
  const T& operator[](size_t index) const { return data[index]; }
  T& operator[](size_t index) { return data[index]; }
  size_t size;
  T* data;
};

// A simple buffer of float samples describing a spectrogram with a given number
// of steps and feature dimensions.
//
// Similar to AudioBuffer, except transposed.
//
// The values buffer is populated like:
// [
//   [sample0_dim0, sample0_dim1, ..., sample0_dimn],
//   [sample1_dim0, sample1_dim1, ..., sample1_dimn],
//   ...,
//   [samplem_dim0, samplem_dim1, ..., samplem_dimn],
// ]
struct Spectrogram {
  Spectrogram(Spectrogram&& other) = default;
  Spectrogram(size_t num_steps)
      : num_steps(num_steps),
        num_dims(kNumRotators),
        values(num_steps * kNumRotators) {}
  Spectrogram(size_t num_steps, size_t num_dims)
      : num_steps(num_steps),
        num_dims(num_dims),
        values(num_steps * num_dims) {}
  Spectrogram(size_t num_steps, size_t num_dims, std::vector<float> values)
      : num_steps(num_steps), num_dims(num_dims), values(values) {
    assert(num_steps * num_dims == values.size());
  }
  Spectrogram& operator=(Spectrogram&& other) = default;
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
//
// Similar to Spectrogram, except transposed.
//
// The frames buffer is populated like:
// [
//   [sample0_channel_0, sample1_channel0, ... samplen_channel0],
//   [sample0_channel_1, sample1_channel1, ... samplen_channel1],
//   ...,
//   [sample0_channel_m, sample1_channelm, ... samplen_channelm],
// ]
struct AudioBuffer {
  AudioBuffer(AudioBuffer&& other) = default;
  AudioBuffer(float sample_rate, size_t num_frames, size_t num_channels)
      : sample_rate(sample_rate),
        num_frames(num_frames),
        num_channels(num_channels) {
    frames = std::vector<float>(num_frames * num_channels);
  }
  AudioBuffer& operator=(AudioBuffer&& other) = default;
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

template <typename T>
Spectrogram WindowMean(size_t num_steps, size_t num_channels,
                       size_t step_window, size_t channel_window,
                       T input_loader) {
  Spectrogram tmp_a(num_steps, num_channels);
  Spectrogram tmp_b(num_steps, num_channels);

  // Populate tmp_b with prefix sums across the step axis.
  {
    Span<float> channel_prefix_sum_data = tmp_b[0];
    for (size_t channel_index = 0; channel_index < num_channels;
         ++channel_index) {
      channel_prefix_sum_data[channel_index] = input_loader(0, channel_index);
    }
  }
  for (size_t step_index = 1; step_index < num_steps; ++step_index) {
    Span<float> channel_prefix_sum_data = tmp_b[step_index];
    Span<const float> channel_prev_prefix_sum_data = tmp_b[step_index - 1];
    for (size_t channel_index = 0; channel_index < num_channels;
         ++channel_index) {
      channel_prefix_sum_data[channel_index] =
          input_loader(step_index, channel_index) +
          channel_prev_prefix_sum_data[channel_index];
    }
  }

  // Populate tmp_a with windowed sums across the step axis using the prefix
  // sums in tmp_b.
  // 1: Copy the step_window first rows from tmp_b to tmp_a.
  std::memcpy(tmp_a.values.data(), tmp_b.values.data(),
              step_window * num_channels * sizeof(float));
  // 2: Compute windowed sums by subtracting prefix sums from each other.
  for (size_t step_index = step_window; step_index < num_steps; ++step_index) {
    Span<const float> curr_window_sum_data = tmp_b[step_index];
    Span<const float> prev_window_sum_data = tmp_b[step_index - step_window];
    Span<float> channel_window_sum_data = tmp_a[step_index];
    for (size_t channel_index = 0; channel_index < num_channels;
         ++channel_index) {
      channel_window_sum_data[channel_index] =
          curr_window_sum_data[channel_index] -
          prev_window_sum_data[channel_index];
    }
  }

  for (size_t step_index = 0; step_index < num_steps; ++step_index) {
    // Populate tmp_b with prefix sums across the channel axis of the windowed
    // sums across the step axis in tmp_a.
    {
      Span<const float> channel_window_sum_data = tmp_a[step_index];
      Span<float> step_prefix_sum_data = tmp_b[step_index];
      step_prefix_sum_data[0] = channel_window_sum_data[0];
      for (size_t channel_index = 1; channel_index < num_channels;
           ++channel_index) {
        step_prefix_sum_data[channel_index] =
            step_prefix_sum_data[channel_index - 1] +
            channel_window_sum_data[channel_index];
      }
    }
    // Populate tmp_a with windowed sums across steps-and-channels axes using
    // the "prefix sums across the channel axis and windowed sums across the
    // step axis" of tmp_b.
    {
      Span<const float> step_prefix_sum_data = tmp_b[step_index];
      Span<float> step_window_sum_data = tmp_a[step_index];
      std::memcpy(step_window_sum_data.data, step_prefix_sum_data.data,
                  channel_window * sizeof(float));
      for (size_t channel_index = channel_window; channel_index < num_channels;
           ++channel_index) {
        step_window_sum_data[channel_index] =
            step_prefix_sum_data[channel_index] -
            step_prefix_sum_data[channel_index - channel_window];
      }
    }
  }

  // Divide all windowed sums by step_window * channel_window to make them mean
  // values.
  const float reciprocal = 1.0 / (step_window * channel_window);
  for (size_t step_index = 0; step_index < num_steps; ++step_index) {
    Span<float> result_data = tmp_a[step_index];
    for (size_t channel_index = 0; channel_index < num_channels;
         ++channel_index) {
      result_data[channel_index] *= reciprocal;
    }
  }

  return tmp_a;
}

// Returns a slightly nonstandard version of the NSIM neural structural
// similarity metric between arrays a and b.
//
// step_window and channel_window are the number of time steps and channels
// in the array over which to window the mean, standard deviance, and
// covariance measures in NSIM.
//
// time_pairs is the dynamic time warp computed between spectrograms a and
// b, i.e. pairs of time step indices where a and b are considered to match
// each other in time.
//
// See https://doi.org/10.1016/j.specom.2011.09.004 for details.
float NSIM(const Spectrogram& a, const Spectrogram& b,
           const std::vector<std::pair<size_t, size_t>>& time_pairs,
           size_t step_window, size_t channel_window) {
  assert(a.num_dims == b.num_dims);
  const size_t num_channels = a.num_dims;
  const size_t num_steps = time_pairs.size();

  const Spectrogram mean_a =
      WindowMean(num_steps, num_channels, step_window, channel_window,
                 [&](size_t step_index, size_t channel_index) {
                   return a[time_pairs[step_index].first][channel_index];
                 });
  const Spectrogram mean_b =
      WindowMean(num_steps, num_channels, step_window, channel_window,
                 [&](size_t step_index, size_t channel_index) {
                   return b[time_pairs[step_index].second][channel_index];
                 });
  // NB: This computes (value - mean) using the mean computed for the window
  // at the same position as the value, so that each value gets a different mean
  // subtracted.
  const Spectrogram var_a = WindowMean(
      num_steps, num_channels, step_window, channel_window,
      [&](size_t step_index, size_t channel_index) {
        const float delta = a[time_pairs[step_index].first][channel_index] -
                            mean_a[step_index][channel_index];
        return delta * delta;
      });
  const Spectrogram var_b = WindowMean(
      num_steps, num_channels, step_window, channel_window,
      [&](size_t step_index, size_t channel_index) {
        const float delta = b[time_pairs[step_index].second][channel_index] -
                            mean_b[step_index][channel_index];
        return delta * delta;
      });
  const Spectrogram cov = WindowMean(
      num_steps, num_channels, step_window, channel_window,
      [&](size_t step_index, size_t channel_index) {
        const float delta_a = a[time_pairs[step_index].first][channel_index] -
                              mean_a[step_index][channel_index];
        const float delta_b = b[time_pairs[step_index].second][channel_index] -
                              mean_b[step_index][channel_index];
        return delta_a * delta_b;
      });

  // nsim-inspired ad hoc aggregation
  // main changes:
  // The aggregation tries to be more L1 than L2
  // Clamping of structure value
  // Adding a small amount of a-b L1 diff
  //
  // These changes were measured to be small improvements on a multi-corpus
  // test.
  const float C1 = 12.611504825401516;
  const float C3 = 8.296128127652123;
  const float C4 = 4.3913017618850251e-05;
  const float C5 = 5.2623574756945486e-07;
  const float C6 = 1.9650853180896431e-06;
  const float C7 = 0.00015815354596871763;
  const float C8 = 0.55786604844807497;

  float nsim_sum = 0.0;
  for (size_t step_index = 0; step_index < num_steps; ++step_index) {
    for (size_t channel_index = 0; channel_index < num_channels;
         ++channel_index) {
      const float mean_a_vec = mean_a[step_index][channel_index];
      const float mean_b_vec = mean_b[step_index][channel_index];
      const float std_a_vec = std::sqrt(var_a[step_index][channel_index]);
      const float std_b_vec = std::sqrt(var_b[step_index][channel_index]);
      const float cov_vec = cov[step_index][channel_index];
      const float intensity =
          (2 * std::sqrt(mean_a_vec * mean_b_vec) + C1) /
          (std::abs(mean_a_vec) + std::abs(mean_b_vec) + C1);
      const float structure_base =
          (cov_vec + C3) / (std_a_vec * std_b_vec + C3);
      const float structure_clamped = structure_base < C8 ? C8 : structure_base;
      const float structure =
          std::sqrt(std::sqrt(structure_clamped + C4) + C5) + C6;
      const float nsim = intensity * structure;
      const float aval = a[time_pairs[step_index].first][channel_index];
      const float bval = b[time_pairs[step_index].second][channel_index];
      const float diff = aval - bval;
      const float sqrdiff = C7 * std::abs(diff);
      const float nsim2 = nsim + sqrdiff;
      nsim_sum += nsim2;
    }
  }
  return std::clamp<float>(
      nsim_sum / static_cast<float>(num_steps * num_channels), 0.0, 1.0);
}

// A simple buffer of double cost values describing the time warp costs between
// two spectrograms.
struct CostMatrix {
  const double get(size_t step_a, size_t step_b) {
    return values[step_a * steps_b + step_b];
  }
  void set(size_t step_a, size_t step_b, double value) {
    values[step_a * steps_b + step_b] = value;
  }
  CostMatrix(size_t steps_a, size_t steps_b)
      : steps_a(steps_a),
        steps_b(steps_b),
        values(std::vector<double>(steps_a * steps_b,
                                   std::numeric_limits<double>::infinity())) {
    set(0, 0, 0);
  }
  size_t steps_a;
  size_t steps_b;
  std::vector<double> values;
};

// Computes the norm of the delta between two spectrograms at two given steps.
double delta_norm(const Spectrogram& a, const Spectrogram& b, size_t step_a,
                  size_t step_b) {
  Span<const float> dims_a = a[step_a];
  Span<const float> dims_b = b[step_b];
  assert(dims_a.size == dims_b.size);
  double result = 0;
  for (size_t index = 0; index < dims_a.size; index++) {
    float delta = dims_a[index] - dims_b[index];
    result += delta * delta;
  }
  return std::pow(result, 0.23289303544689094);
}

// Computes the DTW (https://en.wikipedia.org/wiki/Dynamic_time_warping)
// between two arrays.
std::vector<std::pair<size_t, size_t>> DTW(const Spectrogram& spec_a,
                                           const Spectrogram& spec_b) {
  // Sanity check that both spectrograms have the same number of feature
  // dimensions.
  assert(spec_a.num_dims == spec_b.num_dims);
  // Initialize a cost matrix with 0 at the start point, infinity at [*, 0] and
  // [0, *].
  CostMatrix cost_matrix(spec_a.num_steps, spec_b.num_steps);
  // Compute cost as cost as weighted sum of feature dimension norms to each
  // cell.
  static const double kMul00 = 0.98585952515276176;
  for (size_t spec_a_index = 1; spec_a_index < spec_a.num_steps;
       ++spec_a_index) {
    for (size_t spec_b_index = 1; spec_b_index < spec_b.num_steps;
         ++spec_b_index) {
      const double cost_at_index =
          delta_norm(spec_a, spec_b, spec_a_index, spec_b_index);
      const double sync_cost =
          cost_matrix.get(spec_a_index - 1, spec_b_index - 1);
      const double bwd_cost = cost_matrix.get(spec_a_index - 1, spec_b_index);
      const double fwd_cost = cost_matrix.get(spec_a_index, spec_b_index - 1);
      const double unsync_cost = std::min(bwd_cost, fwd_cost);
      const double costmin = std::min(sync_cost + kMul00 * cost_at_index,
                                      unsync_cost + cost_at_index);
      cost_matrix.set(spec_a_index, spec_b_index, costmin);
    }
  }

  // Track the cheapest path through the cost matrix.
  std::vector<std::pair<size_t, size_t>> result;
  std::pair<size_t, size_t> pos = {0, 0};
  result.push_back(pos);
  while (pos.first + 1 < spec_a.num_steps &&
         pos.second + 1 < spec_b.num_steps) {
    double min_cost = std::numeric_limits<double>::infinity();
    for (const auto& test_pos :
         {std::pair<size_t, size_t>{pos.first + 1, pos.second + 1},
          std::pair<size_t, size_t>{pos.first + 1, pos.second},
          std::pair<size_t, size_t>{pos.first, pos.second + 1}}) {
      double cost = cost_matrix.get(test_pos.first, test_pos.second);
      if (cost < min_cost) {
        min_cost = cost;
        pos = test_pos;
      }
    }
    result.push_back(pos);
  }
  return result;
}

// Expected signal sample rate.
constexpr float kSampleRate = 48000;

// Contains the energy in dB FS, and maximum absolute amplitude, of a signal.
struct EnergyAndMaxAbsAmplitude {
  float energy_db_fs;
  float max_abs_amplitude;
};

// Returns the energy and maximum absolute amplitude of a signal.
EnergyAndMaxAbsAmplitude Measure(Span<const float> signal) {
  float signal_max = 0;
  float signal_energy = 0;
  for (size_t index = 0; index < signal.size; ++index) {
    const float amplitude = signal.data[index];
    signal_energy += amplitude * amplitude;
    signal_max = std::max(signal_max, std::abs(amplitude));
  }
  return {.energy_db_fs =
              20 * std::log10(signal_energy / static_cast<float>(signal.size)),
          .max_abs_amplitude = signal_max};
}

// Normalizes the amplitude of the signal array to have the provided maximum
// absolute amplitude.
//
// Returns the energy in dB FS, and maximum absolute amplitude, of the result.
EnergyAndMaxAbsAmplitude NormalizeAmplitude(float max_abs_amplitude,
                                            Span<float> signal) {
  float signal_max = 0;
  float signal_energy = 0;
  for (size_t index = 0; index < signal.size; ++index) {
    signal_max = std::max(signal_max, std::abs(signal.data[index]));
  }
  const float scaling = max_abs_amplitude / signal_max;
  for (size_t index = 0; index < signal.size; ++index) {
    const float new_amplitude = scaling * signal.data[index];
    signal_energy += new_amplitude * new_amplitude;
    signal.data[index] = new_amplitude;
  }
  return {.energy_db_fs =
              20 * std::log10(signal_energy / static_cast<float>(signal.size)),
          .max_abs_amplitude = max_abs_amplitude};
}

// Contains parameters and code to compute perceptual spectrograms of sounds.
struct Zimtohrli {
  void Analyze(Span<const float> signal, Spectrogram& spectrogram) const {
    assert(spectrogram.num_dims == kNumRotators);
    Rotators rots;
    rots.FilterAndDownsample(signal.data, signal.size,
                             spectrogram.values.data(), spectrogram.num_steps,
                             spectrogram.num_dims,
                             signal.size / spectrogram.num_steps);
  }

  Spectrogram Analyze(Span<const float> signal) const {
    size_t num_steps =
        static_cast<size_t>(std::ceil(static_cast<float>(signal.size) *
                                      perceptual_sample_rate / kSampleRate));
    Spectrogram spec(num_steps, kNumRotators);
    Analyze(signal, spec);
    return spec;
  }

  float Distance(const Spectrogram& spectrogram_a,
                 const Spectrogram& spectrogram_b) const {
    assert(spectrogram_a.num_dims == spectrogram_b.num_dims);
    std::vector<std::pair<size_t, size_t>> time_pairs;
    time_pairs = DTW(spectrogram_a, spectrogram_b);
    return 1 - NSIM(spectrogram_a, spectrogram_b, time_pairs, nsim_step_window,
                    nsim_channel_window);
  }

  // The window in perceptual_sample_rate time steps when compting the NSIM.
  size_t nsim_step_window = 6;
  // The window in channels when computing the NSIM.
  size_t nsim_channel_window = 5;
  // The clock frequency of the brain?!
  float high_gamma_band = 84.0;
  int samples_per_perceptual_block = int(kSampleRate / high_gamma_band);
  // Sample rate corresponding to the human hearing sensitivity to timing
  // differences.
  float perceptual_sample_rate = kSampleRate / samples_per_perceptual_block;
  // The reference dB SPL of a sine signal of amplitude 1.
  float full_scale_sine_db = 78.3;
  float epsilon = 1e-9;
};

}  // namespace

}  // namespace zimtohrli

#endif  // CPP_ZIMT_ZIMTOHRLI_H_
