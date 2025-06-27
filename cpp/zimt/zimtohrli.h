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

#ifndef CPP_ZIMT_ZIMTOHRLI_H_
#define CPP_ZIMT_ZIMTOHRLI_H_

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <memory>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace zimtohrli {

// Lightweight non-owning view of a contiguous array.
// Similar to std::span but available pre-C++20.
template <typename T>
struct Span {
  Span(const Span& other) = default;
  Span(std::vector<T>& vec) : size(vec.size()), data(vec.data()) {}
  Span(T* data, size_t size) : size(size), data(data) {}
  template <typename U>
  Span(const std::vector<U>& vec) noexcept
      : data(vec.data()), size(vec.size()) {
    static_assert(std::is_convertible_v<U(*)[], T(*)[]>,
                  "Cannot construct Span from vector of incompatible type.");
  }
  template <typename U>
  Span(const Span<U>& other) noexcept : size(other.size), data(other.data) {
    static_assert(std::is_convertible_v<U(*)[], T(*)[]>,
                  "Cannot construct Span from Span of incompatible type.");
  }
  Span& operator=(const Span& other) = default;
  const T& operator[](size_t index) const { return data[index]; }
  T& operator[](size_t index) { return data[index]; }
  size_t size;
  T* data;
};

namespace {

// Expected signal sample rate.
constexpr float kSampleRate = 48000;

#define assert_eq(a, b)                                                        \
  do {                                                                         \
    if ((a) != (b)) {                                                          \
      std::cerr << "Assertion failed: " << #a << " (" << std::to_string(a)     \
                << ") == " << #b << " (" << std::to_string(b) << ") at "       \
                << __FILE__ << ":" << std::to_string(__LINE__) << "\n";        \
      std::abort();                                                            \
    }                                                                          \
  } while (0)

constexpr int64_t kNumRotators = 128;

// Converts energy values in frequency channels to loudness in dB using
// psychoacoustic weighting factors for each frequency band.
// Applies frequency-dependent gain correction and logarithmic scaling.
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
      1.30207, 1.26835, 1.25100, 1.24664, 1.24041, 1.24297, 1.07569, 0.97131,
      0.95906, 1.21035, 0.85762, 0.77298, 1.12289, 0.74092, 0.99662, 1.11603,
  };
  static const float kBaseNoise = 886018.44434708043;
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

// Computes dot product of two 32-element float arrays.
// Optimized for SIMD vectorization with -ffast-math.
inline float Dot32(const float* a, const float* b) {
  // -ffast-math is helpful here, and clang can simdify this.
  float sum = 0;
  for (int i = 0; i < 32; ++i) sum += a[i] * b[i];
  return sum;
}

// Returns the center frequency in Hz for filter bank channel i.
// The 128 channels are spaced to match human auditory perception,
// with finer resolution at lower frequencies.
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

// Calculates the effective bandwidth in Hz for filter bank channel i.
// Uses geometric mean spacing between adjacent channels.
double CalculateBandwidthInHz(int i) {
  return std::sqrt(Freq(i + 1) * Freq(i)) - std::sqrt(Freq(i - 1) * Freq(i));
}

// Core signal processing engine using rotating phasors (Goertzel-like algorithm)
// for efficient frequency analysis. Implements the Tabuli filterbank.
class Rotators {
 private:
  // Four arrays of rotators, with memory layout for up to 128-way
  // simd-parallel. [0..1] is real and imag for rotation speed [2..3] is real
  // and image for a frequency rotator of length sqrt(gain[i])
  float rot[4][kNumRotators];
  // [0..1] is for real and imag of 1st leaking accumulation
  // [2..3] is for real and imag of 2nd leaking accumulation
  // [4..5] is for real and imag of 3rd leaking accumulation
  float accu[6][kNumRotators] = {{0}};
  float window[kNumRotators];
  float gain[kNumRotators];

  // Renormalizes the rotating phasors to prevent numerical drift.
  // Called periodically during signal processing.
  void OccasionallyRenormalize() {
    for (int i = 0; i < kNumRotators; ++i) {
      float norm =
          gain[i] / sqrt(rot[2][i] * rot[2][i] + rot[3][i] * rot[3][i]);
      rot[2][i] *= norm;
      rot[3][i] *= norm;
    }
  }
  // Updates all rotators and accumulators with a new signal sample.
  // Applies windowing, rotates phasors, and accumulates energy.
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
  // Main signal processing function that converts time-domain audio to a
  // perceptual spectrogram. Applies resonator filtering, frequency analysis
  // via rotating phasors, and downsampling.
  // in: input audio samples
  // in_size: number of input samples
  // out: output spectrogram buffer
  // out_shape0: number of time steps in output
  // out_stride: stride between time steps in output buffer
  // downsample: downsampling factor
  void FilterAndDownsample(const float* in, size_t in_size, float* out,
                           size_t out_shape0, size_t out_stride,
                           int downsample) {
    static const float kHzToRad = 2.0f * M_PI / kSampleRate;
    static const double kWindow = 0.9996028710680265;
    static const double kBandwidthMagic = 0.7328516996032982;
    // A big value for normalization. Ideally 1.0, but this works better.
    static const double kScale = 929900594411.23657;
    const float gainer = sqrt(kScale / downsample);
    for (int i = 0; i < kNumRotators; ++i) {
      float bandwidth = CalculateBandwidthInHz(i);  // bandwidth per bucket.
      window[i] = std::pow(kWindow, bandwidth * kBandwidthMagic);
      float windowM1 = 1.0f - window[i];
      const float f = Freq(i) * kHzToRad;
      gain[i] = gainer * (windowM1 * windowM1 * windowM1) * Freq(i) / bandwidth;
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
    std::vector<float> downsample_window(downsample);
    for (int i = 0; i < downsample; ++i) {
      downsample_window[i] =
          1.0 / (1.0 + exp(7.9446 * ((2.0 / downsample) * (i + 0.5) - 1)));
    }
    Resonator resonator;
    size_t out_ix = 0;
    constexpr size_t kKernelSize = 32;
    static const float reso_kernel[kKernelSize] = {
      -0.0075642284403770708, 0.0041328270786934662, -7.6269851290751061e-06, 0.0061764514689768733,
      -0.0028376753880472038, -1.1759452250705732e-05, -0.0065499115361845562, -0.0069727090984949783,
      0.0034584201864033401, 0.003329316161974918, -0.0029971240720728575, 0.0034898641766847685,
      0.0017717742743446263, -0.0015229487607625498, 0.0039309982613565655, 0.001278227701047937,
      -0.0116877416785343, -0.00039070521292690666, -0.0015923522740827827, -0.0082269584153230185,
      -0.0063814620315990021, -0.0008796390298788419, -0.0071855544224704287, 0.0034822736952680863,
      -0.00041538926556568181, 0.0001753900488857857, -0.0011326124605282573, 0.00095353008231245965,
      0.0073567454219722467, -0.0016601446765057634, -0.0069136302438569507, 0.010715105623693549,
    };
    static const float linear_kernel[kKernelSize] = {
      -0.30960591444509439, -0.079455203026254709, -0.14108618014504098, 0.070751037303552131,
      0.14104891038659864, -0.17036477880916376, 0.014288229833457814, 0.27147357420390988,
      0.17978692186268302, 0.065653189749429991, 0.014169704877201516, 0.18257259370291729,
      0.0021021318985668257, 0.065359875882277235, -0.015544998395038102, -0.049398120278478827,
      -0.064034911106614606, -0.57876116795333099, 0.57561220696398696, 0.40135227167310927,
      -0.33118848897270026, 0.17695279679195522, 1.0491938729586434, -0.58835602045486513,
      -1.4541325309560014, 0.071462019783188307, 0.72056751090553661, 1.2265425406909325,
      -0.72083484154250099, 0.84200784192262634, -0.10112736611558046, -0.44049413285605787,
    };
    for (size_t in_ix = 0, dix = 0; in_ix + kKernelSize < in_size; ++in_ix) {
      const float weight = downsample_window[dix];
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
        values(std::make_unique<float[]>(num_steps * kNumRotators)) {}
  Spectrogram(size_t num_steps, size_t num_dims)
      : num_steps(num_steps),
        num_dims(num_dims),
        values(std::make_unique<float[]>(num_steps * num_dims)) {}
  Spectrogram(size_t num_steps, size_t num_dims,
              std::unique_ptr<float[]> values)
      : num_steps(num_steps), num_dims(num_dims), values(std::move(values)) {}
  Spectrogram(size_t num_steps, size_t num_dims, std::vector<float> data)
      : num_steps(num_steps),
        num_dims(num_dims),
        values(std::make_unique<float[]>(data.size())) {
    std::memcpy(values.get(), data.data(), data.size() * sizeof(float));
  }
  Spectrogram(size_t num_steps, size_t num_dims, float* data)
      : num_steps(num_steps), num_dims(num_dims), values(data) {}
  Spectrogram& operator=(Spectrogram&& other) = default;
  Span<const float> operator[](size_t n) const {
    return Span<const float>(values.get() + n * num_dims, num_dims);
  }
  Span<float> operator[](size_t n) {
    return Span<float>(values.get() + n * num_dims, num_dims);
  }
  // Returns the maximum absolute value across all spectrogram values.
  float max() const {
    float res = 0;
    for (size_t step_idx = 0; step_idx < num_steps; ++step_idx) {
      for (size_t dim_idx = 0; dim_idx < num_dims; ++dim_idx) {
        res = std::max(res, std::abs(operator[](step_idx)[dim_idx]));
      }
    }
    return res;
  }
  // Multiplies all spectrogram values by the given factor.
  void rescale(float f) {
    for (size_t step_idx = 0; step_idx < num_steps; ++step_idx) {
      for (size_t dim_idx = 0; dim_idx < num_dims; ++dim_idx) {
        operator[](step_idx)[dim_idx] *= f;
      }
    }
  }
  size_t size() const { return num_steps * num_dims; }
  size_t num_steps;
  size_t num_dims;
  std::unique_ptr<float[]> values;
};

// Computes windowed mean values over a 2D spectrogram using efficient
// prefix sum computation. Used by NSIM to compute local statistics.
// num_steps: number of time steps
// num_channels: number of frequency channels
// step_window: window size in time dimension
// channel_window: window size in frequency dimension
// input_loader: function(step, channel) that loads input values
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
  std::memcpy(tmp_a.values.get(), tmp_b.values.get(),
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
  assert_eq(a.num_dims, b.num_dims);
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
  static const float C1 = 28.341082593304403;
  static const float C3 = 1.6705576583956854;
  static const float C4 = 5.5778917823818053e-05;
  static const float C5 = 2.5568733818058373e-07;
  static const float C6 = 3.510912492638396e-08;
  static const float C7 = 2.4720299934548813e-07;
  static const float C8 = 0.54045365472095119;
  static const float P0 = 0.84013864788155035;
  static const float P1 = 1.7336006370531516;
  static const float P2 = 0.19488365206961764;

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
	pow((2 * std::sqrt(mean_a_vec * mean_b_vec) + C1) /
	    (std::abs(mean_a_vec) + std::abs(mean_b_vec) + C1), P0);
      const float structure_base =
          (cov_vec + C3) / (std_a_vec * std_b_vec + C3);
      const float structure_clamped = structure_base < C8 ? C8 : structure_base;
      const float structure =
	std::pow(std::pow(structure_clamped + C4, P1) + C5, P2) + C6;
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
  double get(size_t step_a, size_t step_b) const {
    return values[step_a * steps_b + step_b];
  }
  void set(size_t step_a, size_t step_b, double value) {
    values[step_a * steps_b + step_b] = value;
  }
  CostMatrix(size_t steps_a, size_t steps_b)
      : steps_a(steps_a),
        steps_b(steps_b),
        values(std::vector<double>(steps_a * steps_b,
                                   std::numeric_limits<double>::max())) {
    set(0, 0, 0);
  }
  size_t steps_a;
  size_t steps_b;
  std::vector<double> values;
};

// Computes the perceptual distance between two spectrogram frames.
// Uses L2 norm with psychoacoustic weighting (power 0.233).
// Used by DTW to compute frame-to-frame alignment costs.
double delta_norm(const Spectrogram& a, const Spectrogram& b, size_t step_a,
                  size_t step_b) {
  Span<const float> dims_a = a[step_a];
  Span<const float> dims_b = b[step_b];
  assert_eq(dims_a.size, dims_b.size);
  double result = 0;
  for (size_t index = 0; index < dims_a.size; index++) {
    float delta = dims_a[index] - dims_b[index];
    result += delta * delta;
  }
  static const float pp = 0.35491343190704761;
  return std::pow(result, pp);
}

// Computes the DTW (https://en.wikipedia.org/wiki/Dynamic_time_warping)
// between two arrays.
std::vector<std::pair<size_t, size_t>> DTW(const Spectrogram& spec_a,
                                           const Spectrogram& spec_b) {
  // Sanity check that both spectrograms have the same number of feature
  // dimensions.
  assert_eq(spec_a.num_dims, spec_b.num_dims);
  CostMatrix cost_matrix(spec_a.num_steps, spec_b.num_steps);
  // Compute cost as cost as weighted sum of feature dimension norms to each
  // cell.
  static const double kMul00 = 0.97775949394431627;
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
    double min_cost = std::numeric_limits<double>::max();
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

// Main class for psychoacoustic audio analysis.
// Converts audio signals to perceptual spectrograms and computes
// perceptual distance between audio signals using the Zimtohrli metric.
// Expected input: 48kHz mono audio with samples in range [-1, 1].
struct Zimtohrli {
  // Analyzes an audio signal and fills the provided spectrogram.
  // signal: input audio samples at 48kHz, range [-1, 1]
  // spectrogram: pre-allocated output spectrogram to fill
  void Analyze(Span<const float> signal, Spectrogram& spectrogram) const {
    assert_eq(spectrogram.num_dims, kNumRotators);
    Rotators rots;
    rots.FilterAndDownsample(signal.data, signal.size, spectrogram.values.get(),
                             spectrogram.num_steps, spectrogram.num_dims,
                             signal.size / spectrogram.num_steps);
  }

  // Analyzes an audio signal and returns a new spectrogram.
  // signal: input audio samples at 48kHz, range [-1, 1]
  // Returns: perceptual spectrogram representation
  Spectrogram Analyze(Span<const float> signal) const {
    Spectrogram spec(SpectrogramSteps(signal.size), kNumRotators);
    Analyze(signal, spec);
    return spec;
  }

  // Calculates the number of time steps in the output spectrogram
  // based on the input signal length and perceptual sample rate.
  size_t SpectrogramSteps(size_t num_samples) const {
    return static_cast<size_t>(std::ceil(static_cast<float>(num_samples) *
                                         perceptual_sample_rate / kSampleRate));
  }

  // Computes perceptual distance between two spectrograms.
  // Uses DTW for time alignment and NSIM for similarity measurement.
  // Returns: distance in range [0, 1], where 0 = identical, 1 = maximally different
  // Note: both spectrograms may be rescaled to match energy levels
  float Distance(Spectrogram& spectrogram_a,
                 Spectrogram& spectrogram_b) const {
    assert_eq(spectrogram_a.num_dims, spectrogram_b.num_dims);
    const double max_a = spectrogram_a.max();
    const double max_b = spectrogram_b.max();
    if (max_a != max_b) {
      float cora = 0.48234235170721046;
      float corb = 0.43404193485438936;
      if (max_a > max_b) {
	std::swap(cora, corb);
      }
      spectrogram_b.rescale(pow(max_a / max_b, cora));
      spectrogram_a.rescale(pow(max_b / max_a, corb));
    }
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
};

}  // namespace

}  // namespace zimtohrli

#endif  // CPP_ZIMT_ZIMTOHRLI_H_
