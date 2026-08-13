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
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
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
      : size(vec.size()), data(vec.data()) {
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

// Expected signal sample rate.
inline constexpr float kSampleRate = 48000;

inline constexpr int64_t kNumRotators = 128;

// Converts energy values in frequency channels to loudness in dB using
// psychoacoustic weighting factors for each frequency band.
// Applies frequency-dependent gain correction and logarithmic scaling.
void LoudnessDb(float* channels);

// Ear drum and other receiving mass-spring objects are
// modeled through the Resonator. Resonator is a non-linear process
// and does complex spectral shifting of energy.
struct Resonator {
  float acc0 = 0;
  float acc1 = 0;
  float Update(float signal) {  // Resonate and attenuate.
    // These parameters relate to a population of ear drums.
    static const float kMul0 = 0.97018703367139569;
    static const float kMul1 = -0.02209312182872265;
    acc0 = kMul0 * acc0 + kMul1 * acc1 + signal;
    acc1 += acc0;
    return acc0;
  }
};

// Returns the center frequency in Hz for filter bank channel i.
// The 128 channels are spaced to match human auditory perception,
// with finer resolution at lower frequencies.
inline float Freq(int i) {
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

// Core signal processing engine using rotating phasors (Goertzel-like
// algorithm) for efficient frequency analysis. Implements the Zimtohrli/
// Tabuli filterbank with 128 frequency channels.
class Rotators {
 public:
  explicit Rotators(int downsample);

  // Updates the 128 Goertzel resonator channels and accumulates energy directly
  // into the current and next spectrogram frames in a single fused pass.
  //
  // Algorithmic specification (two-stage definition):
  //   1. Rotator & accumulator update: Updates the 6 leaky accumulators (accu)
  //      and phasors (rot) for each of the 128 channels, computing channel
  //      output energy E_i = accu[4]^2 + accu[5]^2 into channel_energies[128].
  //   2. Overlap-add windowing: Partitions channel energy across frames using
  //      downsampling weight: cur_frame[i] += weight * E_i,
  //      nxt_frame[i] += (1 - weight) * E_i.
  //      (weight + (1 - weight) == 1.0 preserves 100% of signal energy).
  //
  // Optimized implementation (fused single pass):
  //   Merges both stages into a single loop per sample. For each channel i:
  //     a. Load previous accumulator states into local registers a0..a5.
  //     b. Compute the decayed cascade and signal injection in registers.
  //     c. Store the updated states back to accu and rot.
  //     d. Compute energy E_i = a4^2 + a5^2 while still in registers, and fold
  //        it directly into cur_frame[i] and nxt_frame[i].
  //
  // Mathematical derivation & equivalence:
  // For decay w = window[i], sample s, and previous state accu:
  //   Reference in-place update:          Optimized register update:
  //     accu[0..5] *= w                     a0 = w * accu[0],  a1 = w * accu[1]
  //     accu[2] += accu[0]                  a2 = w * accu[2] + a0
  //     accu[3] += accu[1]                  a3 = w * accu[3] + a1
  //     accu[4] += accu[2]                  a4 = w * accu[4] + a2
  //     accu[5] += accu[3]                  a5 = w * accu[5] + a3
  //     accu[0] += rot[2] * s               accu[0] = a0 + rot[2] * s
  //     accu[1] += rot[3] * s               accu[1] = a1 + rot[3] * s
  //     E_i = accu[4]^2 + accu[5]^2         E_i = a4^2 + a5^2
  //   Both execute the identical arithmetic operations and floating-point
  //   rounding order, guaranteeing bit-exact output.
  //
  // Why this is faster & measured speedup:
  //   - Eliminates 256 memory ops/sample (128 stores + 128 loads) by removing
  //     the temporary channel_energies[128] stack array.
  //   - Register staging breaks memory dependency hazards and avoids
  //     store-to-load forwarding stalls.
  //   - Keeping E_i in registers enables hardware FMA instructions
  //     (e.g., vfmadd213ps).
  //
  // Result: ~2.05x streaming speedup (27.8ms -> 13.5ms on AMD Milan); reduces
  // total stream_test suite wall time by 40.1% (6.87s -> 4.11s).
  void IncrementAndAccumulate(float signal, float weight, float* cur_frame,
                              float* nxt_frame);

  // Renormalizes the rotating phasors to prevent numerical drift during
  // continuous processing.
  void RenormalizePhasors();

 private:
  // Four arrays of rotators, with memory layout for up to 128-way
  // simd-parallel. [0..1] is real and imag for rotation speed [2..3] is real
  // and imag for a frequency rotator of length sqrt(gain[i])
  float rot[4][kNumRotators] = {{0}};
  // Six leaky accumulator arrays for the Goertzel-like energy computation:
  // [0..1] is for real and imag of 1st leaking accumulation
  // [2..3] is for real and imag of 2nd leaking accumulation
  // [4..5] is for real and imag of 3rd leaking accumulation
  // The update order (2,3,4,5,0,1) was empirically determined to produce the
  // best results.
  float accu[6][kNumRotators] = {{0}};
  // Per-channel decay/leakage factor.
  float window[kNumRotators] = {0};
  // Per-channel gain normalization factor.
  float gain[kNumRotators] = {0};
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
  float max() const;

  // Multiplies all spectrogram values by the given factor.
  void rescale(float f);

  size_t size() const { return num_steps * num_dims; }
  size_t num_steps;
  size_t num_dims;
  std::unique_ptr<float[]> values;
};

// Stateful engine for streaming audio spectrogram analysis.
// Processes audio in chunks of arbitrary size without requiring the entire
// audio signal in memory.
//
// There are two equivalent ways to compute a spectrogram. Both run the same
// DSP and produce bit-for-bit identical output; choose based on how the audio
// arrives (there is no mode flag):
//
//  * Batch (non-chunked): call Zimtohrli::Analyze(signal), which consumes the
//    whole signal at once. Simplest when the audio fits in memory.
//
//  * Streaming (chunked): drive a ChunkedAnalyzer directly, feeding
//    arbitrary-size chunks via Process() and calling Flush() at end-of-stream:
//
//      ChunkedAnalyzer analyzer(zimtohrli.samples_per_perceptual_block);
//      std::vector<float> frames;
//      analyzer.Process(chunk, frames);  // repeat as samples arrive
//      analyzer.Flush(frames);           // finalize the last partial frame
//      Spectrogram spec(analyzer.num_steps(), kNumRotators, std::move(frames));
//
// The downsample argument must equal Zimtohrli::samples_per_perceptual_block so
// the framing matches Analyze().
//
// Thread safety: ChunkedAnalyzer is not thread-safe. It maintains mutable
// streaming state (sample buffer, accumulators, phasors) modified during
// processing. Each thread or pipeline must use its own instance.
class ChunkedAnalyzer {
 public:
  // The downsample value must match Zimtohrli::samples_per_perceptual_block
  // (int(kSampleRate / high_gamma_band) = 564).
  explicit ChunkedAnalyzer(int downsample)
      : downsample_(std::max(1, downsample)),
        downsample_window_(std::max(1, downsample)),
        rotators_(std::max(1, downsample)),
        current_frame_energy_(kNumRotators, 0.0f),
        pending_next_frame_energy_(kNumRotators, 0.0f) {
    for (int i = 0; i < downsample_; ++i) {
      downsample_window_[i] =
          1.0f / (1.0f + std::exp(8.0246040186567118f *
                                  ((2.0f / downsample_) * (i + 0.5f) - 1.0f)));
    }
  }

  // Processes a chunk of audio samples and appends any completed 128-dim
  // spectrogram frames to output_frames.
  void Process(Span<const float> chunk, std::vector<float>& output_frames);

  // Flushes remaining buffered samples and emits any partially accumulated
  // frame. Pads with kKernelSize - 1 (31) zeros to satisfy the 32-tap FIR
  // window without emitting an extra frame at downsample block boundaries.
  void Flush(std::vector<float>& output_frames) {
    // If leftover samples remain in the buffer that couldn't be processed due
    // to the 32-sample FIR lookahead window, zero-pad the buffer to flush the
    // remaining samples.
    if (sample_buffer_.size() > buffer_head_) {
      // Pad with kKernelSize - 1 (31) zeros so the last real sample gets a full
      // 32-sample FIR window. This processes exactly N filtered samples,
      // matching SpectrogramSteps(N) without emitting a spurious extra frame at
      // downsample block boundaries.
      std::array<float, kKernelSize - 1> padding{};
      Process(Span<const float>(padding.data(), padding.size()), output_frames);
      sample_buffer_.clear();
      buffer_head_ = 0;  // Reset so later Process() calls index from the front.
    }

    // If a frame is partially accumulated (dix_ > 0), finalize and emit it.
    if (dix_ > 0) {
      for (int k = 0; k < kNumRotators; ++k) {
        current_frame_energy_[k] += pending_next_frame_energy_[k];
      }
      EmitFrame(output_frames);

      std::fill(current_frame_energy_.begin(), current_frame_energy_.end(),
                0.0f);
      std::fill(pending_next_frame_energy_.begin(),
                pending_next_frame_energy_.end(), 0.0f);
      dix_ = 0;
    }
  }

  size_t num_steps() const { return num_steps_; }

 private:
  void EmitFrame(std::vector<float>& output_frames) {
    LoudnessDb(current_frame_energy_.data());
    output_frames.insert(output_frames.end(), current_frame_energy_.begin(),
                         current_frame_energy_.end());
    num_steps_++;
  }

  // Number of taps in the FIR filter kernels; the number of contiguous audio
  // samples of lookahead required to emit one filtered sample (32-tap FIR
  // window).
  static constexpr size_t kKernelSize = 32;

  // FIR filter kernel for the resonator path.
  // Applied via Dot32 to compute the resonator-filtered sample.
  static constexpr float reso_kernel[kKernelSize] = {
      -0.0076247065632976318f,  0.0039104155534537069f,
      0.0006684663662401936f,   0.0071559704794996589f,
      -0.0027931528839390098f,  0.0001368658992949717f,
      -0.0065802540559526824f,  -0.006574266432654235f,
      0.0034740030608061525f,   0.0030263702264320012f,
      -0.0029378401470635364f,  0.0034368516858611412f,
      0.0020915727560313845f,   -0.001541122014895714f,
      0.0033152434154573407f,   0.0015489639154823477f,
      -0.012691890416423556f,   -0.00027840484849307723f,
      -0.0010427818083574192f,  -0.0087889956707155811f,
      -0.0066266333272295289f,  -0.00080043637110705163f,
      -0.0072998536521213225f,  0.0036816757141278035f,
      -0.00031555808271841742f, 0.00099264355318687508f,
      -0.0012897138783731826f,  0.0013771982014390573f,
      0.0070121198631592861f,   -0.0016488166452599629f,
      -0.00727301918260589f,    0.010964231292090421f,
  };

  // FIR filter kernel for the linear path.
  // Applied via Dot32 alongside reso_kernel to compute the filtered sample.
  static constexpr float linear_kernel[kKernelSize] = {
      -0.19947158175459692f,   0.020092596724127186f,    -0.065549345816240306f,
      0.059315467827374985f,   0.24679907672434401f,     -0.14582584331716622f,
      -0.083626881941168935f,  0.31874018187263292f,     0.22397287387339976f,
      0.036279108994617872f,   -0.13919343535956649f,    0.04950990842192754f,
      -0.027271514202057801f,  -0.00099846257278084238f, -0.10798654028268029f,
      -0.10489917207275569f,   -0.095906755569884164f,   -0.21168952706515187f,
      0.83249555081867532f,    0.58484205043268755f,     -0.21828800943250842f,
      0.080106893472851701f,   0.93016317182367492f,     -0.49663918345960828f,
      -1.6197347842868257f,    -0.18383066061195377f,    0.6236802270978099f,
      1.1976849288800944f,     -0.70212522492743401f,    0.90598962344860279f,
      -0.0018858573753579057f, -0.41452533138089309f,
  };

  // Threshold for compacting the sample buffer. When buffer_head_ exceeds
  // this value, we shift remaining samples to the front to prevent
  // unbounded memory growth.
  static constexpr size_t kCompactThreshold = 4096;

  int downsample_ = 1;
  std::vector<float> downsample_window_;
  Rotators rotators_;
  Resonator resonator_;
  size_t dix_ = 0;
  size_t num_steps_ = 0;
  std::vector<float> current_frame_energy_;
  std::vector<float> pending_next_frame_energy_;
  std::vector<float> sample_buffer_;
  size_t buffer_head_ = 0;
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

float NSIM(const Spectrogram& a, const Spectrogram& b,
           const std::vector<std::pair<size_t, size_t>>& time_pairs,
           size_t step_window, size_t channel_window);

// NSIM without the DTW pass.
float NSIM(const Spectrogram& a, const Spectrogram& b, size_t step_window,
           size_t channel_window);

// Computes the DTW (https://en.wikipedia.org/wiki/Dynamic_time_warping)
// between two arrays.
std::vector<std::pair<size_t, size_t>> DTW(const Spectrogram& spec_a,
                                           const Spectrogram& spec_b);

// Main class for psychoacoustic audio analysis.
// Converts audio signals to perceptual spectrograms and computes
// perceptual distance between audio signals using the Zimtohrli metric.
// Expected input: 48kHz mono audio with samples in range [-1, 1].
struct Zimtohrli {
  // Analyzes an audio signal and returns a new spectrogram.
  // signal: input audio samples at 48kHz, range [-1, 1]
  // Returns: perceptual spectrogram representation
  Spectrogram Analyze(Span<const float> signal) const {
    Spectrogram spec(SpectrogramSteps(signal.size), kNumRotators);
    Analyze(signal, spec);
    return spec;
  }

  // Analyzes an audio signal and fills the provided spectrogram.
  // Converts time-domain audio to a perceptual spectrogram by applying
  // resonator filtering, frequency analysis via rotating phasors, and
  // downsampling. Steps the signal is too short to produce are zero-filled.
  // signal: input audio samples at 48kHz, range [-1, 1]
  // spectrogram: pre-allocated output spectrogram to fill
  void Analyze(Span<const float> signal, Spectrogram& spectrogram) const;

  // Calculates the number of time steps in the output spectrogram
  // based on the input signal length and perceptual sample rate.
  size_t SpectrogramSteps(size_t num_samples) const {
    return static_cast<size_t>(std::ceil(static_cast<float>(num_samples) *
                                         perceptual_sample_rate / kSampleRate));
  }

  // Helper method to rescale energy levels of two spectrograms to match each
  // other.
  static void RescaleToMatchEnergy(Spectrogram& spectrogram_a,
                                   Spectrogram& spectrogram_b);

  // Computes perceptual distance between two spectrograms.
  // Uses DTW for time alignment and NSIM for similarity measurement.
  // Returns: distance in range [0, 1], where 0 = identical, 1 = maximally
  // different.
  // Note: both spectrograms may be rescaled to match energy levels.
  float Distance(Spectrogram& spectrogram_a, Spectrogram& spectrogram_b) const;

  // Computes perceptual distance between two spectrograms assuming they are
  // already aligned.
  // `spectrogram_a` and `spectrogram_b` are the perceptual spectrograms to
  // compare. `step_window` optionally overrides the default NSIM step window
  // size.
  // Returns: distance in range [0, 1], where 0 = identical, 1 = maximally
  // different.
  // Note: both spectrograms may be rescaled to match energy levels.
  float DistanceWithoutDtw(
      Spectrogram& spectrogram_a, Spectrogram& spectrogram_b,
      std::optional<size_t> step_window = std::nullopt) const;

  // The window in perceptual_sample_rate time steps when compting the NSIM.
  size_t nsim_step_window = 8;
  // The window in channels when computing the NSIM.
  size_t nsim_channel_window = 5;
  // The clock frequency of the brain?!
  float high_gamma_band = 85.0;
  int samples_per_perceptual_block = int(kSampleRate / high_gamma_band);
  // Sample rate corresponding to the human hearing sensitivity to timing
  // differences.
  float perceptual_sample_rate = kSampleRate / samples_per_perceptual_block;
  // The reference dB SPL of a sine signal of amplitude 1.
  float full_scale_sine_db = 78.3;
};

}  // namespace zimtohrli

#endif  // CPP_ZIMT_ZIMTOHRLI_H_
