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
#define _USE_MATH_DEFINES
#include <cmath>
#include <math.h>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <memory>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <variant>
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
      0.69111, 0.68478, 0.68763, 0.68845, 0.68595, 0.68576, 0.68883, 0.68932,
      0.68713, 0.69239, 0.68762, 0.68928, 0.68449, 0.69143, 0.69494, 0.69796,
      0.69697, 0.70122, 0.72878, 0.79911, 0.85713, 0.88063, 0.88563, 0.87561,
      0.81948, 0.70435, 0.63479, 0.58382, 0.52065, 0.48390, 0.46452, 0.47952,
      0.52686, 0.63677, 0.75972, 0.89449, 0.97411, 1.01874, 1.01105, 0.99306,
      0.93613, 0.92825, 0.93149, 0.98687, 1.05782, 1.16461, 1.25028, 1.30768,
      1.31484, 1.28574, 1.23002, 1.15336, 1.08800, 1.01472, 0.94610, 0.91856,
      0.87797, 0.85825, 0.82836, 0.82198, 0.81394, 0.82724, 0.84235, 0.86009,
      0.88276, 0.89349, 0.92543, 0.94822, 0.98526, 0.99730, 1.00532, 1.02506,
      1.03689, 1.04897, 1.05307, 1.05817, 1.05174, 1.04766, 1.03553, 1.03437,
      1.03238, 1.05164, 1.08115, 1.13753, 1.21037, 1.31175, 1.44154, 1.52549,
      1.60840, 1.67304, 1.71593, 1.72853, 1.76630, 1.70865, 1.68923, 1.65506,
      1.57241, 1.51275, 1.37840, 1.28644, 1.23809, 1.21714, 1.30432, 1.30430,
      1.33396, 1.34255, 1.33987, 1.35309, 1.35169, 1.35219, 1.35385, 1.35851,
      1.34995, 1.20201, 1.17218, 1.19284, 1.23571, 1.34281, 1.16209, 0.89999,
      0.89264, 1.08696, 0.78787, 0.78445, 1.12917, 0.65317, 1.02086, 1.11196,
  };
  static const float kBaseNoise = 766068.03396368888;
  static const float kBaseNoiseSlope[32] = {
    -427.1872751241109, -370.2893289163535, -357.01506023770378, -301.28879097655118,
    -216.78500670398833, -168.07806679629724, -168.71805754864141, -159.53956835871321,
    -268.72445005379404, -311.16419962879075, -277.03504398276948, -288.39213525341091,
    -305.32237068568082, -258.6335011904703, -254.78634459132866, -181.46038594163568,
    -93.950223670617163, -88.818104801961908, -26.156023442931389, -38.752447643769138,
    -47.906764099227942, -21.676071849485375, 10.884646488419072, 21.595865980708961,
    -52.559415237056015, -57.62886752507012, -80.132855392693315, -84.248190048411175,
    -87.193989053900296, -134.86546270102167, -146.23587896776439, -211.30970199319108,
  };
  float noise = kBaseNoise;
  for (int k = 0; k < kNumRotators; ++k) {
    channels[k] = log(channels[k] + noise) * kMul[k];
    noise += kBaseNoiseSlope[k >> 2];
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
    static const float kMul0 = 0.97018703367139569;
    static const float kMul1 = -0.02209312182872265;
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

// Core signal processing engine using rotating phasors (Goertzel-like
// algorithm) for efficient frequency analysis. Implements the Zimtohrli/
// Tabuli filterbank with 128 frequency channels.
class Rotators {
 public:
  explicit Rotators(int downsample) {
    downsample = std::max(1, downsample);
    static const float kSampleRate = 48000.0f;
    static const float kHzToRad = 2.0f * M_PI / kSampleRate;
    static const double kWindow = 0.9996073584827937;
    static const double kBandwidthMagic = 0.73227703638356523;
    // A big value for normalization. Ideally 1.0, but this works better
    // for an unknown reason even if the base noise level is adapted similarly.
    static const double kScale = 931912404783.44507;
    const float gainer = std::sqrt(kScale / downsample);
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
  }

  // Updates all rotators and accumulators with a new signal sample,
  // and computes the channel output energies (accu[4]^2 + accu[5]^2).
  void IncrementAll(float signal, float* channel_energies) {
    for (int i = 0; i < kNumRotators; i++) {
      const float w = window[i];
      for (int k = 0; k < 6; ++k) accu[k][i] *= w;
      accu[2][i] += accu[0][i];
      accu[3][i] += accu[1][i];
      accu[4][i] += accu[2][i];
      accu[5][i] += accu[3][i];
      accu[0][i] += rot[2][i] * signal;
      accu[1][i] += rot[3][i] * signal;
      const float a = rot[2][i], b = rot[3][i];
      rot[2][i] = rot[0][i] * a - rot[1][i] * b;
      rot[3][i] = rot[0][i] * b + rot[1][i] * a;
      channel_energies[i] = accu[4][i] * accu[4][i] + accu[5][i] * accu[5][i];
    }
  }

  // Renormalizes the rotating phasors to prevent numerical drift during
  // continuous processing.
  void RenormalizePhasors() {
    for (int i = 0; i < kNumRotators; ++i) {
      float norm =
          gain[i] / std::sqrt(rot[2][i] * rot[2][i] + rot[3][i] * rot[3][i]);
      rot[2][i] *= norm;
      rot[3][i] *= norm;
    }
  }

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
  void Process(Span<const float> chunk, std::vector<float>& output_frames) {
    // sample_buffer_ is consumed via a read offset (buffer_head_) with periodic
    // compaction, giving amortized O(1) advancement instead of erasing from the
    // front on every call. Samples that don't yet complete a 32-sample FIR
    // window are kept across calls, so any chunk size (even a single sample) is
    // handled correctly; small chunks just defer processing until 32 samples
    // have accumulated.
    if (chunk.size > 0) {
      sample_buffer_.insert(sample_buffer_.end(), chunk.data,
                            chunk.data + chunk.size);
    }

    size_t processed = 0;
    float channel_energies[kNumRotators];

    // The FIR kernels need kKernelSize (32) contiguous samples per output, so
    // process only while a full 32-sample window is resident in sample_buffer_.
    // The + kKernelSize in the bound keeps Dot32 from reading past the end.
    while (buffer_head_ + processed + kKernelSize <= sample_buffer_.size()) {
      const float* in = &sample_buffer_[buffer_head_ + processed];
      const float weight = downsample_window_[dix_];

      // 1. Evaluate FIR filtering (resonator kernel + linear kernel).
      // 2. Feed output to the 2nd-order IIR Resonator.
      // 3. Update 128 rotating phasors and leaky accumulators in Rotators.
      rotators_.IncrementAll(resonator_.Update(Dot32(in, &reso_kernel[0])) +
                                 Dot32(in, &linear_kernel[0]),
                             channel_energies);

      // Overlap-add downsampling windowing:
      // The sigmoid downsampling window weight smoothly partitions the sample's
      // energy between the current time step (weight) and the subsequent time
      // step (1 - weight). Because weight + (1 - weight) == 1.0, 100% of the
      // signal's perceptual energy is preserved without boundary artifacts
      // across chunks.
      for (int k = 0; k < kNumRotators; ++k) {
        current_frame_energy_[k] += weight * channel_energies[k];
        pending_next_frame_energy_[k] += (1.0f - weight) * channel_energies[k];
      }

      processed++;
      dix_++;

      // When the downsampling block is full, finalize the spectrogram frame.
      if (dix_ == downsample_) {
        EmitFrame(output_frames);

        // Advance to next frame: the overlap energy that spilled into the next
        // frame becomes the new current frame. Swap (instead of copy) reuses
        // the just-emitted buffer as the next pending buffer, avoiding
        // reallocation.
        std::swap(current_frame_energy_, pending_next_frame_energy_);
        std::fill(pending_next_frame_energy_.begin(),
                  pending_next_frame_energy_.end(), 0.0f);
        dix_ = 0;
        rotators_.RenormalizePhasors();
      }
    }

    // Advance the read offset past processed samples in O(1).
    buffer_head_ += processed;

    // Periodic compaction: shift remaining samples to the front when read
    // offset exceeds kCompactThreshold (4096). Amortized O(1).
    if (buffer_head_ > kCompactThreshold) {
      size_t remaining = sample_buffer_.size() - buffer_head_;
      if (remaining > 0) {
        std::memmove(sample_buffer_.data(),
                     sample_buffer_.data() + buffer_head_,
                     remaining * sizeof(float));
      }
      sample_buffer_.resize(remaining);
      buffer_head_ = 0;
    }
  }

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
// Sentinel type indicating that the input spectrograms are already perfectly
// aligned in time, allowing direct frame-by-frame comparison.
struct PreAligned {};

// Represents how the two spectrograms are aligned in time:
// - `PreAligned`: The signals are already perfectly aligned (direct
// comparison).
// - `const std::vector<...>*`: A pointer to the warp path computed by DTW
//   that maps matching frame indices between a and b.
using Alignment =
    std::variant<PreAligned, const std::vector<std::pair<size_t, size_t>>*>;

// Performs the NSIM similarity computation using the specified alignment.
// - If `alignment` is `PreAligned`, we assume the signals are already perfectly
//   aligned and compare them directly frame-by-frame. In this case, both
//   spectrograms must have the exact same step length.
// - If `alignment` is `const std::vector<...>*`, it represents the warp path
//   calculated by DTW, mapping frames between a and b.
float NSIMImpl(const Spectrogram& a, const Spectrogram& b, Alignment alignment,
               size_t step_window, size_t channel_window) {
  assert_eq(a.num_dims, b.num_dims);

  const bool is_pre_aligned = std::holds_alternative<PreAligned>(alignment);
  const auto* time_pairs =
      is_pre_aligned
          ? nullptr
          : std::get<const std::vector<std::pair<size_t, size_t>>*>(alignment);

  if (is_pre_aligned) {
    assert_eq(a.num_steps, b.num_steps);
  }
  const size_t num_channels = a.num_dims;
  // The total comparison steps over which we compute similarity.
  // For aligned runs, this is a.num_steps (matching b.num_steps).
  // For DTW runs, this is the length of the warp path.
  const size_t num_steps = is_pre_aligned ? a.num_steps : time_pairs->size();

  if (num_steps == 0 || num_channels == 0 || step_window == 0 ||
      channel_window == 0) {
    return 0.0f;
  }

  step_window = std::min(step_window, num_steps);
  channel_window = std::min(channel_window, num_channels);

  auto step_a = [&](size_t step) {
    return is_pre_aligned ? step : (*time_pairs)[step].first;
  };
  auto step_b = [&](size_t step) {
    return is_pre_aligned ? step : (*time_pairs)[step].second;
  };

  const Spectrogram mean_a =
      WindowMean(num_steps, num_channels, step_window, channel_window,
                 [&](size_t step_index, size_t channel_index) {
                   return a[step_a(step_index)][channel_index];
                 });
  const Spectrogram mean_b =
      WindowMean(num_steps, num_channels, step_window, channel_window,
                 [&](size_t step_index, size_t channel_index) {
                   return b[step_b(step_index)][channel_index];
                 });
  // NB: This computes (value - mean) using the mean computed for the window
  // at the same position as the value, so that each value gets a different mean
  // subtracted.
  const Spectrogram var_a =
      WindowMean(num_steps, num_channels, step_window, channel_window,
                 [&](size_t step_index, size_t channel_index) {
                   const float delta = a[step_a(step_index)][channel_index] -
                                       mean_a[step_index][channel_index];
                   return delta * delta;
                 });
  const Spectrogram var_b =
      WindowMean(num_steps, num_channels, step_window, channel_window,
                 [&](size_t step_index, size_t channel_index) {
                   const float delta = b[step_b(step_index)][channel_index] -
                                       mean_b[step_index][channel_index];
                   return delta * delta;
                 });
  const Spectrogram cov =
      WindowMean(num_steps, num_channels, step_window, channel_window,
                 [&](size_t step_index, size_t channel_index) {
                   const float delta_a = a[step_a(step_index)][channel_index] -
                                         mean_a[step_index][channel_index];
                   const float delta_b = b[step_b(step_index)][channel_index] -
                                         mean_b[step_index][channel_index];
                   return delta_a * delta_b;
                 });

  // nsim-inspired ad hoc aggregation
  // main changes:
  // The aggregation tries to be more L1 than L2
  // Clamping of structure value
  //
  // These changes were measured to be small improvements on a multi-corpus
  // test.
  static const float C1 = 26.426389124321354;
  static const float C3 = 1.9522719384622791;
  static const float C8 = 0.6325126087671703;
  static const float P0 = 1.0500187278772866;
  static const float P1 = 0.25808223975919764;

  double nsim_sum = 0.0;
  for (size_t step_index = 0; step_index < num_steps; ++step_index) {
    double nsim_accu = 0.0;
    for (size_t channel_index = 0; channel_index < num_channels;
         ++channel_index) {
      const float mean_a_vec = mean_a[step_index][channel_index];
      const float mean_b_vec = mean_b[step_index][channel_index];
      const float std_a_vec = std::sqrt(var_a[step_index][channel_index]);
      const float std_b_vec = std::sqrt(var_b[step_index][channel_index]);
      const float cov_vec = cov[step_index][channel_index];
      const float intensity =
          pow((2 * std::sqrt(mean_a_vec * mean_b_vec) + C1) /
                  (std::abs(mean_a_vec) + std::abs(mean_b_vec) + C1),
              P0);
      const float structure_base =
          (cov_vec + C3) / (std_a_vec * std_b_vec + C3);
      const float structure_clamped = structure_base < C8 ? C8 : structure_base;
      const float structure = std::pow(structure_clamped, P1);
      const float nsim = intensity * structure;
      nsim_accu += nsim;
    }
    nsim_sum += nsim_accu;
  }
  return std::clamp<float>(
      nsim_sum / static_cast<float>(num_steps * num_channels), 0.0, 1.0);
}

float NSIM(const Spectrogram& a, const Spectrogram& b,
           const std::vector<std::pair<size_t, size_t>>& time_pairs,
           size_t step_window, size_t channel_window) {
  return NSIMImpl(a, b, &time_pairs, step_window, channel_window);
}

// NSIM without the DTW pass.
float NSIM(const Spectrogram& a, const Spectrogram& b, size_t step_window,
           size_t channel_window) {
  return NSIMImpl(a, b, PreAligned{}, step_window, channel_window);
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
// Uses p norm with psychoacoustic weighting.
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
  static const float pp = 0.32264042946823823;
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
  // kMul00 value below 1.0 reduces the cost of going in sync, advancing
  // a and b traversal separately is a distance of 1. Purely geometrically
  // sqrt(2) might be a good value, but this works better for an unknown
  // reason (favoring a and b traversing together).
  static const double kMul00 = 0.90394786214451761;

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
  void Analyze(Span<const float> signal, Spectrogram& spectrogram) const {
    assert_eq(spectrogram.num_dims, kNumRotators);
    ChunkedAnalyzer analyzer(samples_per_perceptual_block);
    std::vector<float> frames;
    analyzer.Process(signal, frames);
    analyzer.Flush(frames);
    for (size_t step = 0; step < spectrogram.num_steps; ++step) {
      for (size_t k = 0; k < spectrogram.num_dims; ++k) {
        spectrogram[step][k] = step < analyzer.num_steps()
                                   ? frames[step * spectrogram.num_dims + k]
                                   : 0.0f;
      }
    }
  }

  // Calculates the number of time steps in the output spectrogram
  // based on the input signal length and perceptual sample rate.
  size_t SpectrogramSteps(size_t num_samples) const {
    return static_cast<size_t>(std::ceil(static_cast<float>(num_samples) *
                                         perceptual_sample_rate / kSampleRate));
  }

  // Helper method to rescale energy levels of two spectrograms to match each
  // other.
  void RescaleToMatchEnergy(Spectrogram& spectrogram_a,
                            Spectrogram& spectrogram_b) const {
    assert_eq(spectrogram_a.num_dims, spectrogram_b.num_dims);
    const double max_a = spectrogram_a.max();
    const double max_b = spectrogram_b.max();
    if (max_a != max_b && max_a > 0.0 && max_b > 0.0) {
      // For full correction cora + corb would be 1.0.
      // It is very much unclear why optimization prefers
      // to have overcorrection for distance. Perhaps it
      // softens the error vallay and in combination with the
      // preference of going straight in the path-finding good
      // things happens. (This is pure speculation without trying
      // to obtain evidence about this).
      float cora = 0.5828284197882053;
      float corb = 0.6310239126768997;
      if (max_a > max_b) {
	std::swap(cora, corb);
      }
      spectrogram_b.rescale(pow(max_a / max_b, cora));
      spectrogram_a.rescale(pow(max_b / max_a, corb));
    }
  }

  // Computes perceptual distance between two spectrograms.
  // Uses DTW for time alignment and NSIM for similarity measurement.
  // Returns: distance in range [0, 1], where 0 = identical, 1 = maximally
  // different.
  // Note: both spectrograms may be rescaled to match energy levels.
  float Distance(Spectrogram& spectrogram_a, Spectrogram& spectrogram_b) const {
    assert_eq(spectrogram_a.num_dims, spectrogram_b.num_dims);
    if (spectrogram_a.num_steps == 0 || spectrogram_b.num_steps == 0) {
      return 1.0f;
    }
    RescaleToMatchEnergy(spectrogram_a, spectrogram_b);
    std::vector<std::pair<size_t, size_t>> time_pairs;
    time_pairs = DTW(spectrogram_a, spectrogram_b);
    return 1 - NSIM(spectrogram_a, spectrogram_b, time_pairs, nsim_step_window,
                    nsim_channel_window);
  }

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
      std::optional<size_t> step_window = std::nullopt) const {
    assert_eq(spectrogram_a.num_dims, spectrogram_b.num_dims);
    assert_eq(spectrogram_a.num_steps, spectrogram_b.num_steps);
    // Note: Empty spectrograms (num_steps == 0) are handled by NSIM, which
    // returns 0.0 for empty inputs, yielding distance = 1.0f.
    // No explicit guard needed here.

    RescaleToMatchEnergy(spectrogram_a, spectrogram_b);

    size_t sw = step_window.value_or(nsim_step_window);
    return 1 - NSIM(spectrogram_a, spectrogram_b, sw, nsim_channel_window);
  }

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

}  // namespace

}  // namespace zimtohrli

#endif  // CPP_ZIMT_ZIMTOHRLI_H_
