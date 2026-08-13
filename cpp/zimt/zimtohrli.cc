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

#include "zimt/zimtohrli.h"

#include <algorithm>
#include <array>
#define _USE_MATH_DEFINES
#include <cmath>
#include <math.h>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "zimt/zimtohrli.cc"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"
#include "hwy/contrib/math/math-inl.h"
#include "hwy/contrib/algo/transform-inl.h"

#ifndef assert_eq
#define assert_eq(a, b)                                                    \
  do {                                                                     \
    if ((a) != (b)) {                                                      \
      std::cerr << "Assertion failed: " << #a << " (" << std::to_string(a) \
                << ") == " << #b << " (" << std::to_string(b) << ") at "   \
                << __FILE__ << ":" << std::to_string(__LINE__) << "\n";    \
      std::abort();                                                        \
    }                                                                      \
  } while (0)
#endif

namespace zimtohrli {

#ifndef ZIMTOHRLI_ALIGNMENT_DEFINED
#define ZIMTOHRLI_ALIGNMENT_DEFINED
namespace {
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

}  // namespace
#endif

namespace HWY_NAMESPACE {
namespace {

namespace hn = hwy::HWY_NAMESPACE;

// Computes dot product of two 32-element float arrays.
HWY_ATTR float Dot32(const float* a, const float* b) {
  HWY_CAPPED(float, 32) d;
  using V = hn::Vec<decltype(d)>;
  V sum = hn::Zero(d);
  for (int i = 0; i < 32; i += Lanes(d)) {
    const V va = hn::LoadU(d, &a[i]);
    const V vb = hn::LoadU(d, &b[i]);
    sum = hn::MulAdd(va, vb, sum);
  }
  return hn::ReduceSum(d, sum);
}

// Computes the perceptual distance between two spectrogram frames.
// Uses p norm with psychoacoustic weighting.
// Used by DTW to compute frame-to-frame alignment costs.
HWY_ATTR double delta_norm(const Spectrogram& a, const Spectrogram& b,
                           size_t step_a, size_t step_b) {
  Span<const float> dims_a = a[step_a];
  Span<const float> dims_b = b[step_b];
  assert_eq(dims_a.size, dims_b.size);
  HWY_FULL(double) d;
  hn::Repartition<float, decltype(d)> df;
  using V = hn::Vec<decltype(d)>;
  using Vf = hn::Vec<decltype(df)>;
  V result1 = hn::Zero(d);
  V result2 = hn::Zero(d);
  size_t index;
  for (index = 0; index + Lanes(df) <= dims_a.size; index += Lanes(df)) {
    const Vf va = hn::LoadU(df, &dims_a[index]);
    const Vf vb = hn::LoadU(df, &dims_b[index]);
    const Vf delta = hn::Sub(va, vb);
    const V delta1 = hn::PromoteLowerTo(d, delta);
    const V delta2 = hn::PromoteUpperTo(d, delta);
    result1 = hn::MulAdd(delta1, delta1, result1);
    result2 = hn::MulAdd(delta2, delta2, result2);
  }
  if (HWY_LIKELY(index != dims_a.size)) {
    size_t remaining = dims_a.size - index;
    const Vf va = hn::LoadNOr(hn::Zero(df), df, &dims_a[index], remaining);
    const Vf vb = hn::LoadNOr(hn::Zero(df), df, &dims_b[index], remaining);
    const Vf delta = hn::Sub(va, vb);
    const V delta1 = hn::PromoteLowerTo(d, delta);
    const V delta2 = hn::PromoteUpperTo(d, delta);
    result1 = hn::MulAdd(delta1, delta1, result1);
    result2 = hn::MulAdd(delta2, delta2, result2);
  }
  static const float pp = 0.32264042946823823;
  return std::pow(hn::ReduceSum(d, hn::Add(result1, result2)), pp);
}

HWY_ATTR void IncrementAndAccumulateRotators(float signal_value,
                                             float weight_value,
                                             float* cur_frame, float* nxt_frame,
                                             const float window[kNumRotators],
                                             float accu[6][kNumRotators],
                                             float rot[4][kNumRotators]) {
  HWY_CAPPED(float, kNumRotators) d;
  using V = hn::Vec<decltype(d)>;
  const V inv_weight = hn::Set(d, 1.0f - weight_value);
  const V signal = hn::Set(d, signal_value);
  const V weight = hn::Set(d, weight_value);
#if HWY_HAVE_CONSTEXPR_LANES
  static_assert(kNumRotators % Lanes(d) == 0);
#else
  assert_eq(kNumRotators % Lanes(d), 0);
#endif
  for (int i = 0; i < kNumRotators; i += Lanes(d)) {
    const V w = hn::LoadU(d, &window[i]);
    const V a0 = hn::Mul(hn::LoadU(d, &accu[0][i]), w);
    const V a1 = hn::Mul(hn::LoadU(d, &accu[1][i]), w);
    const V a2 = hn::MulAdd(hn::LoadU(d, &accu[2][i]), w, a0);
    const V a3 = hn::MulAdd(hn::LoadU(d, &accu[3][i]), w, a1);
    const V a4 = hn::MulAdd(hn::LoadU(d, &accu[4][i]), w, a2);
    const V a5 = hn::MulAdd(hn::LoadU(d, &accu[5][i]), w, a3);
    const V r2 = hn::LoadU(d, &rot[2][i]);
    const V r3 = hn::LoadU(d, &rot[3][i]);
    hn::StoreU(hn::MulAdd(r2, signal, a0), d, &accu[0][i]);
    hn::StoreU(hn::MulAdd(r3, signal, a1), d, &accu[1][i]);
    hn::StoreU(a2, d, &accu[2][i]);
    hn::StoreU(a3, d, &accu[3][i]);
    hn::StoreU(a4, d, &accu[4][i]);
    hn::StoreU(a5, d, &accu[5][i]);
    const V r0 = hn::LoadU(d, &rot[0][i]);
    const V r1 = hn::LoadU(d, &rot[1][i]);
    hn::StoreU(hn::MulSub(r0, r2, hn::Mul(r1, r3)), d, &rot[2][i]);
    hn::StoreU(hn::MulAdd(r0, r3, hn::Mul(r1, r2)), d, &rot[3][i]);
    const V energy = hn::MulAdd(a4, a4, hn::Mul(a5, a5));
    hn::StoreU(hn::MulAdd(weight, energy, hn::LoadU(d, &cur_frame[i])), d,
               &cur_frame[i]);
    hn::StoreU(hn::MulAdd(inv_weight, energy, hn::LoadU(d, &nxt_frame[i])), d,
               &nxt_frame[i]);
  }
}

HWY_ATTR void RenormalizePhasors(const float gain[kNumRotators],
                                 float rot[4][kNumRotators]) {
  HWY_CAPPED(float, kNumRotators) d;
  using V = hn::Vec<decltype(d)>;
  for (int i = 0; i < kNumRotators; i += Lanes(d)) {
    const V r2 = hn::LoadU(d, &rot[2][i]);
    const V r3 = hn::LoadU(d, &rot[3][i]);
    const V norm = hn::Div(hn::LoadU(d, &gain[i]),
                           hn::Sqrt(hn::MulAdd(r2, r2, hn::Mul(r3, r3))));
    hn::StoreU(hn::Mul(r2, norm), d, &rot[2][i]);
    hn::StoreU(hn::Mul(r3, norm), d, &rot[3][i]);
  }
}

HWY_ATTR float SpectrogramMax(const float* values, const size_t count) {
  HWY_FULL(float) d;
  using V = hn::Vec<decltype(d)>;
  V max = hn::Zero(d);
  hn::Foreach(d, values, count, hn::Zero(d), [&max](auto d, auto vec) HWY_ATTR {
    max = hn::Max(max, hn::Abs(vec));
  });
  return hn::ReduceMax(d, max);
}

HWY_ATTR void RescaleSpectrogram(float* values, const size_t count,
                                 const float factor) {
  HWY_FULL(float) d;
  using V = hn::Vec<decltype(d)>;
  V mul = hn::Set(d, factor);
  hn::Transform(d, values, count, [&mul](auto d, auto vec) HWY_ATTR {
    return hn::Mul(vec, mul);
  });
}

// Performs the NSIM similarity computation using the specified alignment.
// - If `alignment` is `PreAligned`, we assume the signals are already perfectly
//   aligned and compare them directly frame-by-frame. In this case, both
//   spectrograms must have the exact same step length.
// - If `alignment` is `const std::vector<...>*`, it represents the warp path
//   calculated by DTW, mapping frames between a and b.
HWY_ATTR float NSIMImpl(const Spectrogram& a, const Spectrogram& b,
                        Alignment alignment, size_t step_window,
                        size_t channel_window) {
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

  HWY_FULL(double) d;
  using V = hn::Vec<decltype(d)>;
  hn::Repartition<float, decltype(d)> df;
  using Vf = hn::Vec<decltype(df)>;

  // nsim-inspired ad hoc aggregation
  // main changes:
  // The aggregation tries to be more L1 than L2
  // Clamping of structure value
  //
  // These changes were measured to be small improvements on a multi-corpus
  // test.
  const Vf C1 = hn::Set(df, 26.426389124321354);
  const Vf C3 = hn::Set(df, 1.9522719384622791);
  const Vf C8 = hn::Set(df, 0.6325126087671703);
  const Vf P0 = hn::Set(df, 1.0500187278772866);
  const Vf P1 = hn::Set(df, 0.25808223975919764);

  double nsim_sum = 0.0;
  for (size_t step_index = 0; step_index < num_steps; ++step_index) {
    V nsim_accu1 = hn::Zero(d);
    V nsim_accu2 = hn::Zero(d);
    size_t channel_index = 0;
    for (; channel_index + Lanes(df) <= num_channels;
         channel_index += Lanes(df)) {
      const Vf mean_a_vec = hn::LoadU(df, &mean_a[step_index][channel_index]);
      const Vf mean_b_vec = hn::LoadU(df, &mean_b[step_index][channel_index]);
      const Vf std_a_vec =
          hn::Sqrt(hn::LoadU(df, &var_a[step_index][channel_index]));
      const Vf std_b_vec =
          hn::Sqrt(hn::LoadU(df, &var_b[step_index][channel_index]));
      const Vf cov_vec = hn::LoadU(df, &cov[step_index][channel_index]);
      const Vf intensity = hn::Pow(
          df,
          hn::Div(
              hn::MulAdd(hn::Set(df, 2.f),
                         hn::Sqrt(hn::Mul(mean_a_vec, mean_b_vec)), C1),
              hn::Add(hn::Add(hn::Abs(mean_a_vec), hn::Abs(mean_b_vec)), C1)),
          P0);
      const Vf structure_base =
          hn::Div(hn::Add(cov_vec, C3), hn::MulAdd(std_a_vec, std_b_vec, C3));
      const Vf structure_clamped =
          hn::IfThenElse(hn::Lt(structure_base, C8), C8, structure_base);
      const Vf structure = hn::Pow(df, structure_clamped, P1);

      const V intensity1 = hn::PromoteLowerTo(d, intensity);
      const V structure1 = hn::PromoteLowerTo(d, structure);
      nsim_accu1 = hn::MulAdd(intensity1, structure1, nsim_accu1);
      const V intensity2 = hn::PromoteUpperTo(d, intensity);
      const V structure2 = hn::PromoteUpperTo(d, structure);
      nsim_accu2 = hn::MulAdd(intensity2, structure2, nsim_accu2);
    }
    if (HWY_LIKELY(channel_index != num_channels)) {
      const size_t remaining = num_channels - channel_index;
      const Vf mean_a_vec = hn::LoadNOr(
          hn::Zero(df), df, &mean_a[step_index][channel_index], remaining);
      const Vf mean_b_vec = hn::LoadNOr(
          hn::Zero(df), df, &mean_b[step_index][channel_index], remaining);
      const Vf std_a_vec = hn::Sqrt(hn::LoadNOr(
          hn::Zero(df), df, &var_a[step_index][channel_index], remaining));
      const Vf std_b_vec = hn::Sqrt(hn::LoadNOr(
          hn::Zero(df), df, &var_b[step_index][channel_index], remaining));
      const Vf cov_vec = hn::LoadNOr(
          hn::Zero(df), df, &cov[step_index][channel_index], remaining);
      Vf intensity = hn::Pow(
          df,
          hn::Div(
              hn::MulAdd(hn::Set(df, 2.f),
                         hn::Sqrt(hn::Mul(mean_a_vec, mean_b_vec)), C1),
              hn::Add(hn::Add(hn::Abs(mean_a_vec), hn::Abs(mean_b_vec)), C1)),
          P0);
      const Vf structure_base =
          hn::Div(hn::Add(cov_vec, C3), hn::MulAdd(std_a_vec, std_b_vec, C3));
      const Vf structure_clamped =
          hn::IfThenElse(hn::Lt(structure_base, C8), C8, structure_base);
      const Vf structure = hn::Pow(df, structure_clamped, P1);

      intensity =
          hn::IfThenElse(hn::FirstN(df, remaining), intensity, hn::Zero(df));

      const V intensity1 = hn::PromoteLowerTo(d, intensity);
      const V structure1 = hn::PromoteLowerTo(d, structure);
      nsim_accu1 = hn::MulAdd(intensity1, structure1, nsim_accu1);
      const V intensity2 = hn::PromoteUpperTo(d, intensity);
      const V structure2 = hn::PromoteUpperTo(d, structure);
      nsim_accu2 = hn::MulAdd(intensity2, structure2, nsim_accu2);
    }
    nsim_sum += hn::ReduceSum(d, hn::Add(nsim_accu1, nsim_accu2));
  }
  return std::clamp<float>(
      nsim_sum / static_cast<float>(num_steps * num_channels), 0.0, 1.0);
}

}  // namespace
}  // namespace HWY_NAMESPACE

#if HWY_ONCE

namespace {

HWY_EXPORT(Dot32);
HWY_EXPORT(delta_norm);
HWY_EXPORT(IncrementAndAccumulateRotators);
HWY_EXPORT(RenormalizePhasors);
HWY_EXPORT(SpectrogramMax);
HWY_EXPORT(RescaleSpectrogram);
HWY_EXPORT(NSIMImpl);

float Dot32(const float* a, const float* b) {
  return HWY_DYNAMIC_DISPATCH(Dot32)(a, b);
}

// Calculates the effective bandwidth in Hz for filter bank channel i.
// Uses geometric mean spacing between adjacent channels.
double CalculateBandwidthInHz(int i) {
  return std::sqrt(Freq(i + 1) * Freq(i)) - std::sqrt(Freq(i - 1) * Freq(i));
}

}  // namespace

Rotators::Rotators(int downsample) {
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

void Rotators::IncrementAndAccumulate(float signal, float weight,
                                      float* cur_frame, float* nxt_frame) {
  return HWY_DYNAMIC_DISPATCH(IncrementAndAccumulateRotators)(
      signal, weight, cur_frame, nxt_frame, window, accu, rot);
}

void Rotators::RenormalizePhasors() {
  HWY_DYNAMIC_DISPATCH(RenormalizePhasors)(gain, rot);
}

float Spectrogram::max() const {
  return HWY_DYNAMIC_DISPATCH(SpectrogramMax)(values.get(),
                                              num_steps * num_dims);
}

void Spectrogram::rescale(float f) {
  HWY_DYNAMIC_DISPATCH(RescaleSpectrogram)(values.get(), num_steps * num_dims,
                                           f);
}

void ChunkedAnalyzer::Process(Span<const float> chunk,
                              std::vector<float>& output_frames) {
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

  // The FIR kernels need kKernelSize (32) contiguous samples per output, so
  // process only while a full 32-sample window is resident in sample_buffer_.
  // The + kKernelSize in the bound keeps Dot32 from reading past the end.
  while (buffer_head_ + processed + kKernelSize <= sample_buffer_.size()) {
    const float* in = &sample_buffer_[buffer_head_ + processed];
    const float weight = downsample_window_[dix_];

    // 1. Evaluate FIR filtering (resonator kernel + linear kernel).
    // 2. Feed output to the 2nd-order IIR Resonator.
    // 3. Update 128 rotating phasors/accumulators and fold each channel's
    //    energy directly into the current and next spectrogram frames via
    //    overlap-add in one fused pass (see IncrementAndAccumulate), avoiding
    //    a per-sample channel_energies[] stack array.
    rotators_.IncrementAndAccumulate(
        resonator_.Update(Dot32(in, &reso_kernel[0])) +
            Dot32(in, &linear_kernel[0]),
        weight, current_frame_energy_.data(),
        pending_next_frame_energy_.data());

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
      std::memmove(sample_buffer_.data(), sample_buffer_.data() + buffer_head_,
                   remaining * sizeof(float));
    }
    sample_buffer_.resize(remaining);
    buffer_head_ = 0;
  }
}

float NSIM(const Spectrogram& a, const Spectrogram& b,
           const std::vector<std::pair<size_t, size_t>>& time_pairs,
           size_t step_window, size_t channel_window) {
  return HWY_DYNAMIC_DISPATCH(NSIMImpl)(a, b, &time_pairs, step_window,
                                        channel_window);
}

float NSIM(const Spectrogram& a, const Spectrogram& b, size_t step_window,
           size_t channel_window) {
  return HWY_DYNAMIC_DISPATCH(NSIMImpl)(a, b, PreAligned{}, step_window,
                                        channel_window);
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
      const double cost_at_index = HWY_DYNAMIC_DISPATCH(delta_norm)(
          spec_a, spec_b, spec_a_index, spec_b_index);
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

void Zimtohrli::Analyze(Span<const float> signal,
                        Spectrogram& spectrogram) const {
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

void Zimtohrli::RescaleToMatchEnergy(Spectrogram& spectrogram_a,
                                     Spectrogram& spectrogram_b) {
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

float Zimtohrli::Distance(Spectrogram& spectrogram_a,
                          Spectrogram& spectrogram_b) const {
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

float Zimtohrli::DistanceWithoutDtw(Spectrogram& spectrogram_a,
                                    Spectrogram& spectrogram_b,
                                    std::optional<size_t> step_window) const {
  assert_eq(spectrogram_a.num_dims, spectrogram_b.num_dims);
  assert_eq(spectrogram_a.num_steps, spectrogram_b.num_steps);
  // Note: Empty spectrograms (num_steps == 0) are handled by NSIM, which
  // returns 0.0 for empty inputs, yielding distance = 1.0f.
  // No explicit guard needed here.

  RescaleToMatchEnergy(spectrogram_a, spectrogram_b);

  size_t sw = step_window.value_or(nsim_step_window);
  return 1 - NSIM(spectrogram_a, spectrogram_b, sw, nsim_channel_window);
}

void LoudnessDb(float* channels) {
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
      -427.1872751241109,  -370.2893289163535,  -357.01506023770378,
      -301.28879097655118, -216.78500670398833, -168.07806679629724,
      -168.71805754864141, -159.53956835871321, -268.72445005379404,
      -311.16419962879075, -277.03504398276948, -288.39213525341091,
      -305.32237068568082, -258.6335011904703,  -254.78634459132866,
      -181.46038594163568, -93.950223670617163, -88.818104801961908,
      -26.156023442931389, -38.752447643769138, -47.906764099227942,
      -21.676071849485375, 10.884646488419072,  21.595865980708961,
      -52.559415237056015, -57.62886752507012,  -80.132855392693315,
      -84.248190048411175, -87.193989053900296, -134.86546270102167,
      -146.23587896776439, -211.30970199319108,
  };
  float noise = kBaseNoise;
  for (int k = 0; k < kNumRotators; ++k) {
    channels[k] = log(channels[k] + noise) * kMul[k];
    noise += kBaseNoiseSlope[k >> 2];
  }
}

#endif

}  // namespace zimtohrli
