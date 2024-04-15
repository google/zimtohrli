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

#include "zimt/loudness.h"

#include <array>
#include <cstddef>

#include "absl/log/check.h"
#include "hwy/aligned_allocator.h"

// This file uses a lot of magic from the SIMD library Highway.
// In simplified terms, it will compile the code for multiple architectures
// using the "foreach_target.h" header file, and use the special namespace
// convention HWY_NAMESPACE to find the code to adapt to the SIMD functions,
// which are then called via HWY_DYNAMIC_DISPATCH. This leads to a lot of
// hard-to-explain Highway-related conventions being followed, like this here
// #define that makes this entire file be included by Highway in the process of
// building.
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "zimt/loudness.cc"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
// Must come after foreach_target.h to avoid redefinition errors.
#include "hwy/contrib/math/math-inl.h"
#include "hwy/highway.h"

// This is Highway magic conventions.
HWY_BEFORE_NAMESPACE();
namespace zimtohrli {
namespace HWY_NAMESPACE {

const hwy::HWY_NAMESPACE::ScalableTag<float> d;
using Vec = hwy::HWY_NAMESPACE::Vec<decltype(d)>;

Vec Sq(const Vec& v) { return Mul(v, v); }

// Reproduces the a_f computation from `loudness_parameter_computation.ipynb`,
// namely:
//
// params[0] - params[1] * np.log(params[2] * (x-params[3])) + 0.04 * params[4]
// * np.exp(-(0.0000001*params[5]*(x-14000*params[6])**2)) - 0.03 * params[7] *
// np.exp(-(0.0000001*params[8]*(x-5000*params[9])**2))
Vec Af(const hwy::AlignedNDArray<float, 2>& params_array, const Vec& hz) {
  const auto p = [&](size_t index) {
    return Load(d, params_array[{index}].data());
  };
  // params[0] - params[1] * np.log(params[2] * (x-params[3]))
  const Vec part0 = NegMulAdd(p(1), Log(d, Mul(Sub(hz, p(3)), p(2))), p(0));
  // 0.04 * params[4] * np.exp(-(0.0000001*params[5]*(x-14000*params[6])**2))
  const Vec part1 = Mul(Mul(Set(d, 0.04), p(4)),
                        Exp(d, Mul(Mul(Set(d, -0.0000001), p(5)),
                                   Sq(NegMulAdd(Set(d, 14000), p(6), hz)))));
  // 0.03 * params[7] * np.exp(-(0.0000001*params[8]*(x-5000*params[9])**2))
  const Vec part2 = Mul(Mul(Set(d, 0.03), p(7)),
                        Exp(d, Mul(Mul(Set(d, -0.0000001), p(8)),
                                   Sq(NegMulAdd(Set(d, 5000), p(9), hz)))));
  return Sub(Add(part0, part1), part2);
}

// Reproduces the L_U computation from `loudness_parameter_computation.ipynb`,
// namely:
//
// params[0] + params[1] * np.log(params[2] * (x - params[3])) - 5*params[4]*
// np.exp(-0.00001*params[5]*(x-1500*params[6])**2) + 5*params[7]*
// np.exp(-0.000001*params[8]*(x-3000*params[9])**2) -
// 15*params[10]*np.exp(-0.0000001*params[11]*(x-9000*params[12])**2) -
// 5*params[13]*np.exp(-0.00000001*params[14]*(x-12500*params[15])**2)
Vec Lu(const hwy::AlignedNDArray<float, 2>& params_array, const Vec& hz) {
  const auto p = [&](size_t index) {
    return Load(d, params_array[{index}].data());
  };
  // params[0] + params[1] * np.log(params[2] * (x - params[3]))
  const Vec part0 = MulAdd(p(1), Log(d, Mul(Sub(hz, p(3)), p(2))), p(0));
  // 5*params[4]*np.exp(-0.00001*params[5]*(x-1500*params[6])**2)
  const Vec part1 = Mul(Mul(Set(d, 5), p(4)),
                        Exp(d, Mul(Mul(Set(d, -0.00001), p(5)),
                                   Sq(NegMulAdd(Set(d, 1500), p(6), hz)))));
  // 5*params[7]*np.exp(-0.000001*params[8]*(x-3000*params[9])**2)
  const Vec part2 = Mul(Mul(Set(d, 5), p(7)),
                        Exp(d, Mul(Mul(Set(d, -0.000001), p(8)),
                                   Sq(NegMulAdd(Set(d, 3000), p(9), hz)))));
  // 15*params[10]*np.exp(-0.0000001*params[11]*(x-9000*params[12])**2)
  const Vec part3 = Mul(Mul(Set(d, 15), p(10)),
                        Exp(d, Mul(Mul(Set(d, -0.0000001), p(11)),
                                   Sq(NegMulAdd(Set(d, 9000), p(12), hz)))));
  // 5*params[13]*np.exp(-0.00000001*params[14]*(x-20000*params[15])**2)
  const Vec part4 = Mul(Mul(Set(d, 5), p(13)),
                        Exp(d, Mul(Mul(Set(d, -0.00000001), p(14)),
                                   Sq(NegMulAdd(Set(d, 12500), p(15), hz)))));
  return Sub(Sub(Add(Sub(part0, part1), part2), part3), part4);
}

// Reproduces the T_f computation from `loudness_parameter_computatation.ipynb`,
// namely:
//
// params[0] + params[1] * np.log(params[2] * (x - params[3])) + 5*params[4] *
// np.exp(-0.00001*params[5]*(x-1200*params[6])**2) -
// 10*params[7]*np.exp(-0.0000001*params[8]*(x-3300*params[9])**2) +
// 20*params[10]*np.exp(-0.00000001*params[11]*(x-12000*params[12])**2)
Vec Tf(const hwy::AlignedNDArray<float, 2>& params_array, const Vec& hz) {
  const auto p = [&](size_t index) {
    return Load(d, params_array[{index}].data());
  };
  // params[0] + params[1] * np.log(params[2] * (x - params[3]))
  const Vec part0 = MulAdd(p(1), Log(d, Mul(Sub(hz, p(3)), p(2))), p(0));
  // 5*params[4] * np.exp(-0.00001*params[5]*(x-1200*params[6])**2)
  const Vec part1 = Mul(Mul(Set(d, 5), p(4)),
                        Exp(d, Mul(Mul(Set(d, -0.00001), p(5)),
                                   Sq(NegMulAdd(Set(d, 1200), p(6), hz)))));
  // 10*params[7]*np.exp(-0.0000001*params[8]*(x-3300*params[9])**2)
  const Vec part2 = Mul(Mul(Set(d, 10), p(7)),
                        Exp(d, Mul(Mul(Set(d, -0.0000001), p(8)),
                                   Sq(NegMulAdd(Set(d, 3300), p(9), hz)))));
  // 20*params[10]*np.exp(-0.00000001*params[11]*(x-12000*params[12])**2)
  const Vec part3 = Mul(Mul(Set(d, 20), p(10)),
                        Exp(d, Mul(Mul(Set(d, -0.00000001), p(11)),
                                   Sq(NegMulAdd(Set(d, 12000), p(12), hz)))));
  return Add(Sub(Add(part0, part1), part2), part3);
}

template <size_t S>
hwy::AlignedNDArray<float, 2> ExpandLaneDimension(
    const std::array<float, S>& floats) {
  hwy::AlignedNDArray<float, 2> result({S, Lanes(d)});
  for (size_t value_index = 0; value_index < S; ++value_index) {
    for (size_t lane_index = 0; lane_index < Lanes(d); ++lane_index) {
      result[{value_index}][lane_index] = floats[value_index];
    }
  }
  return result;
}

// Defined here since Highway doesn't have a pow function.
Vec Pow(const Vec& base, const Vec& exponent) {
  return Exp(d, Mul(exponent, Log(d, base)));
}

// Reproduces the loudness from SPL computation in ISO 226.
void HwyPhonsFromSPL(const Loudness& l,
                     const hwy::AlignedNDArray<float, 2>& channels_db_spl,
                     const hwy::AlignedNDArray<float, 2>& thresholds_hz,
                     hwy::AlignedNDArray<float, 2>& channels_phons) {
  const Vec point_four = Set(d, 0.4);
  const Vec ten = Set(d, 10);
  const Vec point_one = Set(d, 0.1);
  const Vec nine = Set(d, 9);
  const Vec point_005135 = Set(d, 0.005135);
  const Vec forty = Set(d, 40);
  const Vec ninetyfour = Set(d, 94);
  const Vec zero = Zero(d);
  const size_t num_samples = channels_db_spl.shape()[0];
  const size_t num_channels = channels_db_spl.shape()[1];
  const hwy::AlignedNDArray<float, 2> a_f_params =
      ExpandLaneDimension(l.a_f_params);
  const hwy::AlignedNDArray<float, 2> l_u_params =
      ExpandLaneDimension(l.l_u_params);
  const hwy::AlignedNDArray<float, 2> t_f_params =
      ExpandLaneDimension(l.t_f_params);
  hwy::AlignedNDArray<float, 1> a_f_array({num_channels});
  hwy::AlignedNDArray<float, 1> l_u_array({num_channels});
  hwy::AlignedNDArray<float, 1> t_f_array({num_channels});
  for (size_t channel_index = 0; channel_index < num_channels;
       channel_index += Lanes(d)) {
    const Vec center_hz = Load(d, thresholds_hz[{1}].data() + channel_index);
    Store(Af(a_f_params, center_hz), d, a_f_array.data() + channel_index);
    Store(Lu(l_u_params, center_hz), d, l_u_array.data() + channel_index);
    Store(Tf(t_f_params, center_hz), d, t_f_array.data() + channel_index);
  }
  for (size_t sample_index = 0; sample_index < num_samples; ++sample_index) {
    for (size_t channel_index = 0; channel_index < num_channels;
         channel_index += Lanes(d)) {
      const Vec a_f = Load(d, a_f_array.data() + channel_index);
      const Vec l_u = Load(d, l_u_array.data() + channel_index);
      const Vec t_f = Load(d, t_f_array.data() + channel_index);
      const auto expf = [&](const Vec& x) HWY_ATTR {
        return Pow(
            Mul(point_four, Pow(ten, Sub(Mul(Add(x, l_u), point_one), nine))),
            a_f);
      };
      const Vec b_f =
          Add(Sub(expf(Load(d, channels_db_spl[{sample_index}].data() +
                                   channel_index)),
                  expf(t_f)),
              point_005135);
      const Vec center_hz = Load(d, thresholds_hz[{1}].data() + channel_index);
      // Compensating for 0 dB SPL at 0 Hz (since the arrays are padded with
      // zeros) not being properly handled by the parameterized phons-from-db
      // function.
      const Vec phon = IfThenElseZero(Gt(center_hz, zero),
                                      MulAdd(forty, Log10(d, b_f), ninetyfour));
      Store(phon, d, channels_phons[{sample_index}].data() + channel_index);
    }
  }
}

}  // namespace HWY_NAMESPACE
}  // namespace zimtohrli
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace zimtohrli {

HWY_EXPORT(HwyPhonsFromSPL);

void Loudness::PhonsFromSPL(
    const hwy::AlignedNDArray<float, 2>& channels_db_spl,
    const hwy::AlignedNDArray<float, 2>& thresholds_hz,
    hwy::AlignedNDArray<float, 2>& channels_phons) const {
  CHECK_EQ(channels_db_spl.shape()[0], channels_phons.shape()[0]);
  CHECK_EQ(channels_db_spl.shape()[1], channels_phons.shape()[1]);
  CHECK_EQ(thresholds_hz.shape()[1], channels_db_spl.shape()[1]);
  CHECK_EQ(thresholds_hz.shape()[0], 3);
  HWY_DYNAMIC_DISPATCH(HwyPhonsFromSPL)
  (*this, channels_db_spl, thresholds_hz, channels_phons);
}

}  // namespace zimtohrli

#endif  // HWY_ONCE
