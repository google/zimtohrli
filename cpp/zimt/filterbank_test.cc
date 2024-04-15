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

#include "zimt/filterbank.h"

#include <cmath>
#include <cstddef>
#include <cstring>
#include <vector>

#include "absl/log/check.h"
#include "benchmark/benchmark.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "hwy/aligned_allocator.h"
#include "hwy/base.h"
#include "zimt/cam.h"
#include "zimt/elliptic.h"

namespace zimtohrli {

namespace {

void Copy(hwy::Span<float> destination, hwy::Span<const float> source) {
  std::memcpy(destination.data(), source.data(), source.size() * sizeof(float));
}

TEST(Filterbank, SingleSectionFilterTest) {
  // Golden data produced by:
  //
  // scipy.signal.lfilter([0.5, 0.5], [1.0, 0.5], [1, 0, 0, 0])
  //
  // and
  //
  // scipy.signal.lfilter([0.5, -0.5], [1.0, 0.5], [1, 0, 0, 0]))
  hwy::AlignedNDArray<float, 1> sig({4});
  sig[{}][0] = 1.0;

  const std::vector<std::vector<BACoeffs>> coeffs = {
      {{.b_coeffs = {0.5, 0.5}, .a_coeffs = {1.0, 0.5}}},
      {{.b_coeffs = {0.5, -0.5}, .a_coeffs = {1.0, 0.5}}}};

  hwy::AlignedNDArray<float, 2> want_filtered_sig({4, 2});
  Copy(want_filtered_sig[{0}], {0.5, 0.5});
  Copy(want_filtered_sig[{1}], {0.25, -0.75});
  Copy(want_filtered_sig[{2}], {-0.125, 0.375});
  Copy(want_filtered_sig[{3}], {0.0625, -0.1875});

  hwy::AlignedNDArray<float, 2> got_filtered_sig(
      {sig.shape()[0], coeffs.size()});

  Filterbank filter(coeffs);
  CHECK_EQ(2, filter.Size());
  filter.Filter(sig[{}], got_filtered_sig);

  ASSERT_THAT(got_filtered_sig.shape(), want_filtered_sig.shape());
  for (size_t t = 0; t < want_filtered_sig.shape()[0]; ++t) {
    for (size_t f = 0; f < want_filtered_sig.shape()[1]; ++f) {
      ASSERT_NEAR((got_filtered_sig[{t}][f]), (want_filtered_sig[{t}][f]), 1e-6)
          << "t=" << t << ", f=" << f << "";
    }
  }
}

TEST(Filterbank, MultiSectionFilterTest) {
  // Golden data produced by:
  //
  // scipy.signal.lfilter([0.5, 0.5], [1.0, 0.5],
  //                      scipy.signal.lfilter([1.0, -0.5], [0.5, 0.0],
  //                                           [1, 0, 0, 0]))
  //
  // and
  //
  // scipy.signal.lfilter([0.5, -0.5], [1.0, 0.5],
  //                      scipy.signal.lfilter([0.5, 1.0], [0.5, 0.0],
  //                                           [1, 0, 0, 0]))
  hwy::AlignedNDArray<float, 1> sig({4});
  sig[{}][0] = 1.0;

  const std::vector<std::vector<BACoeffs>> coeffs = {
      {{.b_coeffs = {0.5, 0.5}, .a_coeffs = {1.0, 0.5}},
       {.b_coeffs = {1.0, -0.5}, .a_coeffs = {0.5, 0.0}}},
      {{.b_coeffs = {0.5, -0.5}, .a_coeffs = {1.0, 0.5}},
       {.b_coeffs = {0.5, 1.0}, .a_coeffs = {0.5, 0.0}}}};

  hwy::AlignedNDArray<float, 2> want_filtered_sig({4, 2});
  Copy(want_filtered_sig[{0}], {1.0, 0.5});
  Copy(want_filtered_sig[{1}], {0.0, 0.25});
  Copy(want_filtered_sig[{2}], {-0.5, -1.125});
  Copy(want_filtered_sig[{3}], {0.25, 0.5625});

  hwy::AlignedNDArray<float, 2> got_filtered_sig(
      {sig.shape()[0], coeffs.size()});

  Filterbank filter(coeffs);
  CHECK_EQ(2, filter.Size());
  filter.Filter(sig[{}], got_filtered_sig);

  ASSERT_THAT(got_filtered_sig.shape(), want_filtered_sig.shape());
  for (size_t t = 0; t < want_filtered_sig.shape()[0]; ++t) {
    for (size_t f = 0; f < want_filtered_sig.shape()[1]; ++f) {
      ASSERT_NEAR((got_filtered_sig[{t}][f]), (want_filtered_sig[{t}][f]), 1e-6)
          << "t=" << t << ", f=" << f << "";
    }
  }
}

TEST(Filterbank, RealSignalFilterTest) {
  // Golden data produced by:
  //
  // signal = np.zeros((100,))
  // signal[0] = 1
  // sos0 = np.asarray(
  //     scipy.signal.ellip(
  //         N=4,
  //         rp=1,
  //         rs=30,
  //         Wn=[15, 25],
  //         btype='bandpass',
  //         analog=False,
  //         fs=100,
  //         output='sos',
  //     )
  // )
  // sos1 = np.asarray(
  //     scipy.signal.ellip(
  //         N=4,
  //         rp=1,
  //         rs=30,
  //         Wn=[25, 35],
  //         btype='bandpass',
  //         analog=False,
  //         fs=100,
  //         output='sos',
  //     )
  // )
  // filtered_0 = scipy.signal.sosfilt(sos0, signal)
  // filtered_1 = scipy.signal.sosfilt(sos1, signal)
  // want_filtered = np.asarray([filtered_0, filtered_1]).T
  hwy::AlignedNDArray<float, 1> sig({100});
  sig[{}][0] = 1.0;

  const std::vector<std::vector<BACoeffs>> coeffs = {
      {{.b_coeffs = {0.04371546531495744, 0.03943667465513112,
                     0.043715465314957444},
        .a_coeffs = {1.0, -0.26722799717268303, 0.773908926150085}},
       {.b_coeffs = {1.0, -1.6422797077033742, 1.0},
        .a_coeffs = {1.0, -0.861731316839711, 0.7967364939417214}},
       {.b_coeffs = {1.0, 0.19842299143287995, 1.0},
        .a_coeffs = {1.0, -0.0004824844375998036, 0.9475212648320236}},
       {.b_coeffs = {1.0, -1.298283994707207, 1.0},
        .a_coeffs = {1.0, -1.1501686955480686, 0.9573238983855463}}},
      {{.b_coeffs = {0.04371546531495742, -0.03943667465513113,
                     0.043715465314957416},
        .a_coeffs = {1.0, 0.26722799717268275, 0.7739089261500852}},
       {.b_coeffs = {1.0, 1.642279707703374, 0.9999999999999998},
        .a_coeffs = {1.0, 0.8617313168397108, 0.7967364939417212}},
       {.b_coeffs = {1.0, -0.19842299143288059, 0.9999999999999999},
        .a_coeffs = {1.0, 0.0004824844375993433, 0.9475212648320236}},
       {.b_coeffs = {1.0, 1.298283994707207, 1.0},
        .a_coeffs = {1.0, 1.150168695548068, 0.9573238983855461}}},
  };

  hwy::AlignedNDArray<float, 2> got_filtered_sig(
      {sig.shape()[0], coeffs.size()});

  const std::vector<std::vector<float>> want_filtered_sig = {
      {0.04371546531495744, 0.04371546531495742},
      {0.019216950997638003, -0.019216950997637983},
      {-0.04058281253655116, -0.0405828125365512},
      {-0.07969469603539944, 0.0796946960353994},
      {0.023893574282494272, 0.02389357428249442},
      {0.13685027962520904, -0.13685027962520907},
      {0.06389270769308888, 0.06389270769308866},
      {-0.1243118627171007, 0.12431186271710096},
      {-0.1554216140929805, -0.15542161409298033},
      {0.03677628134862089, -0.03677628134862138},
      {0.1827890966196795, 0.18278909661967963},
      {0.07228012337922984, -0.07228012337922934},
      {-0.1244578142385641, -0.12445781423856453},
      {-0.1275194194925982, 0.12751941949259793},
      {0.028818044446633082, 0.02881804444663362},
      {0.09750564120861491, -0.09750564120861491},
      {0.024844416798183718, 0.02484441679818332},
      {-0.027903865325417172, 0.027903865325417297},
      {-0.0037733437534210115, -0.003773343753420845},
      {-0.005058584099275981, 0.005058584099275974},
      {-0.04682670273264107, -0.04682670273264117},
      {-0.02411030884083278, 0.024110308840832616},
      {0.05895534956241046, 0.058955349562410685},
      {0.06832249768400268, -0.0683224976840025},
      {-0.01948151314782963, -0.019481513147829997},
      {-0.07019991959554642, 0.0701999195955464},
      {-0.022203116183308936, -0.022203116183308603},
      {0.029904072967873833, -0.029904072967873962},
      {0.020502946959957434, 0.020502946959957302},
      {0.002938654846422927, -0.0029386548464228855},
      {0.012777301568995023, 0.012777301568995049},
      {0.0044677910630356536, -0.0044677910630355165},
      {-0.031136288596501072, -0.031136288596501197},
      {-0.03124007430207751, 0.03124007430207732},
      {0.015257115269453744, 0.015257115269454011},
      {0.0371531356910558, -0.03715313569105573},
      {0.007694863978823229, 0.00769486397882297},
      {-0.01605042088563693, 0.01605042088563698},
      {-0.0059574717104290435, -0.005957471710428947},
      {-0.0016208554933552846, 0.0016208554933553106},
      {-0.01503573608528515, -0.01503573608528517},
      {-0.0062506082136497745, 0.006250608213649577},
      {0.02471176739540224, 0.024711767395402387},
      {0.02515925842663852, -0.025159258426638298},
      {-0.010984427861716271, -0.010984427861716571},
      {-0.027395355662157735, 0.027395355662157665},
      {-0.005520788304712063, -0.005520788304711783},
      {0.010537594124424022, -0.010537594124424081},
      {0.0023586092297023745, 0.0023586092297022705},
      {0.0016559093549600515, -0.0016559093549600688},
      {0.014323258849275393, 0.014323258849275409},
      {0.006446381576346074, -0.006446381576345882},
      {-0.02043289022071307, -0.02043289022071321},
      {-0.02187866410547885, 0.021878664105478625},
      {0.007812920843682845, 0.007812920843683142},
      {0.02266804122928115, -0.02266804122928108},
      {0.005936796396464593, 0.00593679639646431},
      {-0.007997368681006416, 0.007997368681006489},
      {-0.0029057962137844706, -0.0029057962137843674},
      {-0.002232783870342231, 0.0022327838703422205},
      {-0.011183657631142105, -0.011183657631142107},
      {-0.004527899881639269, 0.004527899881639105},
      {0.016362996347792758, 0.016362996347792858},
      {0.017340313286840763, -0.01734031328684055},
      {-0.006041277568520322, -0.006041277568520584},
      {-0.01811127369654711, 0.01811127369654704},
      {-0.00529050331531373, -0.00529050331531347},
      {0.006145084065179585, -0.006145084065179662},
      {0.002929780521083985, 0.0029297805210838973},
      {0.002322032526739657, -0.002322032526739638},
      {0.008556771594444654, 0.008556771594444642},
      {0.0030224635289540077, -0.0030224635289538577},
      {-0.012945325142797028, -0.012945325142797118},
      {-0.013416658979918327, 0.013416658979918138},
      {0.004768801542632213, 0.004768801542632451},
      {0.014194649691523231, -0.014194649691523177},
      {0.004356231093691425, 0.004356231093691196},
      {-0.004671359291573689, 0.004671359291573762},
      {-0.0025289232488638655, -0.002528923248863791},
      {-0.002131329038194907, 0.002131329038194885},
      {-0.006669734369094326, -0.006669734369094308},
      {-0.0021301648019013053, 0.0021301648019011717},
      {0.01022171784730147, 0.010221717847301543},
      {0.01047446166905929, -0.010474461669059115},
      {-0.0036966367720746404, -0.0036966367720748534},
      {-0.011128879701379827, 0.011128879701379777},
      {-0.003609629744046465, -0.003609629744046259},
      {0.0035183508558845255, -0.0035183508558845967},
      {0.002141817511822524, 0.0021418175118224636},
      {0.0019183688869546774, -0.0019183688869546531},
      {0.005237956209278493, 0.005237956209278466},
      {0.0015174677901770221, -0.0015174677901769024},
      {-0.008084857528949364, -0.008084857528949423},
      {-0.008224011308233876, 0.008224011308233718},
      {0.0028441181308030417, 0.0028441181308032316},
      {0.008750602070501512, -0.008750602070501472},
      {0.0030169100466182516, 0.00301691004661807},
      {-0.0026403466489508076, 0.002640346648950878},
      {-0.001824910331138443, -0.0018249103311383972},
      {-0.0017172684813795145, 0.001717268481379491},
  };

  Filterbank filter(coeffs);
  CHECK_EQ(2, filter.Size());
  filter.Filter(sig[{}], got_filtered_sig);

  for (size_t t = 0; t < want_filtered_sig.size(); ++t) {
    for (size_t f = 0; f < want_filtered_sig.front().size(); ++f) {
      ASSERT_NEAR((got_filtered_sig[{t}][f]), (want_filtered_sig[t][f]), 1e-6)
          << "t=" << t << ", f=" << f << "";
    }
  }
}

TEST(Filterbank, ManyFilterTest) {
  const float sample_rate = 48000;
  const float low_threshold = 318;
  const float high_threshold = 340;
  const float center_frequency = (low_threshold + high_threshold) * 0.5;
  const size_t signal_length = 4;
  const size_t num_filters = 1024;
  std::vector<std::vector<BACoeffs>> all_coeffs(num_filters);
  Cam cam;
  for (size_t filter_index = 0; filter_index < num_filters; ++filter_index) {
    const std::vector<BACoeffs> coeffs =
        DigitalSOSBandPass(cam.filter_order, cam.filter_pass_band_ripple,
                           cam.filter_stop_band_ripple, low_threshold,
                           high_threshold, sample_rate);
    all_coeffs[filter_index] = coeffs;
  }
  Filterbank filter(all_coeffs);

  hwy::AlignedNDArray<float, 1> in_signal({signal_length});
  for (size_t sample_index = 0; sample_index < in_signal.shape()[0];
       ++sample_index) {
    in_signal[{}][sample_index] =
        std::sin(static_cast<float>(sample_index) * 2 * M_PI *
                 center_frequency / sample_rate);
  }

  hwy::AlignedNDArray<float, 2> out_signal({in_signal.shape()[0], num_filters});

  filter.Filter(in_signal[{}], out_signal);

  for (size_t filter_index = 0; filter_index < num_filters; ++filter_index) {
    float energy = 0;
    for (size_t sample_index = 0; sample_index < out_signal.shape()[0];
         ++sample_index) {
      energy += std::pow(out_signal[{sample_index}][filter_index], 2);
    }
    EXPECT_GT(energy, 0);
  }
}

void BM_Filterbank(benchmark::State& state) {
  const size_t sample_rate = 48000;
  const size_t filter_order = 3;
  const size_t num_filters = 1024;

  hwy::AlignedNDArray<float, 1> in_signal(
      {size_t(sample_rate * state.range(0))});
  in_signal[{}][0] = 1;

  hwy::AlignedNDArray<float, 2> out_signals(
      {in_signal.shape()[0], num_filters});

  std::vector<std::vector<BACoeffs>> coeffs(num_filters);
  for (auto& filter_coeffs : coeffs) {
    std::vector<double> b_coeffs(filter_order);
    b_coeffs[0] = 1.0;
    std::vector<double> a_coeffs(filter_order);
    a_coeffs[0] = 0.5;
    a_coeffs[1] = 0.5;
    filter_coeffs.push_back({.b_coeffs = b_coeffs, .a_coeffs = a_coeffs});
  }

  Filterbank filter(coeffs);

  for (auto s : state) {
    filter.Filter(in_signal[{}], out_signals);
  }
  state.SetItemsProcessed(out_signals.size() * state.iterations());
}
BENCHMARK_RANGE(BM_Filterbank, 1, 64);

void BM_MultiSectionFilterbank(benchmark::State& state) {
  const size_t sample_rate = 48000;
  const size_t filter_order = 3;
  const size_t num_sections = 3;
  const size_t num_filters = 1024;

  hwy::AlignedNDArray<float, 1> in_signal(
      {size_t(sample_rate * state.range(0))});
  in_signal[{}][0] = 1;

  hwy::AlignedNDArray<float, 2> out_signals(
      {in_signal.shape()[0], num_filters});

  std::vector<std::vector<BACoeffs>> coeffs(num_filters);
  for (auto& filter_coeffs : coeffs) {
    std::vector<double> b_coeffs(filter_order);
    b_coeffs[0] = 1.0;
    std::vector<double> a_coeffs(filter_order);
    a_coeffs[0] = 0.5;
    a_coeffs[1] = 0.5;
    for (size_t i = 0; i < num_sections; ++i) {
      filter_coeffs.push_back({.b_coeffs = b_coeffs, .a_coeffs = a_coeffs});
    }
  }

  Filterbank filter(coeffs);

  for (auto s : state) {
    filter.Filter(in_signal[{}], out_signals);
  }
  state.SetItemsProcessed(out_signals.size() * state.iterations());
}
BENCHMARK_RANGE(BM_MultiSectionFilterbank, 1, 64);

}  // namespace

}  // namespace zimtohrli
