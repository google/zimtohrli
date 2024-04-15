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

#include <cstddef>
#include <vector>

#include "benchmark/benchmark.h"
#include "gtest/gtest.h"
#include "hwy/aligned_allocator.h"
#include "hwy/base.h"
#include "zimt/cam.h"

namespace zimtohrli {

namespace {

// These are the frequencies for which ISO 226 defines a precise set of a_f,
// L_U, and T_f parameters.
const std::vector<float> golden_hz = {
    20.,   25.,   31.5,  40.,   50.,   63.,   80.,   100.,   125.,  160.,
    200.,  250.,  315.,  400.,  500.,  630.,  800.,  1000.,  1250., 1600.,
    2000., 2500., 3150., 4000., 5000., 6300., 8000., 10000., 12500.};

const std::vector<float> golden_db_spl = {0,  10, 20, 30, 40,
                                          50, 60, 70, 80, 90};

// These are values generated using the ISO 226 formula for the frequencies and
// sound pressures in golden_hz and golden_db_spl.
const std::vector<std::vector<float>> golden_phons = {
    {-1.3903099e+01, -7.3551025e+00, -4.7630081e+00, -3.6093903e+00,
     -3.1443329e+00, -3.0974045e+00, -3.2647018e+00, -3.6001129e+00,
     -3.8923492e+00, -4.2106018e+00, -4.1214066e+00, -3.8409424e+00,
     -3.1294327e+00, -2.2001190e+00, -1.1803741e+00, -2.5842285e-01,
     3.0133057e-01,  2.1774292e-02,  -9.9832153e-01, 9.9319458e-01,
     3.5518494e+00,  6.0658493e+00,  7.4066086e+00,  6.8140335e+00,
     3.6595917e+00,  -2.8535309e+00, -8.0267334e+00, -6.2076874e+00,
     -4.1128769e+00},
    {-1.3898758e+01, -7.3454361e+00, -4.7379532e+00, -3.5460663e+00,
     -3.0026474e+00, -2.7937469e+00, -2.6539078e+00, -2.4963150e+00,
     -2.0761871e+00, -1.2966461e+00, -4.3716431e-02, 1.5078430e+00,
     3.4538727e+00,  5.5299225e+00,  7.3065720e+00,  8.9019928e+00,
     1.0028633e+01,  1.0021263e+01,  8.7837753e+00,  9.6363068e+00,
     1.2494278e+01,  1.5052795e+01,  1.6168289e+01,  1.5420959e+01,
     1.2250916e+01,  6.0301819e+00,  1.5581512e-01,  -3.1636810e-01,
     9.6312714e-01},
    {-1.3883995e+01, -7.3144531e+00, -4.6624985e+00, -3.3667755e+00,
     -2.6252060e+00, -2.0384369e+00, -1.2491455e+00, -1.6953278e-01,
     1.4027100e+00,  3.6595001e+00,  6.1953659e+00,  8.8953323e+00,
     1.1750778e+01,  1.4522568e+01,  1.6674217e+01,  1.8564072e+01,
     1.9925385e+01,  2.0020973e+01,  1.8590820e+01,  1.8725807e+01,
     2.1753792e+01,  2.4339554e+01,  2.5315926e+01,  2.4455063e+01,
     2.1275146e+01,  1.5286537e+01,  9.1077957e+00,  7.1482315e+00,
     8.0844879e+00},
    {-1.3833824e+01, -7.2154999e+00, -4.4365921e+00, -2.8655548e+00,
     -1.6438599e+00, -2.3941040e-01, 1.7635803e+00,  4.2788925e+00,
     7.3141479e+00,  1.1054703e+01,  1.4610123e+01,  1.8039047e+01,
     2.1343002e+01,  2.4371765e+01,  2.6600967e+01,  2.8527328e+01,
     2.9919456e+01,  3.0020813e+01,  2.8412086e+01,  2.8091606e+01,
     3.1205032e+01,  3.3807014e+01,  3.4700153e+01,  3.3753975e+01,
     3.0568062e+01,  2.4769516e+01,  1.8556381e+01,  1.5867737e+01,
     1.7019875e+01},
    {-1.3664116e+01, -6.9019547e+00, -3.7716980e+00, -1.5111694e+00,
     7.6158142e-01,  3.6687775e+00,  7.4610367e+00,  1.1620468e+01,
     1.5952805e+01,  2.0688728e+01,  2.4752937e+01,  2.8426956e+01,
     3.1779446e+01,  3.4746346e+01,  3.6859612e+01,  3.8664875e+01,
     3.9968727e+01,  4.0020721e+01,  3.8241444e+01,  3.7622776e+01,
     4.0769558e+01,  4.3381035e+01,  4.4225311e+01,  4.3211868e+01,
     4.0022392e+01,  3.4386562e+01,  2.8306976e+01,  2.5455246e+01,
     2.7271065e+01},
    {-1.3098549e+01, -5.9332275e+00, -1.9056168e+00, 1.8552094e+00,
     5.9703827e+00,  1.0904160e+01,  1.6490486e+01,  2.1854713e+01,
     2.6874260e+01,  3.1930981e+01,  3.6025532e+01,  3.9577930e+01,
     4.2713623e+01,  4.5423294e+01,  4.7307434e+01,  4.8901115e+01,
     5.0049103e+01,  5.0020664e+01,  4.8075401e+01,  4.7250942e+01,
     5.0400078e+01,  5.3017059e+01,  5.3832977e+01,  5.2763355e+01,
     4.9571800e+01,  4.4081619e+01,  3.8234604e+01,  3.5585052e+01,
     3.8342419e+01},
    {-1.1300659e+01, -3.1479034e+00, 2.7532196e+00, 8.9248657e+00,
     1.5193466e+01,  2.1811180e+01,  2.8428535e+01, 3.4209621e+01,
     3.9280151e+01,  4.4133587e+01,  4.7944469e+01, 5.1155720e+01,
     5.3923893e+01,  5.6267879e+01,  5.7860573e+01, 5.9192566e+01,
     6.0146931e+01,  6.0020638e+01,  5.7911972e+01, 5.6935322e+01,
     6.0068733e+01,  6.2688881e+01,  6.3488449e+01, 6.2369312e+01,
     5.9176548e+01,  5.3821640e+01,  4.8263771e+01, 4.6032246e+01,
     4.9877022e+01},
    {-6.2854538e+00, 3.6035843e+00, 1.2077072e+01, 2.0554581e+01, 2.8196548e+01,
     3.5474358e+01,  4.2217159e+01, 4.7786701e+01, 5.2496593e+01, 5.6846653e+01,
     6.0204872e+01,  6.2959904e+01, 6.5282127e+01, 6.7203476e+01, 6.8471588e+01,
     6.9514694e+01,  7.0254539e+01, 7.0020622e+01, 6.7750031e+01, 6.6652054e+01,
     6.9759323e+01,  7.2381287e+01, 7.3171463e+01, 7.2006760e+01, 6.8813301e+01,
     6.3587433e+01,  5.8350441e+01, 5.6658066e+01, 6.1658684e+01},
    {4.4353256e+00, 1.5924835e+01, 2.6197578e+01, 3.5700245e+01, 4.3583611e+01,
     5.0657776e+01, 5.6952721e+01, 6.1970581e+01, 6.6113098e+01, 6.9814133e+01,
     7.2638031e+01, 7.4880775e+01, 7.6718193e+01, 7.8187920e+01, 7.9114166e+01,
     7.9853806e+01, 8.0367615e+01, 8.0020615e+01, 7.7588928e+01, 7.6387344e+01,
     7.9462494e+01, 8.2085503e+01, 8.2870293e+01, 8.1662338e+01, 7.8468475e+01,
     7.3367943e+01, 6.8469452e+01, 6.7382225e+01, 7.3568047e+01},
    {2.0816589e+01, 3.2714939e+01, 4.3324997e+01, 5.2687622e+01, 6.0107780e+01,
     6.6536476e+01, 7.2117874e+01, 7.6433060e+01, 7.9917076e+01, 8.2904297e+01,
     8.5156677e+01, 8.6860893e+01, 8.8194786e+01, 8.9198418e+01, 8.9773895e+01,
     9.0202286e+01, 9.0483742e+01, 9.0020607e+01, 8.7428299e+01, 8.6133240e+01,
     8.9172867e+01, 9.1796471e+01, 9.2578186e+01, 9.1328331e+01, 8.8134247e+01,
     8.3156845e+01, 7.8606567e+01, 7.8159828e+01, 8.5542358e+01}};

TEST(Loudness, GoldenTest) {
  hwy::AlignedNDArray<float, 2> channels_db_spl(
      {golden_db_spl.size(), golden_hz.size()});
  for (size_t sample_index = 0; sample_index < channels_db_spl.shape()[0];
       ++sample_index) {
    for (size_t channel_index = 0; channel_index < channels_db_spl.shape()[1];
         ++channel_index) {
      channels_db_spl[{sample_index}][channel_index] =
          golden_db_spl[sample_index];
    }
  }
  hwy::AlignedNDArray<float, 2> frequencies({3, golden_hz.size()});
  for (size_t hz_index = 0; hz_index < frequencies.shape()[1]; ++hz_index) {
    frequencies[{1}][hz_index] = golden_hz[hz_index];
  }
  const Loudness l;
  hwy::AlignedNDArray<float, 2> phons(
      {channels_db_spl.shape()[0], channels_db_spl.shape()[1]});
  l.PhonsFromSPL(channels_db_spl, frequencies, phons);
  for (size_t sample_index = 1; sample_index < channels_db_spl.shape()[0];
       ++sample_index) {
    for (size_t channel_index = 0; channel_index < channels_db_spl.shape()[1];
         ++channel_index) {
      // The parameterization is worse at the low end (0dB or 20Hz) so a greater
      // tolerance is allowed there.
      // The parameterization also diverges from ISO 226 at ~10k+ since it's
      // believed to be too sensitive in that range.
      float tolerance = sample_index > 0 && channel_index > 0 &&
                                golden_hz[channel_index] < 10000
                            ? 2
                        : golden_hz[channel_index] < 12500 ? 3
                                                           : 5;
      EXPECT_NEAR(phons[{sample_index}][channel_index],
                  golden_phons[sample_index][channel_index], tolerance)
          << "hz=" << golden_hz[channel_index]
          << ", dB SPL=" << golden_db_spl[sample_index];
    }
  }
}

TEST(Loudness, VeryHighFrequencyTest) {
  hwy::AlignedNDArray<float, 2> channels_db_spl({1, 1});
  channels_db_spl[{0}] = {21.6323};
  hwy::AlignedNDArray<float, 2> frequencies({3, 1});
  frequencies[{1}] = {19914.4};
  hwy::AlignedNDArray<float, 2> channels_phons({1, 1});
  Loudness().PhonsFromSPL(channels_db_spl, frequencies, channels_phons);
  EXPECT_NEAR(channels_phons[{0}][0], 2.4216, 1e-2);
}

void BM_PhonsFromSPL(benchmark::State& state) {
  const Cam cam;
  const CamFilterbank filterbank = cam.CreateFilterbank(48000);
  hwy::AlignedNDArray<float, 2> channels(
      {static_cast<size_t>(100 * state.range(0)), filterbank.filter.Size()});
  const Loudness loudness;
  for (auto s : state) {
    loudness.PhonsFromSPL(channels, filterbank.thresholds_hz, channels);
  }
  state.SetItemsProcessed(channels.size() * state.iterations());
}
BENCHMARK_RANGE(BM_PhonsFromSPL, 1, 64);

}  // namespace

}  // namespace zimtohrli
