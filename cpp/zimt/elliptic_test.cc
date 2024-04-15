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

#include "zimt/elliptic.h"

#include <cmath>
#include <complex>
#include <cstdlib>
#include <cstring>
#include <ostream>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "benchmark/benchmark.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace zimtohrli {

namespace {

TEST(Elliptic, EllipticIntegralTest) {
  // Golden values produced by:
  //
  // scipy.special.ellipk(np.linspace(0, 1, 100, endpoint=False))
  const std::vector<double> golden = {
      1.57079633, 1.57474556, 1.57873991, 1.58278034, 1.58686785, 1.59100345,
      1.59518822, 1.59942324, 1.60370965, 1.60804862, 1.61244135, 1.61688909,
      1.62139314, 1.62595483, 1.63057555, 1.63525673, 1.63999987, 1.64480649,
      1.64967821, 1.65461667, 1.6596236,  1.66470079, 1.66985009, 1.67507343,
      1.68037282, 1.68575035, 1.6912082,  1.69674862, 1.70237398, 1.70808673,
      1.71388945, 1.71978481, 1.72577561, 1.73186478, 1.73805537, 1.7443506,
      1.7507538,  1.7572685,  1.76389839, 1.77064732, 1.77751937, 1.7845188,
      1.79165012, 1.79891804, 1.80632756, 1.81388394, 1.82159273, 1.8294598,
      1.83749136, 1.845694,   1.85407468, 1.8626408,  1.87140024, 1.88036136,
      1.88953308, 1.89892491, 1.90854702, 1.91841027, 1.92852632, 1.93890767,
      1.94956775, 1.96052104, 1.97178316, 1.98337098, 1.99530278, 2.0075984,
      2.02027943, 2.03336941, 2.04689408, 2.06088165, 2.07536314, 2.09037275,
      2.10594832, 2.12213186, 2.13897018, 2.15651565, 2.17482709, 2.19397093,
      2.2140225,  2.23506776, 2.25720533, 2.28054914, 2.30523174, 2.33140857,
      2.35926355, 2.38901649, 2.42093296, 2.45533803, 2.49263532, 2.53333455,
      2.57809211, 2.62777333, 2.68355141, 2.747073,   2.8207525,  2.90833725,
      3.01611249, 3.15587495, 3.35414145, 3.69563736};
  const double step = 1.0 / golden.size();
  for (int i = 0; i < golden.size(); ++i) {
    ASSERT_NEAR(golden[i], EllipticIntegral1(step * static_cast<double>(i)),
                1e-8);
  }
}

std::vector<double> LinspaceWithoutEndpoint(double a, double b, int steps) {
  std::vector<double> result;
  const double step = (b - a) / steps;
  for (int i = 0; i < steps; ++i) {
    result.push_back(a + b * step * i);
  }
  return result;
}

TEST(Elliptic, EllipticJacobianTest) {
  // Golden values produced by:
  //
  // ellipjs = []
  // for u in np.linspace(0.01, 1, 1, endpoint=False):
  //   for m in np.linspace(0, 1, 1, endpoint=False):
  //     ellipjs.append(scipy.special.ellipj(u, m))
  const std::vector<std::vector<double>> golden = {
      {
          0.009999833334166664,
          0.9999500004166653,
          1.0,
          0.01,
      },
      {
          0.009999816668674971,
          0.9999500005833256,
          0.9999950001708305,
          0.009999983333674998,
      },
      {
          0.009999800003199933,
          0.9999500007499855,
          0.9999900003499931,
          0.009999966667366656,
      },
      {
          0.00999978333774156,
          0.999950000916645,
          0.9999850005374878,
          0.009999950001074983,
      },
      {
          0.009999766672299846,
          0.9999500010833039,
          0.9999800007333146,
          0.009999933334799972,
      },
      {
          0.009999750006874798,
          0.9999500012499625,
          0.9999750009374734,
          0.00999991666854163,
      },
      {
          0.009999733341466411,
          0.9999500014166206,
          0.9999700011499644,
          0.009999900002299953,
      },
      {
          0.009999716676074681,
          0.9999500015832783,
          0.9999650013707874,
          0.009999883336074936,
      },
      {
          0.00999970001069961,
          0.9999500017499355,
          0.9999600015999424,
          0.009999866669866584,
      },
      {
          0.009999683345341206,
          0.9999500019165923,
          0.9999550018374296,
          0.0099998500036749,
      },
      {
          0.1087842900157316,
          0.9940653792612301,
          1.0,
          0.109,
      },
      {
          0.108762886449793,
          0.9940677213002691,
          0.9994083567056616,
          0.10897846867905198,
      },
      {
          0.10874148535077643,
          0.9940700626030878,
          0.9988168289895507,
          0.10895693989046498,
      },
      {
          0.10872008671850769,
          0.9940724031698698,
          0.9982254168388901,
          0.10893541363403587,
      },
      {
          0.10869869055281263,
          0.9940747430007987,
          0.9976341202409036,
          0.1089138899095616,
      },
      {
          0.10867729685351707,
          0.9940770820960578,
          0.9970429391828148,
          0.10889236871683913,
      },
      {
          0.1086559056204469,
          0.9940794204558309,
          0.996451873651849,
          0.10887085005566544,
      },
      {
          0.10863451685342786,
          0.9940817580803011,
          0.995860923635231,
          0.1088493339258374,
      },
      {
          0.10861313055228576,
          0.994084094969652,
          0.9952700891201874,
          0.10882782032715199,
      },
      {
          0.1085917467168465,
          0.9940864311240669,
          0.9946793700939442,
          0.10880630925940626,
      },
      {
          0.20650342240103153,
          0.9784458781847165,
          1.0,
          0.20800000000000002,
      },
      {
          0.20635796587329022,
          0.9784765658515476,
          0.9978685479521157,
          0.2078513415512028,
      },
      {
          0.20621256542155716,
          0.9785072191160676,
          0.9957385578415913,
          0.2077027450706572,
      },
      {
          0.20606722103978706,
          0.9785378380077796,
          0.993610029198519,
          0.2075542105430807,
      },
      {
          0.20592193272192044,
          0.9785684225561689,
          0.9914829615528588,
          0.2074057379531889,
      },
      {
          0.20577670046188434,
          0.9785989727907034,
          0.9893573544344378,
          0.20725732728569585,
      },
      {
          0.20563152425359177,
          0.9786294887408331,
          0.9872332073729523,
          0.2071089785253138,
      },
      {
          0.20548640409094193,
          0.9786599704359906,
          0.9851105198979665,
          0.2069606916567533,
      },
      {
          0.20534133996782042,
          0.9786904179055909,
          0.982989291538914,
          0.2068124666647233,
      },
      {
          0.2051963318780989,
          0.9787208311790309,
          0.9808695218250977,
          0.20666430353393098,
      },
      {
          0.3022002672564511,
          0.9532444589244301,
          1.0,
          0.30700000000000005,
      },
      {
          0.30174932371290236,
          0.9533873009636776,
          0.9954369565994023,
          0.30652697364574816,
      },
      {
          0.3012987042446393,
          0.953529805942374,
          0.99088032484458,
          0.3060543579685974,
      },
      {
          0.3008484089817609,
          0.9536719744299625,
          0.9863301021686112,
          0.30558215281532486,
      },
      {
          0.3003984380533547,
          0.9538138069954245,
          0.9817862859940761,
          0.30511035803243175,
      },
      {
          0.2999487915874984,
          0.9539553042072776,
          0.9772488737330935,
          0.30463897346614455,
      },
      {
          0.299499469711262,
          0.9540964666335752,
          0.9727178627873572,
          0.30416799896241525,
      },
      {
          0.2990504725507105,
          0.954237294841905,
          0.9681932505481732,
          0.3036974343669231,
      },
      {
          0.2986018002309056,
          0.9543777893993879,
          0.963675034396497,
          0.30322727952507506,
      },
      {
          0.29815345287590833,
          0.9545179508726767,
          0.9591632117029698,
          0.30275753428200697,
      },
      {
          0.3949376656053987,
          0.9187079189199134,
          1.0,
          0.406,
      },
      {
          0.3939466802430381,
          0.9191332945364832,
          0.9922099582812848,
          0.4049215768380184,
      },
      {
          0.39295664860076207,
          0.9195569978638938,
          0.9844374101303197,
          0.4038446890655743,
      },
      {
          0.3919675734386606,
          0.9199790331157598,
          0.976682356967598,
          0.4027693363134326,
      },
      {
          0.3909794574982166,
          0.9203994045056745,
          0.9689448000406214,
          0.4016955182059214,
      },
      {
          0.3899923035023501,
          0.9208181162471397,
          0.9612247404246654,
          0.40062323436096026,
      },
      {
          0.3890061141554601,
          0.9212351725534957,
          0.9535221790235408,
          0.3995524843900846,
      },
      {
          0.3880208921434702,
          0.9216505776378516,
          0.9458371165703585,
          0.39848326789847427,
      },
      {
          0.3870366401338712,
          0.9220643357130154,
          0.9381695536282914,
          0.39741558448497893,
      },
      {
          0.38605336077576596,
          0.9224764509914257,
          0.9305194905913377,
          0.3963494337421453,
      },
      {
          0.48380744032396017,
          0.8751744744262013,
          1.0,
          0.505,
      },
      {
          0.4820232305595097,
          0.8761584361295471,
          0.9883144036793642,
          0.5029624557542185,
      },
      {
          0.48024063060895644,
          0.8771367833538346,
          0.9766646237795563,
          0.5009290282783255,
      },
      {
          0.47845966169604814,
          0.8781095331048987,
          0.9650507269770388,
          0.4988997195881147,
      },
      {
          0.47668034488880845,
          0.8790767024535951,
          0.9534727785934293,
          0.49687453163353756,
      },
      {
          0.4749027010996289,
          0.8800383085345073,
          0.9419308426015883,
          0.4948534662990031,
      },
      {
          0.47312675108536967,
          0.8809943685446592,
          0.9304249816317497,
          0.49283652540367545,
      },
      {
          0.47135251544747264,
          0.8819448997422344,
          0.9189552569776931,
          0.49082371070177583,
      },
      {
          0.4695800146320825,
          0.8828899194453028,
          0.9075217286029611,
          0.48881502388288256,
      },
      {
          0.4678092689301784,
          0.8838294450305512,
          0.8961244551471134,
          0.4868104665722337,
      },
      {
          0.567939289917337,
          0.8230704483628306,
          1.0,
          0.6040000000000001,
      },
      {
          0.5651299396476751,
          0.8250019098849499,
          0.9839018320601813,
          0.6005907473242772,
      },
      {
          0.5623213029624214,
          0.8269188304994902,
          0.9678630845563482,
          0.597190300910322,
      },
      {
          0.5595134800257243,
          0.8288212507347428,
          0.9518841314471268,
          0.593798681323446,
      },
      {
          0.5567065702612076,
          0.8307092118352867,
          0.9359653400907543,
          0.5904159087246089,
      },
      {
          0.5539006723474296,
          0.8325827557504812,
          0.9201070712621074,
          0.587042002871909,
      },
      {
          0.5510958842135193,
          0.8344419251229647,
          0.9043096791706654,
          0.5836769831221051,
      },
      {
          0.5482923030349911,
          0.8362867632771581,
          0.8885735114793879,
          0.5803208684321735,
      },
      {
          0.5454900252297318,
          0.8381173142077823,
          0.8728989093244951,
          0.5769736773608909,
      },
      {
          0.5426891464541622,
          0.8399336225683867,
          0.8572862073361309,
          0.5736354280704508,
      },
      {
          0.646509311380338,
          0.7629060953344922,
          1.0,
          0.7030000000000001,
      },
      {
          0.6425055388449643,
          0.7662810401892651,
          0.9791417993607229,
          0.6977635403233918,
      },
      {
          0.6384963483922288,
          0.7696248521778578,
          0.9583655266222579,
          0.6925429322550647,
      },
      {
          0.6344820834536634,
          0.7729375691323966,
          0.937672515184747,
          0.6873382730187043,
      },
      {
          0.6304630853990564,
          0.776219233173916,
          0.9170640758309316,
          0.6821496581127248,
      },
      {
          0.6264396934769644,
          0.7794698906542106,
          0.8965414966515736,
          0.6769771813111101,
      },
      {
          0.6224122447567751,
          0.7826895920969131,
          0.8761060429800146,
          0.6718209346646604,
      },
      {
          0.6183810740723265,
          0.7858783921378396,
          0.855758957335773,
          0.6666810085026373,
      },
      {
          0.6143465139670837,
          0.789036349464644,
          0.8355014593770578,
          0.6615574914347994,
      },
      {
          0.6103088946408818,
          0.792163526755824,
          0.8153347458620923,
          0.6564504703538256,
      },
      {
          0.7187480686775715,
          0.6952706047088867,
          1.0,
          0.802,
      },
      {
          0.7134820590896799,
          0.700673498397899,
          0.9742147274270262,
          0.7944553073483936,
      },
      {
          0.7081928168709921,
          0.7060190749068537,
          0.948521263244249,
          0.7869352316860811,
      },
      {
          0.7028812703933732,
          0.7113071908326232,
          0.9229232773741592,
          0.7794401026668213,
      },
      {
          0.6975483460602546,
          0.7165377205064666,
          0.8974243823094185,
          0.7719702444305551,
      },
      {
          0.692194967905445,
          0.7217105558368812,
          0.8720281321168429,
          0.764525975564836,
      },
      {
          0.6868220571993562,
          0.7268256061425219,
          0.8467380214958263,
          0.7571076090692651,
      },
      {
          0.6814305320629023,
          0.731882797975516,
          0.8215574848911846,
          0.7497154523229119,
      },
      {
          0.6760213070893099,
          0.736882074935509,
          0.7964898956603335,
          0.742349807054702,
      },
      {
          0.6705952929740617,
          0.7418233974747847,
          0.7715385652946515,
          0.7350109693167415,
      },
      {
          0.7839481278287302,
          0.6208263306866332,
          1.0,
          0.901,
      },
      {
          0.777486123200175,
          0.628900093998373,
          0.9693046646040223,
          0.8906586214208659,
      },
      {
          0.770962033738929,
          0.6368811054923317,
          0.9386817929983656,
          0.8803503126984921,
      },
      {
          0.7643779270060633,
          0.6447684737220899,
          0.9081398104982372,
          0.8700759773103003,
      },
      {
          0.7577358825791825,
          0.6525613628249434,
          0.8776870358509229,
          0.8598365051165969,
      },
      {
          0.7510379902302124,
          0.6602589925407781,
          0.8473316756828354,
          0.849632772049459,
      },
      {
          0.7442863481167928,
          0.6678606381626097,
          0.8170818191614477,
          0.8394656398165771,
      },
      {
          0.737483060990126,
          0.6753656304200223,
          0.7869454328775558,
          0.8293359556202022,
      },
      {
          0.730630238422031,
          0.6827733552968848,
          0.7569303559527076,
          0.8192445518913015,
      },
      {
          0.7237299930538686,
          0.690083253784822,
          0.7270442953760261,
          0.8091922460390197,
      },
  };
  int golden_index = 0;
  for (const auto& u : LinspaceWithoutEndpoint(0.01, 1.0, 10)) {
    for (const auto& m : LinspaceWithoutEndpoint(0.0, 1.0, 10)) {
      ASSERT_THAT(EllipticJacobian(u, m),
                  testing::Pointwise(
                      testing::DoubleNear(1e-7),
                      std::vector<double>{golden[golden_index].begin(),
                                          golden[golden_index].begin() + 3}));
      ++golden_index;
    }
  }
}

struct LowPassTestCase {
  int order;
  double pass_band_ripple;
  double stop_band_ripple;
  std::vector<std::complex<double>> want_zeros;
  std::vector<std::complex<double>> want_poles;
  double want_gain;
};

template <typename T>
class Near {
 public:
  using is_gtest_matcher = void;

  Near(const std::vector<T> expected, double tolerance)
      : expected(expected), tolerance(tolerance) {}

  template <typename V>
  bool MatchAndExplain(const V& value, std::ostream* listener) const {
    if (value.size() != expected.size()) {
      if (listener != nullptr) {
        *listener << "value.size() = " << value.size()
                  << ", expected.size() = " << expected.size();
      }
      return false;
    }
    for (int i = 0; i < value.size(); ++i) {
      if (std::abs(value[i] - expected[i]) > tolerance) {
        if (listener != nullptr) {
          *listener << "value[i] = " << value[i]
                    << ", expected[i] = " << expected[i]
                    << ", diff = " << std::abs(value[i] - expected[i])
                    << ", tolerance = " << tolerance;
        }
        return false;
      }
    }
    return true;
  }

  void DescribeTo(std::ostream* os) const {
    *os << "value == [";
    for (const auto& expected_value : expected) {
      *os << expected_value << ", ";
    }
    *os << "]";
  }

  void DescribeNegationTo(std::ostream* os) const {
    *os << "value != [";
    for (const auto& expected_value : expected) {
      *os << expected_value << ", ";
    }
    *os << "]";
  }

  std::vector<T> expected;
  double tolerance;
};

template <typename T>
Near<T> N(const std::vector<T> expected, double tolerance) {
  return Near<T>(expected, tolerance);
}

TEST(Elliptic, LowPassTest) {
  // Golden values produced by:
  //
  // scipy.signal.ellipap(order, pass_band_ripple, stop_band_ripple)
  for (const auto& tc : std::vector<LowPassTestCase>{
           {
               .order = 1,
            .pass_band_ripple = 3.0,
            .stop_band_ripple = 0.0,
            .want_zeros = {},
            .want_poles = {-1.00237729},
            .want_gain = 1.0023772930076005,
           },
           {
               .order = 2,
              .pass_band_ripple = 3.0,
              .stop_band_ripple = 6.0,
              .want_zeros = {{0.0, 1.1684782}, {0.0, -1.1684782}},
              .want_poles = {{-0.12823729, -0.9747527},
                             {-0.12823729, 0.9747527}},
              .want_gain = 0.5011872336272715,
           },
           {
               .order = 2,
              .pass_band_ripple = 6.0,
              .stop_band_ripple = 6.0,
              .want_zeros = {{0.0, 1.36594955}, {0.0, -1.36594955}},
              .want_poles = {{-0.0, -0.82646262}, {-0.0, 0.82646262}},
              .want_gain = 0.18347509339613063,
           },
           {
               .order = 3,
              .pass_band_ripple = 3.0,
              .stop_band_ripple = 3.0,
              .want_zeros = {{0.0, 1.21017886}, {0.0, -1.21017886}},
              .want_poles = {{0.0, 0.0}, {0.0, -0.93284248}, {0.0, 0.93284248}},
              .want_gain = 0.0,
           },
           {
               .order = 4,
              .pass_band_ripple = 1.0,
              .stop_band_ripple = 100.0,
              .want_zeros = {{0.0, 19.46500743},
                             {0.0, 8.08840304},
                             {0.0, -19.46500743},
                             {0.0, -8.08840304}},
              .want_poles = {{-0.33780363, -0.40949047},
                             {-0.13835469, -0.98378183},
                             {-0.33780363, 0.40949047},
                             {-0.13835469, 0.98378183}},
              .want_gain = 9.999999999999996e-06,
           },
       }) {
    ZPKCoeffs coeffs = AnalogPrototypeLowPass(tc.order, tc.pass_band_ripple,
                                              tc.stop_band_ripple);
    ASSERT_THAT(coeffs.zeros, (N(tc.want_zeros, 1e-5)));
    ASSERT_THAT(coeffs.poles, (N(tc.want_poles, 1e-5)));
    ASSERT_NEAR(coeffs.gain, tc.want_gain, 1e-5);
  }
}

struct AnalogBandPassTestCase {
  std::vector<std::complex<double>> lp_zeros;
  std::vector<std::complex<double>> lp_poles;
  double lp_gain;
  double wo;
  double bw;
  std::vector<std::complex<double>> want_zeros;
  std::vector<std::complex<double>> want_poles;
  double want_gain;
};

TEST(Elliptic, AnalogBandPassTest) {
  // Golden data produced by:
  //
  // scipy.signal.lp2bp_zpk(lp_zeros, lp_poles, lp_gain, wo, bw)
  for (const auto& tc : std::vector<AnalogBandPassTestCase>{
           {
               .lp_zeros = {},
            .lp_poles = {-1.00237729},
            .lp_gain = 1.0023772930076005,
            .wo = 100.0,
            .bw = 4.0,
            .want_zeros = {0.0},
            .want_poles = {{-2.00475458, -99.97990278},
                           {-2.00475458, 99.97990278}},
            .want_gain = 4.009509172030402,
           },
           {
               .lp_zeros = {{0.0, 19.46500743},
                            {0.0, 8.08840304},
                            {0.0, -19.46500743},
                            {0.0, -8.08840304}},
              .lp_poles = {{-0.33780363, -0.40949047},
                           {-0.13835469, -0.98378183},
                           {-0.33780363, 0.40949047},
                           {-0.13835469, 0.98378183}},
              .lp_gain = 9.999999999999996e-06,
              .wo = 100.0,
              .bw = 4.0,
              .want_zeros = {{0.0, 146.24052719},
                             {0.0, 117.47680142},
                             {0.0, -146.24052719},
                             {0.0, -117.47680142},
                             {0.0, -68.38049747},
                             {0.0, -85.12318926},
                             {0.0, 68.38049747},
                             {0.0, 85.12318926}},
              .want_poles = {{-0.67007422, 99.18209063},
                             {-0.27126598, 98.05140838},
                             {-0.67007422, -99.18209063},
                             {-0.27126598, -98.05140838},
                             {-0.6811403, -100.82005251},
                             {-0.28215278, -101.9865357},
                             {-0.6811403, 100.82005251},
                             {-0.28215278, 101.9865357}},
              .want_gain = 9.999999999999996e-06,
           },
       }) {
    ZPKCoeffs coeffs = AnalogBandPassFromLowPass(
        {.zeros = tc.lp_zeros, .poles = tc.lp_poles, .gain = tc.lp_gain}, tc.wo,
        tc.bw);
    ASSERT_THAT(coeffs.zeros, (N(tc.want_zeros, 1e-5)));
    ASSERT_THAT(coeffs.poles, (N(tc.want_poles, 1e-5)));
    ASSERT_NEAR(coeffs.gain, tc.want_gain, 1e-5);
  }
}

struct DigitalZPKBandPassTestCase {
  std::vector<std::complex<double>> analog_zeros;
  std::vector<std::complex<double>> analog_poles;
  double analog_gain;
  double sample_rate;
  std::vector<std::complex<double>> want_zeros;
  std::vector<std::complex<double>> want_poles;
  double want_gain;
};

TEST(Elliptic, DigitalZPKBandPassCoeffsTest) {
  // Golden data produced by:
  //
  // scipy.signal.bilinear_zpk(analog_zeros, analog_poles, analog_gain)
  for (const auto& tc : std::vector<DigitalZPKBandPassTestCase>{
           {
               .analog_zeros = {0.0},
            .analog_poles = {{-2.00475458, -99.97990278},
                             {-2.00475458, 99.97990278}},
            .analog_gain = 1.0023772930076005,
            .sample_rate = 48000.0,
            .want_zeros = {{1.0}, {-1.0}},
            .want_poles = {{0.99995607, -0.00208283}, {0.99995607, 0.00208283}},
            .want_gain = 4.176393092455553e-05,
           },
           {
               .analog_zeros = {{0.0, 146.24052719},
                                {0.0, 117.47680142},
                                {0.0, -146.24052719},
                                {0.0, -117.47680142},
                                {0.0, -68.38049747},
                                {0.0, -85.12318926},
                                {0.0, 68.38049747},
                                {0.0, 85.12318926}},
              .analog_poles = {{-0.67007422, 99.18209063},
                               {-0.27126598, 98.05140838},
                               {-0.67007422, -99.18209063},
                               {-0.27126598, -98.05140838},
                               {-0.6811403, -100.82005251},
                               {-0.28215278, -101.9865357},
                               {-0.6811403, 100.82005251},
                               {-0.28215278, 101.9865357}},
              .analog_gain = 9.999999999999996e-06,
              .sample_rate = 48000.0,
              .want_zeros = {{0.99999536, 0.00304667},
                             {0.99999701, 0.00244743},
                             {0.99999536, -0.00304667},
                             {0.99999701, -0.00244743},
                             {0.99999899, -0.00142459},
                             {0.99999843, -0.0017734},
                             {0.99999899, 0.00142459},
                             {0.99999843, 0.0017734}},
              .want_poles = {{0.99998391, 0.00206626},
                             {0.99999226, 0.00204272},
                             {0.99998391, -0.00206626},
                             {0.99999226, -0.00204272},
                             {0.9999836, -0.00210039},
                             {0.99999186, -0.0021247},
                             {0.9999836, 0.00210039},
                             {0.99999186, 0.0021247}},
              .want_gain = 9.999610905675573e-06,
           },
       }) {
    ZPKCoeffs coeffs = DigitalBandPassFromAnalog({.zeros = tc.analog_zeros,
                                                  .poles = tc.analog_poles,
                                                  .gain = tc.analog_gain},
                                                 tc.sample_rate);
    ASSERT_THAT(coeffs.zeros, (N(tc.want_zeros, 1e-5)));
    ASSERT_THAT(coeffs.poles, (N(tc.want_poles, 1e-5)));
    ASSERT_NEAR(coeffs.gain, tc.want_gain, 1e-4);
  }
}

struct DigitalBABandPassTestCase {
  std::vector<std::complex<double>> zeros;
  std::vector<std::complex<double>> poles;
  double gain;
  std::vector<float> want_b_coeffs;
  std::vector<float> want_a_coeffs;
};

TEST(Elliptic, DigitalBABandPassCoeffsTest) {
  // Golden data produced by:
  // scipy.signal.zpk2tf(zeros, poles, gain).
  for (const auto& tc : std::vector<DigitalBABandPassTestCase>{
           {
               .zeros = {{1.0}, {-1.0}},
            .poles = {{0.99995607, -0.00208283}, {0.99995607, 0.00208283}},
            .gain = 4.176393092455553e-05,
            .want_b_coeffs = {4.17639309e-05, 0.00000000e+00, -4.17639309e-05},
            .want_a_coeffs = {1., -1.99991213, 0.99991647},
           },
       }) {
    BACoeffs coeffs =
        BAFromZPK({.zeros = tc.zeros, .poles = tc.poles, .gain = tc.gain});
    ASSERT_THAT(tc.want_b_coeffs, (N(coeffs.b_coeffs, 1e-6)));
    ASSERT_THAT(tc.want_a_coeffs, (N(coeffs.a_coeffs, 1e-6)));
  }
}

struct DigitalSOSBandPassTestCase {
  int order;
  double band_pass_ripple;
  double band_stop_ripple;
  double low_threshold;
  double high_threshold;
  double sample_rate;
  std::vector<std::pair<std::vector<double>, std::vector<double>>>
      want_sections;
};

TEST(Elliptic, DigitalSOSBandPassTest) {
  // Golden data produced by:
  //
  // scipy.signal.ellip(
  //   N=order,
  //   rp=band_pass_ripple,
  //   rs=band_stop_ripple,
  //   Wn=[low_threshold, high_threshold],
  //   btype='bandpass',
  //   analog=False,
  //   fs=sample_rate,
  //   output='sos',
  // )
  for (const auto& tc : std::vector<DigitalSOSBandPassTestCase>{
           {
               .order = 3,
            .band_pass_ripple = 1,
            .band_stop_ripple = 30,
            .low_threshold = 20,
            .high_threshold = 21,
            .sample_rate = 48000,
            .want_sections =
                   {
                       {{9.75081372e-06, 0.00000000e+00, -9.75081372e-06},
                        {1.00000000e+00, -1.99991956e+00, 9.99926757e-01}},
                       {{1.00000000e+00, -1.99999208e+00, 1.00000000e+00},
                        {1.00000000e+00, -1.99996493e+00, 9.99972482e-01}},
                       {{1.00000000e+00, -1.99999346e+00, 1.00000000e+00},
                        {1.00000000e+00, -1.99996692e+00, 9.99973776e-01}},
                   },
           },
           {
               .order = 4,
              .band_pass_ripple = 1,
              .band_stop_ripple = 30,
              .low_threshold = 20,
              .high_threshold = 21,
              .sample_rate = 48000,
              .want_sections =
                   {{{
                         0.03162085,
                         -0.06324145,
                         0.03162085,
                     },
                     {1., -1.99994199, 0.99994938}},
                    {{1., -1.99999368, 1.}, {1., -1.99994368, 0.99995069}},
                    {{1., -1.99999233, 1.}, {1., -1.99998132, 0.99998888}},
                    {{1., -1.99999325, 1.}, {1., -1.99998256, 0.99998941}}},
           },
           {
               .order = 2,
              .band_pass_ripple = 6,
              .band_stop_ripple = 6,
              .low_threshold = 1000,
              .high_threshold = 1500,
              .sample_rate = 4800,
              .want_sections =
                   {
                       {{0.20665649, -0.15002595, 0.20665649},
                        {1., -0.41098111, 1.}},
                       {{1., 0.95335468, 1.}, {1., 0.66709753, 1.}},
                   },
           },
           {
               .order = 2,
              .band_pass_ripple = 6,
              .band_stop_ripple = 6,
              .low_threshold = 1000,
              .high_threshold = 1500,
              .sample_rate = 48000,
              .want_sections =
                   {
                       {{0.18370749, -0.3646966, 0.18370749},
                        {1., -1.98165579, 1.}},
                       {{1., -1.95560494, 1.}, {1., -1.96414337, 1.}},
                   },
           },
           {
               .order = 3,
              .band_pass_ripple = 3,
              .band_stop_ripple = 100,
              .low_threshold = 10000,
              .high_threshold = 10100,
              .sample_rate = 24000,
              .want_sections =
                   {
                       {{7.74075393e-06, 0.00000000e+00, -7.74075393e-06},
                        {1.00000000e+00, 1.73834064e+00, 9.92205921e-01}},
                       {{1.00000000e+00, 1.32825231e+00, 1.00000000e+00},
                        {1.00000000e+00, 1.72988548e+00, 9.96020922e-01}},
                       {{1.00000000e+00, 1.91028195e+00, 1.00000000e+00},
                        {1.00000000e+00, 1.75311014e+00, 9.96185302e-01}},
                   },
           },
       }) {
    const std::vector<BACoeffs> coeffs =
        DigitalSOSBandPass(tc.order, tc.band_pass_ripple, tc.band_stop_ripple,
                           tc.low_threshold, tc.high_threshold, tc.sample_rate);

    CHECK_EQ(coeffs.size(), tc.want_sections.size());
    for (int section_index = 0; section_index < coeffs.size();
         ++section_index) {
      ASSERT_THAT(coeffs[section_index].b_coeffs,
                  (N(tc.want_sections[section_index].first, 1e-7)))
          << "test section_index=" << section_index;
      ASSERT_THAT(coeffs[section_index].a_coeffs,
                  (N(tc.want_sections[section_index].second, 1e-7)))
          << "test section_index=" << section_index;
    }
  }
}

void BM_Design1024Filters(benchmark::State& state) {
  for (auto s : state) {
    for (size_t i = 1; i < 1024; ++i) {
      std::vector<BACoeffs> filter = DigitalSOSBandPass(
          4, 1, 30, static_cast<float>(i), static_cast<float>(i + 1), 48000);
    }
  }
}
BENCHMARK(BM_Design1024Filters);

}  // namespace

}  // namespace zimtohrli
