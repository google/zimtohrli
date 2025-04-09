#include "fourier_bank.h"

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
#include "sndfile.hh"

namespace tabuli {

float Loudness(int k, float val) {
  static const float kMul[128] = {
    0.7,
    0.7,
    0.7,
    0.7,
    0.7,
    0.7,
    0.7,
    0.7,
    0.7,
    0.7,
    0.7,
    0.70125,
    0.7025,
    0.705,
    0.7075,
    0.70875,
    0.71,
    0.714,
    0.72,
    0.756663739681244,
    0.791388213634491,
    0.81355094909668,
    0.818272769451141,
    0.808820724487305,
    0.771756112575531,
    0.729331910610199,
    0.663712620735168,
    0.609019339084625,
    0.549431920051575,
    0.514768540859222,
    0.496495336294174,
    0.510636150836945,
    0.555289745330811,
    0.623472094535828,
    0.709445178508759,
    0.792715966701508,
    0.857817113399506,
    0.899907052516937,
    0.89266049861908,
    0.875683665275574,
    0.82198703289032,
    0.814553201198578,
    0.807841420173645,
    0.86853551864624,
    0.93546199798584,
    1.03893947601318,
    1.11974322795868,
    1.17388248443604,
    1.18063831329346,
    1.15319585800171,
    1.10063767433167,
    1.0428718328476,
    0.98123037815094,
    0.939120948314667,
    0.894395768642426,
    0.86841893196106,
    0.830133378505707,
    0.811527013778687,
    0.783333718776703,
    0.777321338653564,
    0.769735217094421,
    0.782284200191498,
    0.796532392501831,
    0.813268423080444,
    0.834646463394165,
    0.853498220443726,
    0.883629024028778,
    0.905125141143799,
    0.931883990764618,
    0.953404128551483,
    0.975725889205933,
    0.994348764419556,
    1.00550627708435,
    1.0169050693512,
    1.02076697349548,
    1.02557492256165,
    1.01951146125793,
    1.0156672000885,
    1.00421977043152,
    1.00312781333923,
    1.00125312805176,
    1.0194239616394,
    1.04725480079651,
    1.10146355628967,
    1.17016696929932,
    1.25128841400146,
    1.34012925624847,
    1.41931223869324,
    1.49751472473145,
    1.54519271850586,
    1.58564329147339,
    1.59752893447876,
    1.60576319694519,
    1.59497201442719,
    1.57665812969208,
    1.54442059993744,
    1.49997091293335,
    1.44369578361511,
    1.38219106197357,
    1.32069706916809,
    1.27119278907776,
    1.23539400100708,
    1.22008681297302,
    1.22006702423096,
    1.22810864448547,
    1.2362095064078784,
    1.2336821856137856,
    1.2461572298376118,
    1.2448330037511317,
    1.2453048403306188,
    1.2468711853321419,
    1.251268707924988,
    1.250212337550022,
    1.2485596587470982,
    1.2121939107320447,
    1.2180838400115952,
    1.2122080073657358,
    1.113969563990006,
    1.0124972841155206,
    0.92572659767737442,
    0.76750135753716764,
    0.72675610034310689,
    0.67197913320699598,
    0.70243807444958895,
    0.76734425846626975,
    0.73426009148905913,
    0.58639078360761954,
    0.37335527143387232,
  };
  static const float off = exp(80) * 1e-10;
  return log(val + off) * kMul[k];
}


void FinalizeDb(hwy::AlignedNDArray<float, 2>& channels, size_t out_ix) {
  for (int k = 0; k < kNumRotators; ++k) {
    channels[{out_ix}][k] = Loudness(k, channels[{out_ix}][k]);
    if (channels[{out_ix}][k] < 0) {
      // Nsim gets confused from negative numbers. This doesn't seem
      // to change anything with usual normalization, but guarantees
      // non-negative values.
      channels[{out_ix}][k] = 0;
    }
  }
}

// Ear drum and other receiving mass-spring objects are
// modeled through the Resonator. Resonator is a non-linear process
// and does complex spectral shifting of energy.
struct Resonator {
  float acc0 = 0;
  float acc1 = 0;
  float attenuator_ = 0;
  float resonator_ = 0;
  Resonator(float attenuator, float resonator) {
    attenuator_ = attenuator;
    resonator_ = resonator;
  }
  float Update(float signal) {
    acc0 *= attenuator_;
    acc0 += signal + resonator_ * acc1;
    acc1 += acc0;
    return acc0;
  }
};

std::vector<float> CreateWeightedWindow(int downsampling) {
  std::vector<float> retval(downsampling);
  const float two_per_downsampling = 2.0 / downsampling;
  static const float scale = 7.944646094630226;
  for (int i = 0; i < downsampling; ++i) {
    float t = two_per_downsampling * (i + 0.5) - 1.0;
    retval[i] = 1.0 / (1.0 + exp(t * scale));
  }
  return retval;
}
  
void Rotators::FilterAndDownsample(hwy::Span<const float> signal,
                                   hwy::AlignedNDArray<float, 2>& channels,
                                   int downsampling) {
  for (size_t zz = 0; zz < channels.shape()[0]; zz++) {
    for (int k = 0; k < kNumRotators; ++k) {
      channels[{zz}][k] = 0;
    }
  }
  const std::vector<float> weights = CreateWeightedWindow(downsampling);

  static const float attenuator = 0.90462356593345827;
  static const float reso_val = -0.040121175888269377;
  Resonator r(attenuator, reso_val);

  size_t out_ix = 0;

  for (int64_t ii = 0; ii + downsampling < signal.size(); ii += downsampling) {
    OccasionallyRenormalize();
    for (int64_t zz = 0; zz < downsampling; ++zz) {
      float weight = weights[zz];
      float one_minus_weight = 1.0 - weight;
      int64_t input_ix = ii + zz;
      if (input_ix >= signal.size()) {
        if (out_ix + 1 < channels.shape()[0]) {
          FinalizeDb(channels, out_ix);
          FinalizeDb(channels, out_ix + 1);
        } else if (out_ix < channels.shape()[0]) {
          FinalizeDb(channels, out_ix);
        }
        if (out_ix + 1 != channels.shape()[0] - 1) {
          fprintf(stderr,
                  "strange thing #9831021 happened in FilterAndDownsample\n");
        }
        return;
      }
      // Outer ear modeling.
      static const double kernel[32] = {
	-0.20263428271281644,
	0.10915979165019389,
	-0.00028745702813933729,
	0.16665601143299927,
	-0.077109948817528556,
	0.0016939747823971956,
	-0.17495689013925,
	-0.18799101662281015,
	0.093828927128163625,
	0.088899024830657578,
	-0.082421273478358703,
	0.093285522408723967,
	0.046727776543880793,
	-0.043231990170718486,
	0.1136826325065256,
	0.035026890053262601,
	-0.31547592207063974,
	-7.5305276952516596e-05,
	-0.040734411011513975,
	-0.22343618655817049,
	-0.17419325091414362,
	-0.020102223073211582,
	-0.19373987309196436,
	0.093639112957862342,
	-0.077525642076473389,
	0.01956287056199759,
	0.018068454697104462,
	0.051125555893398733,
	0.17058381995819644,
	-0.063329032174762032,
	-0.1819749094019717,
	0.29087441598238489,
      };
      static const double kernel2[32] = {
	-0.049593652768953063,
	-0.05543739572245706,
	-0.030516284665381813,
	0.037169622842466908,
	0.016405461972259276,
	-0.0097079565031696208,
	0.054780287391335468,
	0.078588909984566094,
	0.029026209252923485,
	0.039066004968758054,
	-0.0031947923160080402,
	0.091634324720553398,
	-0.065567402400926733,
	0.06035949759044297,
	0.015075884900543632,
	0.0061988983137513205,
	-2.9835073485631671e-05,
	-0.18757765498411677,
	0.1213122548935063,
	0.13773137126042886,
	-0.16744444532289016,
	0.0624457210913111,
	0.51808806761265092,
	-0.46433206319969683,
	-0.46213071656390931,
	0.14568103575648977,
	0.25089814997851162,
	0.64414906304956598,
	0.1680294597208388,
	-0.084688545530498469,
	0.17692762138780532,
	-0.19290207811783072,
      };
      if (input_ix >= 32) {
	float signalval = 0;
	for (int ii = 0; ii < 32; ii += 16) {
	  int k = input_ix - 32 + ii;
	  float a = 
	    signal[k + 0] * kernel[ii + 0] +
	    signal[k + 1] * kernel[ii + 1] +
	    signal[k + 2] * kernel[ii + 2] +
	    signal[k + 3] * kernel[ii + 3] +
	    signal[k + 4] * kernel[ii + 4] +
	    signal[k + 5] * kernel[ii + 5] +
	    signal[k + 6] * kernel[ii + 6] +
	    signal[k + 7] * kernel[ii + 7] +
	    signal[k + 8] * kernel[ii + 8] +
	    signal[k + 9] * kernel[ii + 9] +
	    signal[k + 10] * kernel[ii + 10] +
	    signal[k + 11] * kernel[ii + 11] +
	    signal[k + 12] * kernel[ii + 12] +
	    signal[k + 13] * kernel[ii + 13] +
	    signal[k + 14] * kernel[ii + 14] +
	    signal[k + 15] * kernel[ii + 15];
	  signalval += a;
	}
	float sum_acc1 = r.Update(signalval);
	float signalval_linear = 0;
	for (int ii = 0; ii < 32; ii += 16) {
	  int k = input_ix - 32 + ii;
	  float a = 
	    signal[k + 0] * kernel2[ii + 0] +
	    signal[k + 1] * kernel2[ii + 1] +
	    signal[k + 2] * kernel2[ii + 2] +
	    signal[k + 3] * kernel2[ii + 3] +
	    signal[k + 4] * kernel2[ii + 4] +
	    signal[k + 5] * kernel2[ii + 5] +
	    signal[k + 6] * kernel2[ii + 6] +
	    signal[k + 7] * kernel2[ii + 7] +
	    signal[k + 8] * kernel2[ii + 8] +
	    signal[k + 9] * kernel2[ii + 9] +
	    signal[k + 10] * kernel2[ii + 10] +
	    signal[k + 11] * kernel2[ii + 11] +
	    signal[k + 12] * kernel2[ii + 12] +
	    signal[k + 13] * kernel2[ii + 13] +
	    signal[k + 14] * kernel2[ii + 14] +
	    signal[k + 15] * kernel2[ii + 15];
	  signalval_linear += a;
	}
	static const float mul0 = 0.039194445043162718;
	static const float mul1 = 1.9985255274046856;
	sum_acc1 *= mul0;
	sum_acc1 += mul1 * signalval_linear;
	IncrementAll(sum_acc1);
      }
      if (zz == 0) {
        for (int k = 0; k < kNumRotators; ++k) {
          float energy = (channel.accu[4][k] * channel.accu[4][k] +
			  channel.accu[5][k] * channel.accu[5][k]);
	  // + 1 should be enough ?!
          if (out_ix + 2 < channels.shape()[0]) {
            channels[{out_ix + 1}][k] += one_minus_weight * energy;
          }
          channels[{out_ix}][k] += weight * energy;
        }
      } else {
        for (int k = 0; k < kNumRotators; ++k) {
          float energy =
              channel.accu[4][k] * channel.accu[4][k] +
              channel.accu[5][k] * channel.accu[5][k];
	  // + 1 should be enough ?!
          if (out_ix + 2 < channels.shape()[0]) {
            channels[{out_ix + 1}][k] += one_minus_weight * energy;
          }
          channels[{out_ix}][k] += weight * energy;
        }
      }
    }
    FinalizeDb(channels, out_ix);
    ++out_ix;
    if (out_ix + 1 >= channels.shape()[0]) {
      return;
    }
  }
}

double CalculateBandwidth(double low, double mid, double high) {
  return (std::abs(std::sqrt(low * mid) - mid) + 
	  std::abs(std::sqrt(high * mid) - mid));
}

float Frequency(int i) {
  static const float kFreq[128] = {
    24.3492317199707,
    33.1997528076172,
    42.3596649169922,
    51.8397789001465,
    61.651294708252,
    71.805793762207,
    82.3152618408203,
    93.1921081542969,
    104.449180603027,
    116.099769592285,
    128.157623291016,
    140.636993408203,
    153.552581787109,
    166.919677734375,
    180.754058837891,
    195.072021484375,
    209.890518188477,
    225.227020263672,
    241.099639892578,
    257.527130126953,
    274.528869628906,
    292.124938964844,
    310.336120605469,
    329.183898925781,
    348.690612792969,
    368.879211425781,
    389.773559570312,
    411.398254394531,
    433.778900146484,
    456.941955566406,
    480.914703369141,
    505.725463867188,
    531.403564453125,
    557.979248046875,
    585.484008789062,
    613.9501953125,
    643.411499023438,
    673.902709960938,
    705.459716796875,
    738.119873046875,
    771.921875,
    806.905395507812,
    843.111938476562,
    880.584045410156,
    919.366088867188,
    959.50390625,
    1001.04486083984,
    1044.03784179688,
    1088.53369140625,
    1134.5849609375,
    1182.24609375,
    1231.5732421875,
    1282.62463378906,
    1335.46069335938,
    1390.14392089844,
    1446.73864746094,
    1505.31176757812,
    1565.93225097656,
    1628.671875,
    1693.60473632812,
    1760.80749511719,
    1830.35961914062,
    1902.34301757812,
    1976.84252929688,
    2053.9462890625,
    2133.74560546875,
    2216.33447265625,
    2301.81030273438,
    2390.27416992188,
    2481.83056640625,
    2576.58715820312,
    2674.65625,
    2776.15380859375,
    2881.19970703125,
    2989.9169921875,
    3102.43505859375,
    3218.88623046875,
    3339.408203125,
    3464.14331054688,
    3593.23876953125,
    3726.84765625,
    3865.12646484375,
    4008.2392578125,
    4156.35498046875,
    4309.64794921875,
    4468.30029296875,
    4632.498046875,
    4802.43603515625,
    4978.314453125,
    5160.3408203125,
    5348.72998046875,
    5543.705078125,
    5745.4970703125,
    5954.341796875,
    6170.4873046875,
    6394.1884765625,
    6625.70947265625,
    6865.3251953125,
    7113.31689453125,
    7369.978515625,
    7635.61279296875,
    7910.53125,
    8195.0615234375,
    8489.5390625,
    8794.30859375,
    9109.732421875,
    9436.18359375,
    9774.0458984375,
    10123.7197265625,
    10485.6171875,
    10860.1640625,
    11247.806640625,
    11648.998046875,
    12064.2138671875,
    12493.9453125,
    12938.7001953125,
    13399.0,
    13875.392578125,
    14368.4384765625,
    14878.7177734375,
    15406.8349609375,
    15953.416015625,
    16519.10546875,
    17104.56640625,
    17710.494140625,
    18337.60546875,
    18986.634765625,
    19658.35546875,
  };
  return kFreq[i];
}
  
Rotators::Rotators(int downsample) {
  const float kSampleRate = 48000.0;
  const float scaling_for_downsampling = 1.0f / downsample;
  static const double kWindow = 0.9996028710680265;
  static const double kBandwidthMagic = 0.7328516996032982;
  for (int i = 0; i < kNumRotators; ++i) {
    // The bw parameter relates to the frequency shape overlap and window length
    // of the triple leaking integrator (3rd-order complex gammatone filter).
    float bw = CalculateBandwidth(
       i == 0 ? Frequency(1) : Frequency(i - 1), Frequency(i),
       i + 1 == kNumRotators ? Frequency(i - 1) : Frequency(i + 1));
    window[i] = std::pow(kWindow, bw * kBandwidthMagic);
    float windowM1 = 1.0f - window[i];
    float f = Frequency(i) * 2.0f * M_PI / kSampleRate;
    static const float full_scale_sine_db = exp(80);
    static const float scale_normalizer = 0.00019;
    const float gainer = sqrt(scaling_for_downsampling * full_scale_sine_db * scale_normalizer * Frequency(i));
    gain[i] = gainer * pow(windowM1, 3.0);
    rot[0][i] = float(std::cos(f));
    rot[1][i] = float(-std::sin(f));
    rot[2][i] = gain[i];
    rot[3][i] = 0.0f;
  }
}

void Rotators::OccasionallyRenormalize() {
  for (int i = 0; i < kNumRotators; ++i) {
    float norm = gain[i] / sqrt(rot[2][i] * rot[2][i] + rot[3][i] * rot[3][i]);
    rot[2][i] *= norm;
    rot[3][i] *= norm;
  }
}

void Rotators::IncrementAll(float signal) {
  for (int i = 0; i < kNumRotators; i++) {
    const float w = window[i];
    channel.accu[0][i] *= w;
    channel.accu[1][i] *= w;
    channel.accu[2][i] *= w;
    channel.accu[3][i] *= w;
    channel.accu[4][i] *= w;
    channel.accu[5][i] *= w;
    const float tr = rot[0][i] * rot[2][i] - rot[1][i] * rot[3][i];
    const float tc = rot[0][i] * rot[3][i] + rot[1][i] * rot[2][i];
    rot[2][i] = tr;
    rot[3][i] = tc;
    channel.accu[2][i] += channel.accu[0][i];
    channel.accu[3][i] += channel.accu[1][i];
    channel.accu[4][i] += channel.accu[2][i];
    channel.accu[5][i] += channel.accu[3][i];
    channel.accu[0][i] += rot[2][i] * signal;
    channel.accu[1][i] += rot[3][i] * signal;
  }
}

}  // namespace tabuli
