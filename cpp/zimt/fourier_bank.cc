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
    0.65242584189426134,
    0.65242584189426134,
    0.65242584189426134,
    0.65242584189426134,
    0.65242584189426134,
    0.65242584189426134,
    0.65242584189426134,
    0.65242584189426134,
    0.65242584189426134,
    0.65242584189426134,
    0.65242584189426134,
    0.65367584189426142,
    0.6549258418942614,
    0.65742584189426134,
    0.6599258418942614,
    0.66117584189426137,
    0.66242584189426135,
    0.66642584189426135,
    0.69242584189426136,
    0.7571156116106902,
    0.81184008556393716,
    0.83400282102612622,
    0.83872464138058722,
    0.82927259641675122,
    0.7922079845049772,
    0.68361547130611966,
    0.6179961814310887,
    0.56330289978054571,
    0.50371548074749573,
    0.46905210155514254,
    0.45077889699009471,
    0.4649197115328656,
    0.50957330602673168,
    0.57775565523174865,
    0.69372873920467967,
    0.82085036540717683,
    0.89595151210517476,
    0.93804145122260585,
    0.93079489732474885,
    0.91381806398124277,
    0.86012143159598886,
    0.85268759990424681,
    0.84597581887931383,
    0.89821172158230789,
    0.96513820092190783,
    1.0686156789492478,
    1.1494194308947479,
    1.2035586873721078,
    1.2103145162295275,
    1.1828720609377776,
    1.1303138772677377,
    1.0725480357836679,
    1.0109065810870077,
    0.94178503428736858,
    0.87705985461512759,
    0.85108301793376162,
    0.81279746447840862,
    0.79419109975138857,
    0.7659978047494046,
    0.75998542462626562,
    0.75239930306712266,
    0.76494828616419963,
    0.77919647847453266,
    0.79593250905314561,
    0.81731054936686665,
    0.83616230641642764,
    0.86629311000147957,
    0.88778922711650066,
    0.92272114052517351,
    0.94424127831203852,
    0.96656303896648843,
    0.98518591418011159,
    0.99634342684490584,
    1.007742219111756,
    1.011604123256036,
    1.0164120723222059,
    1.0103486110184858,
    1.006504349849056,
    0.99505692019207603,
    0.99396496309978599,
    0.99209027781231596,
    1.010261111399956,
    1.0380919505570658,
    1.1145308530021252,
    1.1832342660117752,
    1.2643557107139152,
    1.3531965529609251,
    1.4323795354056952,
    1.5105820214439052,
    1.5582600152183153,
    1.5987105881858452,
    1.6105962311912152,
    1.6188304936576452,
    1.6080393111396452,
    1.5897254264045353,
    1.5574878966498953,
    1.5130382096458053,
    1.4567630803275653,
    1.3952583586860252,
    1.3085160513376424,
    1.2590117712473123,
    1.2232129831766325,
    1.2079057951425725,
    1.2078860064005124,
    1.2159276266550225,
    1.2240284885774309,
    1.2215011677833381,
    1.2339762120071642,
    1.2326519859206841,
    1.2331238225001713,
    1.2346901675016944,
    1.2390876900945405,
    1.2380313197195745,
    1.2263786409166506,
    1.2100128929015972,
    1.2059028221811476,
    1.2000269895352882,
    1.1121912322728165,
    1.0107189523983311,
    0.92394826596018476,
    0.76572302581997798,
    0.72497776862591723,
    0.67531295562207327,
    0.70065974273239928,
    0.76556592674908008,
    0.73248175977186947,
    0.58354433953392204,
    0.37525324698643531,
  };
  static const float offmul = 914247.27569635515;
  val += offmul;
  // It is unknown why this works. Perhaps it does something that
  // slightly resembles masking. Perhaps the log(energy) is an overly
  // idealized approximation of the related biological process.
  static const float p = 1.0459967588243515;
  val = pow(val, p);
  return log(val) * kMul[k];
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

  static const float attenuator = 0.89792777092809695;
  static const float reso_val = -0.039805566535150563;
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
	static const float mul0 = 0.037980443862223566;
	static const float mul1 = 2.0843355727615256;
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
    // A big number normalizes better for unknown reasons.
    static const float norm = 8.8263149466024808e11 / downsample;
    static const float norm2 = 1.0261331060796535;
    // Should be 1.0, but this works slightly better.
    static const float freqpow = 2.0;
    const float gain_full = norm;
    const float gainer = sqrt(gain_full);
    gain[i] = gainer * pow(windowM1, 3.0) * Frequency(i) / bw;
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
