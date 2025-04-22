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
    0.65104853771133298,
    0.64996783710984263,
    0.65277857371975445,
    0.6487630763522636,
    0.6487630763522636,
    0.6487630763522636,
    0.6487630763522636,
    0.6487630763522636,
    0.6487630763522636,
    0.6487630763522636,
    0.6487630763522636,
    0.65001307635226369,
    0.65126307635226366,
    0.65376307635226361,
    0.65626307635226366,
    0.65616550754634917,
    0.65741550754634914,
    0.66141550754634915,
    0.68741550754634917,
    0.75375570019697258,
    0.80848017415021955,
    0.83064290961240861,
    0.83536472996686961,
    0.82591268500303361,
    0.77296827754321662,
    0.66437576434435908,
    0.59875647446932811,
    0.55068451643590044,
    0.49109709740285029,
    0.456433718210497,
    0.43816051364544917,
    0.45230132818822005,
    0.49695492268208624,
    0.6006270984522093,
    0.71660018242514023,
    0.84372180862763746,
    0.91882295532563529,
    0.96091289444306649,
    0.95366634054520949,
    0.93668950720170341,
    0.8829928748164495,
    0.87555904312470734,
    0.87861687010170342,
    0.93085277280469736,
    0.99777925214429741,
    1.0985071247316331,
    1.1793108766771332,
    1.2334501331544931,
    1.2402059620119128,
    1.212763506720163,
    1.160205323050123,
    1.0878900168225984,
    1.0262485621259383,
    0.95712701532629918,
    0.89240183565405817,
    0.86642499897269221,
    0.8281394455173392,
    0.80953308079031916,
    0.78133978578833518,
    0.7753274056651962,
    0.76774128410605325,
    0.78029026720313022,
    0.79453845951346325,
    0.81127449009207619,
    0.83265253040579723,
    0.84277473206464248,
    0.87290553564969442,
    0.8944016527647155,
    0.92933356617338836,
    0.94069411700833838,
    0.96301587766278829,
    0.98163875287641145,
    0.99279626554120548,
    1.0041950578080556,
    1.0080569619523356,
    1.0128649110185055,
    1.0068014497147855,
    1.0029571885453556,
    0.99150975888837589,
    0.99041780179608585,
    0.98854311650861582,
    1.0067139500962556,
    1.0345447892533655,
    1.0866863627504808,
    1.1553897757601308,
    1.2510182894363682,
    1.3398591316833781,
    1.4190421141281482,
    1.4972446001663582,
    1.5582228780908987,
    1.5986734510584286,
    1.6105590940637986,
    1.6461886150461551,
    1.5918069269642785,
    1.5734930422291686,
    1.5412555124745286,
    1.463299314781118,
    1.407024185462878,
    1.3455194638213379,
    1.2587771564729551,
    1.2131789046984491,
    1.193411112521833,
    1.1781039244877729,
    1.1780841357457128,
    1.2060596466294748,
    1.2141605085518832,
    1.2116331877577904,
    1.2241082319816166,
    1.2227840058951365,
    1.2232558424746236,
    1.2248221874761467,
    1.2292197100689928,
    1.2281633396940268,
    1.1963538907380762,
    1.1799881427230228,
    1.1758780720025732,
    1.1700022393567138,
    1.1118319012058406,
    1.0103596213313553,
    0.92358893489320881,
    0.74724752893127167,
    0.70377797369166939,
    0.64152894536377958,
    0.68931873015973488,
    0.74428921427325723,
    0.71484744854328919,
    0.57195164114859187,
    0.3572107859002453,
  };
  static const float offmul = 912654.14916639519;
  val += offmul;
  // It is unknown why this works. Perhaps it does something that
  // slightly resembles masking. Perhaps the log(energy) is an overly
  // idealized approximation of the related biological process.
  static const float p = 1.0601812407205606;
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

  static const float attenuator = 0.92532302944402367;
  static const float reso_val = -0.040025628124382151;
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
	-0.20310840849899861,
	0.11095687162599342,
	-6.3383720300776245e-05,
	0.16634196555529437,
	-0.076106633195873602,
	-0.00011217799907935894,
	-0.17548387183280492,
	-0.18705406529444729,
	0.092390203313196528,
	0.088536924355753374,
	-0.080100646650365731,
	0.093956766483992218,
	0.045892040505509234,
	-0.041367977847182234,
	0.10861847526573196,
	0.034202908036470171,
	-0.31427195222863508,
	-0.0029018682130051111,
	-0.040951977686003303,
	-0.22512090963754727,
	-0.17185799630269605,
	-0.023198333547727951,
	-0.19342910232997981,
	0.092508532716129632,
	-0.079060606785247289,
	0.021321156707078688,
	0.018155819442130232,
	0.049876778084345305,
	0.16529803075750987,
	-0.063392583516866288,
	-0.18273949088704589,
	0.29046029005681101,
      };
      static const double kernel2[32] = {
	-0.048033684161467298,
	-0.05622263523124061,
	-0.029381638628899764,
	0.037343630978880542,
	0.017450230983984787,
	-0.0087490753212553889,
	0.051625195210696882,
	0.078196025623490026,
	0.028782027033827075,
	0.0397522510380399,
	-0.0015223552961375687,
	0.082767247103259153,
	-0.063457669756207424,
	0.060841337931765008,
	0.013504363421543635,
	0.0078681510444065343,
	0.0026863402027202969,
	-0.18857808841184936,
	0.11944774401456407,
	0.13809685755347414,
	-0.16633694603341997,
	0.062442321227176627,
	0.51850184922305576,
	-0.46586395690481303,
	-0.46295238989685533,
	0.14588321892162398,
	0.25105265726135589,
	0.64220658808444908,
	0.16799470513575968,
	-0.083939427132346825,
	0.17445474318062471,
	-0.18774623426173354,
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
	static const float mul0 = 0.037265122557594954;
	static const float mul1 = 2.1035961731429338;
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
    // The bandwidth variable relates to the frequency shape overlap (and 
    // window length) of the triple leaking complex-number integrator
    // (3rd-order complex gammatone filter).
    float bandwidth = CalculateBandwidth(
       i == 0 ? Frequency(1) : Frequency(i - 1), Frequency(i),
       i + 1 == kNumRotators ? Frequency(i - 1) : Frequency(i + 1));
    window[i] = std::pow(kWindow, bandwidth * kBandwidthMagic);
    float windowM1 = 1.0f - window[i];
    float f = Frequency(i) * 2.0f * M_PI / kSampleRate;
    // A big number normalizes better for unknown reasons.
    static const float norm = 905697397139.4474 / downsample;
    // Should be 1.0, but this works slightly better.
    const float gain_full = norm;
    const float gainer = sqrt(gain_full);
    gain[i] = gainer * pow(windowM1, 3.0) * Frequency(i) / bandwidth;
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
