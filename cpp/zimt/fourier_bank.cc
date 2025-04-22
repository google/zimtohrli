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
    0.69022944648010776,
    0.68908370797557217,
    0.69206359820200725,
    0.68780644322083062,
    0.68780644322083062,
    0.68780644322083062,
    0.68780644322083062,
    0.68780644322083062,
    0.68780644322083062,
    0.68780644322083062,
    0.68780644322083062,
    0.68913166977173146,
    0.69045689632263207,
    0.69310734942443342,
    0.69575780252623487,
    0.69565436190852481,
    0.69697958845942554,
    0.70122031342230773,
    0.72878502568104242,
    0.79911765343502128,
    0.85713551412855471,
    0.88063203050861982,
    0.88563801587047197,
    0.8756171351133657,
    0.81948646752340204,
    0.70435872214727346,
    0.63479038199236093,
    0.58382539388061494,
    0.52065193003881982,
    0.48390246567910344,
    0.46452955699139042,
    0.47952138329814464,
    0.52686228651128453,
    0.63677358244745352,
    0.75972607050406527,
    0.89449803389384408,
    0.97411886077966425,
    1.0187418246550353,
    1.011059164152657,
    0.99306064391503246,
    0.93613248157031814,
    0.92825127266405905,
    0.93149312346243962,
    0.98687264760025817,
    1.0578268455035744,
    1.1646166464383585,
    1.2502832684308152,
    1.3076806925346713,
    1.3148430955548263,
    1.2857491192552004,
    1.2300279188818783,
    1.153360587802494,
    1.0880094738823687,
    1.0147281066358029,
    0.94610768534502521,
    0.91856753040217931,
    0.87797790483820981,
    0.85825178599661844,
    0.82836178352141432,
    0.82198757090278118,
    0.81394490713595202,
    0.82724910360559234,
    0.84235476980718638,
    0.86009799547075749,
    0.88276259277473246,
    0.89349396108823076,
    0.92543807381693866,
    0.94822785393061615,
    0.98526201322896612,
    0.99730625610843238,
    1.0209713680141346,
    1.0407149909638977,
    1.0525439765842144,
    1.0646287623123996,
    1.0687230806396262,
    1.0738203780459195,
    1.0673920101178804,
    1.0633163965416204,
    1.0511800463648222,
    1.0500223739399046,
    1.048034867765874,
    1.0672992446637448,
    1.0968049782516238,
    1.1520844963349179,
    1.2249225659812257,
    1.3263061222587622,
    1.4204935166188568,
    1.5044418291911075,
    1.5873506378665294,
    1.6519986642135718,
    1.6948836028501453,
    1.7074845385983402,
    1.7452582883596941,
    1.6876038428165716,
    1.6681878057756896,
    1.6340101814826493,
    1.5513624830901918,
    1.4917006466678702,
    1.4264944945677696,
    1.3345319275401968,
    1.2861895163992125,
    1.2652320739631016,
    1.2490036803612088,
    1.2489827007080991,
    1.2786418125466377,
    1.2872301943904423,
    1.2845507762952622,
    1.297776584158522,
    1.296372664503163,
    1.2968728967934211,
    1.2985335063805323,
    1.3031956773391127,
    1.3020757332843207,
    1.2683519522235636,
    1.2510012931876442,
    1.2466438733117888,
    1.2404144257670351,
    1.1787433244931078,
    1.071164316917032,
    0.97917166291086322,
    0.79221781234772859,
    0.74613220534023617,
    0.68013695325392454,
    0.73080278659266906,
    0.78908146264315304,
    0.75786785492255149,
    0.60637240054507502,
    0.3787081741944886,
  };
  // Offmul is an offset that tries to model the noise threshold.
  // It works as a soft threshold to make low values less important,
  // and particularly so after the logarithm. It can be considered
  // as a base energy noise, such as the thermal noise or blood circulation
  // noise in the ear.
  static const float offmul = 912654.14916639519;
  val += offmul;
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

  for (int64_t ii = 0;
       ii < signal.size() && out_ix < channels.shape()[0];
       ii += downsampling, ++out_ix) {
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
        } else {
          fprintf(stderr,
                  "strange thing #17 happened in FilterAndDownsample\n");
	  return;
	}
        if (out_ix + 1 != channels.shape()[0] - 1) {
          fprintf(stderr,
                  "strange thing #9831021 happened in FilterAndDownsample\n");
        }
        return;
      }
      // Outer ear modeling.
      static const double kernel[32] = {
	-0.00756885973519324,
	0.00413482141974997,
	-0.00000236200210516,
	0.00619875373288928,
	-0.00283612301349015,
	-0.00000418032688396,
	-0.00653942799073076,
	-0.00697059266809395,
	0.00344293224958738,
	0.00329933933698966,
	-0.00298496041436849,
	0.00350131041814130,
	0.00171017251385591,
	-0.00154158276443512,
	0.00404768079279660,
	0.00127457555980521,
	-0.01171138281621471,
	-0.00010813847460362,
	-0.00152608046744481,
	-0.00838915828792046,
	-0.00640430929472267,
	-0.00086448874278805,
	-0.00720815920453227,
	0.00344734180928985,
	-0.00294620320133006,
	0.00079453551775897,
	0.00067657883664455,
	0.00185866424809109,
	0.00615985137470771,
	-0.00236233239399860,
	-0.00680980952401827,
	0.01082403830708164,
      };
      static const double kernel2[32] = {
	-0.10104347418401896,
	-0.11826972031644883,
	-0.06180710258042214,
	0.07855591921843502,
	0.03670823911837065,
	-0.01840452136433212,
	0.10859856308297887,
	0.16449286025656040,
	0.06054576192365510,
	0.08362268315743795,
	-0.00320242077511887,
	0.17410886426799152,
	-0.13348931125572602,
	0.12798560564215689,
	0.02840772721429061,
	0.01655141242672416,
	0.00565097497020243,
	-0.39669214512177614,
	0.25126981719959374,
	0.29050002107255307,
	-0.34990576312818494,
	0.13135342797565053,
	1.09071850579315455,
	-0.97998963695018926,
	-0.97386487573440028,
	0.30687938104930101,
	0.52811340907235282,
	1.35094332106162751,
	0.35339301883185964,
	-0.17657465769141492,
	0.36698233014139547,
	-0.39494225991497944,
      };
      if (input_ix >= 32) {
	float signalval = 0;
	float signalval_linear = 0;
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
	  float b = 
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
	  signalval_linear += b;
	}
	float signalval_massful_spring = r.Update(signalval);
	IncrementAll(signalval_massful_spring + signalval_linear);
      }
      if (out_ix + 1 < channels.shape()[0]) {
	for (int k = 0; k < kNumRotators; ++k) {
	  float energy =
	    channel.accu[4][k] * channel.accu[4][k] +
	    channel.accu[5][k] * channel.accu[5][k];
	  channels[{out_ix + 1}][k] += one_minus_weight * energy;
	  channels[{out_ix}][k] += weight * energy;
	}
      } else {
	for (int k = 0; k < kNumRotators; ++k) {
	  float energy =
	    channel.accu[4][k] * channel.accu[4][k] +
	    channel.accu[5][k] * channel.accu[5][k];
	  channels[{out_ix}][k] += energy;
	}
      }
    }
    FinalizeDb(channels, out_ix);
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
