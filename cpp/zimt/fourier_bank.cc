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

float Loudness(float freq, float val) {
  static const float pars[30][4] = {
    // ~[20*1.2589254117941673**i for i in range(31)]
    { 0.635, -31.5, 78.1, 20 },
    { 0.602, -27.2, 68.7, 25 },
    { 0.569, -23.1, 59.5, 31.5 },
    { 0.537, -19.3, 51.1, 40 },
    { 0.509, -16.1, 44.0, 50 },
    { 0.482, -13.1, 37.5, 63 },
    { 0.456, -10.4, 31.5, 80 },
    { 0.433, -8.2, 26.5, 100 },
    { 0.412, -6.3, 22.1, 125 },
    { 0.391, -4.6, 17.9, 160 },
    { 0.373, -3.2, 14.4, 200 },
    { 0.357, -2.1, 11.4, 250 },
    { 0.343, -1.2, 8.6, 315 },
    { 0.330, -0.5, 6.2, 400 },
    { 0.320, 0.0, 4.4, 500 },
    { 0.311, 0.4, 3.0, 630 },
    { 0.303, 0.5, 2.2, 800 },
    { 0.300, 0.0, 2.4, 1000 },
    { 0.295, -2.7, 3.5, 1250 },
    { 0.292, -4.2, 1.7, 1600 },
    { 0.290, -1.2, -1.3, 2000 },
    { 0.290, 1.4, -4.2, 2500 },
    { 0.289, 2.3, -6.0, 3150 },
    { 0.289, 1.0, -5.4, 4000 },
    { 0.289, -2.3, -1.5, 5000 },
    { 0.293, -7.2, 6.0, 6300 },
    { 0.303, -11.2, 12.6, 8000 },
    { 0.323, -10.9, 13.9, 10000 },
    { 0.354, -3.5, 12.3, 12500 },
  };
  int low_ix = 0;
  int high_ix = 0;
  float interp = 0;
  int start_kk = 0;
  if (freq > 200) start_kk = 10;
  if (freq > 2000) start_kk = 20;
  for (int kk = start_kk; kk < 30; ++kk) {
    if (freq >= pars[kk][3] && (kk == 29 || freq < pars[kk + 1][3])) {
      low_ix = kk;
      high_ix = std::min(kk + 1, 29);
      if (low_ix == high_ix) {
        interp = 0;
      } else {
        interp = (freq - pars[low_ix][3]) / (pars[high_ix][3] - pars[low_ix][3]);
      }
      break;
    }
  }
  const float *vals = &pars[low_ix][0];
  const float *valsNext = &pars[high_ix][0];
  float vals1 = (1.0 - interp) * vals[1] + interp * valsNext[1];
  float vals2 = (1.0 - interp) * vals[2] + interp * valsNext[2];
  static const float constant1 = 46.37287;
  static const float constant2 = 80.892916;
  static const float kMul = 3.41562;
  val *= (constant1 + vals1) * (1.0 / constant2);
  val += kMul * vals2;
  return val;
}

float SimpleDb(float energy) {
  // ideally 78.3 db, but somehow this works better
  static const float full_scale_sine_db = 75.8495;
  static const float exp_full_scale_sine_db = exp(full_scale_sine_db);
  // epsilon, but the biggest one you saw (~4.95e23)
  static const float epsilon = 1.0033294789821357e-09 * exp_full_scale_sine_db;
  // kMul allows faster log instead of log10 below, incorporating multiplying by 10 for decibel.
  static const float kMul0 = 7.7888;
  static const float kMul = kMul0 / log(10);
  return kMul * log(energy + epsilon);
}

void PrepareMasker(hwy::AlignedNDArray<float, 2>& freq,
                   float *masker,
                   size_t out_ix) {
  static const double kMul = 0.925585;
  // convolve in time and freq, 5 freq bins, 3 time bins
  static const double c[12] = {
    0.011792864242897921,
    0.02141444616229312,
    0.27204462940133883,
    0.04081130408323199,
    0.32503380520713393,
    0.63960360145409523,
    0.36547565427708706,
    0.64646959791651382,
    1.5271657112757793,
    0.10296035397317034,
    1.3755662770950172,
    7.4439086735915927,
  };
  static const float div = 1.0 / (2*(c[0]+c[1]+c[2]+c[3]+c[4]+c[5]+c[6]+c[7]+c[8])+c[9]+c[10]+c[11]);
  const size_t oi2 = std::max<size_t>(2, out_ix) - 2;
  const size_t oi1 = std::max<size_t>(1, out_ix) - 1;
  const size_t oi0 = out_ix;
  for (int k = 0; k < kNumRotators; ++k) {
    int prev3 = std::max(0, k - 3);
    int prev2 = std::max(0, k - 2);
    int prev1 = std::max(0, k - 1);
    int currk = k;
    int next1 = std::min<int>(kNumRotators - 1, k + 1);
    int next2 = std::min<int>(kNumRotators - 1, k + 2);
    int next3 = std::min<int>(kNumRotators - 1, k + 3);
    float v =
        freq[{oi2}][prev3] * c[0] + freq[{oi1}][prev3] * c[1] + freq[{oi0}][prev3] * c[2] +
        freq[{oi2}][prev2] * c[3] + freq[{oi1}][prev2] * c[4] + freq[{oi0}][prev2] * c[5] +
        freq[{oi2}][prev1] * c[6] + freq[{oi1}][prev1] * c[7] + freq[{oi0}][prev1] * c[8] +
        freq[{oi2}][currk] * c[9] + freq[{oi1}][currk] * c[10] + freq[{oi0}][currk] * c[11] +
        freq[{oi2}][next1] * c[6] + freq[{oi1}][next1] * c[7] + freq[{oi0}][next1] * c[8] +
        freq[{oi2}][next2] * c[3] + freq[{oi1}][next2] * c[4] + freq[{oi0}][next2] * c[5] +
        freq[{oi2}][next3] * c[0] + freq[{oi1}][next3] * c[1] + freq[{oi0}][next3] * c[2];
    masker[k] = v * div * kMul;
  }
  static const double octaves_in_20_to_20000 = log(20000/20.)/log(2);
  static const double octaves_per_rot =
      octaves_in_20_to_20000 / float(kNumRotators - 1);
  static const double masker_step_per_octave_upZ[30] = {
    20.98103630850407,
    20.98103630850407,
    20.98103630850407,
    20.98103630850407,
    20.98103630850407,
    20.971036308504068,
    20.98103630850407,
    20.98103630850407,
    20.98103630850407,
    20.98103630850407,
    27.164728846693212,
    27.164728846693212,
    27.164728846693212,
    27.164728846693212,
    27.164728846693212,
    27.164728846693212,
    27.164728846693212,
    27.234728846693212,
    27.164728846693212,
    27.164728846693212,
    -23.182165772713049,
    -23.66616577271305,
    -23.182165772713049,
    -22.936865772713052,
    -23.112165772713048,
    -23.19216577271305,
    -23.182165772713049,
    -23.182165772713049,
    -23.182165772713049,
    -23.182165772713049,
  };

  // Strange masking (flip from usual positive to negative mask damping per
  // octave in the higher freqs) -- this could be related to middle ear
  // mechanisms having sprung mass, distributing lower and mid frequencies
  // into higher frequencies more directly and causing masking.
  // Could be a fluke.
  static const double masker_step_per_octave_up[30] = {
    20.98103630850407,
    17.593036308504068,
    23.781036308504071,
    20.98103630850407,
    20.98103630850407,
    20.971036308504068,
    20.98103630850407,
    21.98103630850407,
    22.98103630850407,
    23.98103630850407,
    24.164728846693212,
    25.164728846693212,
    26.164728846693212,
    27.164728846693212,
    27.164728846693212,
    27.164728846693212,
    27.164728846693212,
    20.234728846693212,
    17.432728846693212,
    5.1647288466932118,
    -5.1821657727130486,
    -10.66616577271305,
    -15.182165772713049,
    -15.33686577271305,
    -23.112165772713048,
    -23.19216577271305,
    -23.182165772713049,
    -23.182165772713049,
    -17.022165772713048,
    -23.182165772713049,
  };

  static const double masker_step_per_octave_down = 22.2;
  static const double masker_step_per_rot_down = octaves_per_rot * masker_step_per_octave_down;
  // propagate masker up
  float mask = 0;
  for (int k = 0; k < kNumRotators; ++k) {
    float v = masker[k];
    if (mask < v) {
      mask = v;
    }
    masker[k] = std::max<float>(masker[k], mask);
    mask -= octaves_per_rot * masker_step_per_octave_up[30 * k / kNumRotators];
  }
  // propagate masker down
  mask = 0;
  for (int k = kNumRotators - 1; k >= 0; --k) {
    float v = masker[k];
    if (mask < v) {
      mask = v;
    }
    masker[k] = std::max<float>(masker[k], mask);
    mask -= masker_step_per_rot_down;
  }
}

void FinalizeDb(std::vector<float> rotator_frequency,
                hwy::AlignedNDArray<float, 2>& channels, float mul,
                size_t out_ix) {
  float masker[kNumRotators];
  for (int k = 0; k < kNumRotators; ++k) {
    float v = SimpleDb(mul * channels[{out_ix}][k]);
    channels[{out_ix}][k] = Loudness(rotator_frequency[k], v);
  }
  PrepareMasker(channels, &masker[0], out_ix);


  static const double masker_gap = 20.832650866565942;
  static const double maskingStrengthLut[8] = {
    0.40243735146213372,
    0.53929582802463372,
    0.2330768290011962,
    0.45162153347411349,
    0.40441574013400872,
    0.3624992033380407,
    0.40282874306369626,
    0.58086335243869625,
  };
  static const double mulLut[8] = {
    1.0676829999999999,
    0.89912999999999998,
    0.90283000000000002,
    1.0,
    0.92053000000000007,
    1.0,
    1.0008999999999999,
    0.93271699999999991,
  };
  static const double addLut[8] = {
    0,
    0.079470000000000013,
    0.14000000000000001,
    0,
    0.0070000000000000001,
    0.009470000000000001,
    0,
    0,
  };
  static const float min_limit = 0;

  // Scan frequencies from bottom to top, let lower frequencies to mask higher frequencies.
  // 'masker' maintains the masking envelope from one bin to next.
  for (int k = 0; k < kNumRotators; ++k) {
    int ix = 8 * k / kNumRotators;
    float maskingStrength = maskingStrengthLut[ix];
    float v = channels[{out_ix}][k];
    double mask = masker[k] - masker_gap;
    if (v < min_limit) {
      v = min_limit;
    }
    if (v < mask) {
      v = maskingStrength * mask + (1.0 - maskingStrength) * v;
    }
    channels[{out_ix}][k] = v;
    channels[{out_ix}][k] *= mulLut[ix];
    channels[{out_ix}][k] += addLut[ix];
  }
}

void Rotators::FilterAndDownsample(hwy::Span<const float> signal,
                                   hwy::AlignedNDArray<float, 2>& channels,
                                   int downsampling) {
  float scaling_for_downsampling = 1.0f / downsampling;
  size_t out_ix = 0;
  for (int64_t ii = 0; ii < signal.size(); ii += downsampling) {
    OccasionallyRenormalize();
    for (int64_t zz = 0; zz < downsampling; ++zz) {
      int64_t input_ix = ii + zz;
      if (input_ix >= signal.size()) {
        if (out_ix < channels.shape()[0]) {
          FinalizeDb(rotator_frequency, channels, scaling_for_downsampling, out_ix);
        }
        if (out_ix != channels.shape()[0] - 1) {
          fprintf(stderr,
                  "strange thing #9831021 happened in FilterAndDownsample\n");
          abort();
        }
        return;
      }
      IncrementAll(signal[input_ix]);
      if (zz == 0) {
        for (int k = 0; k < kNumRotators; ++k) {
          float energy =
              channel[0].accu[4][k] * channel[0].accu[4][k] +
              channel[0].accu[5][k] * channel[0].accu[5][k];
          channels[{out_ix}][k] = energy;
        }
      } else {
        for (int k = 0; k < kNumRotators; ++k) {
          float energy =
              channel[0].accu[4][k] * channel[0].accu[4][k] +
              channel[0].accu[5][k] * channel[0].accu[5][k];
          channels[{out_ix}][k] += energy;
        }
      }
    }
    FinalizeDb(rotator_frequency, channels, scaling_for_downsampling, out_ix);
    ++out_ix;
    if (out_ix >= channels.shape()[0]) {
      return;
    }
  }
}

double CalculateBandwidth(double low, double mid, double high) {
  const double geo_mean_low = std::sqrt(low * mid);
  const double geo_mean_high = std::sqrt(mid * high);
  return std::abs(geo_mean_high - mid) + std::abs(mid - geo_mean_low);
}

Rotators::Rotators(int num_channels, std::vector<float> frequency,
                   std::vector<float> filter_gains, const float sample_rate) {
  channel.resize(num_channels);
  static const float kWindow = 0.9996028710680265;
  static const double kBandwidthMagic = 0.7328516996032982;
  for (int i = 0; i < kNumRotators; ++i) {
    // The parameter relates to the frequency shape overlap and window length
    // of triple leaking integrator.
    float bw = CalculateBandwidth(
        i == 0 ? frequency[1] : frequency[i - 1], frequency[i],
        i + 1 == kNumRotators ? frequency[i - 1] : frequency[i + 1]);
    window[i] = std::pow(kWindow, bw * kBandwidthMagic);
    float windowM1 = 1.0f - window[i];
    float f = frequency[i] * 2.0f * M_PI / sample_rate;
    static const float full_scale_sine_db = exp(76.639635259053421);
    const float gainer = sqrt(full_scale_sine_db);
    gain[i] = gainer * filter_gains[i] * pow(windowM1, 3.0);
    rot[0][i] = float(std::cos(f));
    rot[1][i] = float(-std::sin(f));
    rot[2][i] = gain[i];
    rot[3][i] = 0.0f;
  }
  rotator_frequency = frequency;
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
    const float tr = rot[0][i] * rot[2][i] - rot[1][i] * rot[3][i];
    const float tc = rot[0][i] * rot[3][i] + rot[1][i] * rot[2][i];
    rot[2][i] = tr;
    rot[3][i] = tc;
    const float w = window[i];
    for (int c = 0; c < 1; ++c) {
      channel[c].accu[0][i] *= w;
      channel[c].accu[1][i] *= w;
      channel[c].accu[2][i] *= w;
      channel[c].accu[3][i] *= w;
      channel[c].accu[4][i] *= w;
      channel[c].accu[5][i] *= w;
      channel[c].accu[2][i] += channel[c].accu[0][i];
      channel[c].accu[3][i] += channel[c].accu[1][i];
      channel[c].accu[4][i] += channel[c].accu[2][i];
      channel[c].accu[5][i] += channel[c].accu[3][i];
      channel[c].accu[0][i] += rot[2][i] * signal;
      channel[c].accu[1][i] += rot[3][i] * signal;
    }
  }
}

}  // namespace tabuli
