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
  static const float pars[30][3] = {
    // ~[20*1.2589254117941673**i for i in range(31)]
    { 0.635, -31.5, 78.1, }, // 20
    { 0.602, -27.2, 68.7, }, // 25
    { 0.569, -23.1, 59.5, }, // 31.5
    { 0.537, -19.3, 51.1, }, // 40
    { 0.509, -16.1, 44.0, }, // 50
    { 0.482, -13.1, 37.5, }, // 63
    { 0.456, -10.4, 31.5, }, // 80
    { 0.433, -8.2, 26.5, }, // 100
    { 0.412, -6.3, 22.1, }, // 125
    { 0.391, -4.6, 17.9, }, // 160
    { 0.373, -3.2, 14.4, }, // 200
    { 0.357, -2.1, 11.4, }, // 250
    { 0.343, -1.2, 8.6, }, // 315
    { 0.330, -0.5, 6.2, }, // 400
    { 0.320, 0.0, 4.4, }, // 500
    { 0.311, 0.4, 3.0, }, // 630
    { 0.303, 0.5, 2.2, }, // 800
    { 0.300, 0.0, 2.4, }, // 1000
    { 0.295, -2.7, 3.5, }, // 1250
    { 0.292, -4.2, 1.7, }, // 1600
    { 0.290, -1.2, -1.3, }, // 2000
    { 0.290, 1.4, -4.2, }, // 2500
    { 0.289, 2.3, -6.0, }, // 3150
    { 0.289, 1.0, -5.4, }, // 4000
    { 0.289, -2.3, -1.5, }, // 5000
    { 0.293, -7.2, 6.0, }, // 6300
    { 0.303, -11.2, 12.6, }, // 8000
    { 0.323, -10.9, 13.9, }, // 10000
    { 0.354, -3.5, 12.3, }, // 12500
  };
  const float *vals = &pars[k / 5][0];
  static float constant1 = 48.70343225300608;
  static float constant2 = 40.43827165462807;
  static const float kMul = -2.7207126528654677;
  val += kMul * vals[2];
  val *= (constant1 + vals[1]) * (1.0 / constant2);
  return val;
}

float SimpleDb(float energy) {
  // ideally 78.3 db
  static const float full_scale_sine_db = 77.39771094914877;
  static const float exp_full_scale_sine_db = exp(full_scale_sine_db);
  // epsilon, but the biggest one you saw (~4.95e23)
  static const float epsilon = 1.0033294789821357e-09 * exp_full_scale_sine_db;
  // kMul allows faster log instead of log10 below, incorporating multiplying by 10 for decibel.
  constexpr float kMul = 10.0/log(10);
  return kMul * log(energy + epsilon);
}

void FinalizeDb(hwy::AlignedNDArray<float, 2>& channels, float mul,
                size_t out_ix) {
  double masker = 0.0;
  static const double octaves_in_20_to_20000 = log(20000/20.)/log(2);
  static const double octaves_per_rot =
      octaves_in_20_to_20000 / float(kNumRotators - 1);
  static const double masker_step_per_octave_up_0 = 19.53945781131615;
  static const double masker_step_per_octave_up_1 = 24.714118008386887;
  static const double masker_step_per_octave_up_2 = 6.449301354309956;
  static const double masker_step_per_rot_up_0 = octaves_per_rot * masker_step_per_octave_up_0;
  static const double masker_step_per_rot_up_1 = octaves_per_rot * masker_step_per_octave_up_1;
  static const double masker_step_per_rot_up_2 = octaves_per_rot * masker_step_per_octave_up_2;
  static const double masker_gap_up = 21.309406898722074;
  static const float maskingStrengthUp = 0.2056434702527141;
  static const float up_blur = 0.9442717063037425;
  static const float fraction_up = 1.1657467617827404;

  static const double masker_step_per_octave_down = 53.40273959309446;
  static const double masker_step_per_rot_down = octaves_per_rot * masker_step_per_octave_down;
  static const double masker_gap_down = 19.08401096304284;
  static const float maskingStrengthDown = 0.18030917038808858;
  static const float down_blur = 0.7148792180987857;

  static const float min_limit = -11.3968870989223;
  static const float fraction_down = 1.0197608300379997;

  static const float temporal0 = 0.09979167061501665;
  static const float temporal1 = 0.14429505133534495;
  static const float temporal2 = 0.009228598592129168;
  static float weightp = 0.1792443302507868;
  static float weightm = 0.7954490998745948;

  static float mask_k = 0.08709005149742773;

  // Scan frequencies from bottom to top, let lower frequencies to mask higher frequencies.
  // 'masker' maintains the masking envelope from one bin to next.
  static const float temporal_masker0 = 0.13104546362447728;
  static const float temporal_masker1 = 0.09719740670406614;
  static const float temporal_masker2 = -0.03085233735225447;

  for (int k = 0; k < kNumRotators; ++k) {
    float v = SimpleDb(mul * channels[{out_ix}][k]);
    if (v < min_limit) {
      v = min_limit;
    }
    float v2 = (1 - up_blur) * v2 + up_blur * v;
    if (k == 0) {
      v2 = v;
    }
    if (masker < v2) {
      masker = v2;
    }
    float mask = masker - masker_gap_up;

    if (v < mask) {
      v = maskingStrengthUp * mask + (1.0 - maskingStrengthUp) * v;
    }

    channels[{out_ix}][k] = v;
    if (3 * k < kNumRotators) {
      masker -= masker_step_per_rot_up_0;
    } else if (3 * k < 2 * kNumRotators) {
      masker -= masker_step_per_rot_up_1;
    } else {
      masker -= masker_step_per_rot_up_2;
    }
  }
  // Scan frequencies from top to bottom, let higher frequencies to mask lower frequencies.
  // 'masker' maintains the masking envelope from one bin to next.
  masker = 0.0;
  for (int k = kNumRotators - 1; k >= 0; --k) {
    float v = channels[{out_ix}][k];
    float v2 = (1 - down_blur) * v2 + down_blur * v;
    if (k == kNumRotators - 1) {
      v2 = v;
    }
    if (masker < v) {
      masker = v;
    }
    float mask = masker - masker_gap_down;
    if (v < mask) {
      v = maskingStrengthDown * mask + (1.0 - maskingStrengthDown) * v;
    }
    channels[{out_ix}][k] = v;
    masker -= masker_step_per_rot_down;
  }
  for (int k = 0; k < kNumRotators; ++k) {
    channels[{out_ix}][k] = Loudness(k, channels[{out_ix}][k]);
  }
  // temporal masker
  if (out_ix >= 3) {
    for (int k = 0; k < kNumRotators; ++k) {
      float v0 = (channels[{out_ix - 1}][k] - channels[{out_ix}][k]);
      float v1 = (channels[{out_ix - 2}][k] - channels[{out_ix}][k]);
      float v2 = (channels[{out_ix - 3}][k] - channels[{out_ix}][k]);

      channels[{out_ix}][k] -= temporal_masker0 * v0 +
                               temporal_masker1 * v1 +
                               temporal_masker2 * v2;
    }
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
          FinalizeDb(channels, scaling_for_downsampling, out_ix);
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
    FinalizeDb(channels, scaling_for_downsampling, out_ix);
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
    static const float full_scale_sine_db = exp(75.27858635739499);
    const float gainer = 2.0f * sqrt(full_scale_sine_db);
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
