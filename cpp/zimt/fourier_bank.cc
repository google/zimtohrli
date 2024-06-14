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

float GetRotatorGains(int i) {
  static const float kRotatorGains[128] = {
      1.050645, 1.948438, 3.050339, 3.967913, 4.818584, 5.303335, 5.560281,
      5.490826, 5.156689, 4.547374, 3.691308, 2.666868, 1.539254, 0.656948,
      0.345893, 0.327111, 0.985318, 1.223506, 0.447645, 0.830961, 1.075181,
      0.613335, 0.902695, 0.855391, 0.817774, 0.823359, 0.841483, 0.838562,
      0.831912, 0.808731, 0.865214, 0.808036, 0.850837, 0.821305, 0.839458,
      0.829195, 0.836373, 0.827271, 0.836018, 0.834514, 0.825624, 0.836999,
      0.833990, 0.832992, 0.830897, 0.832593, 0.846116, 0.824796, 0.829331,
      0.844509, 0.838830, 0.821733, 0.840738, 0.841735, 0.827570, 0.838581,
      0.837742, 0.834965, 0.842970, 0.832145, 0.847596, 0.840942, 0.830891,
      0.850632, 0.841468, 0.838383, 0.841493, 0.855118, 0.826750, 0.848000,
      0.874356, 0.812177, 0.849037, 0.893550, 0.832527, 0.827986, 0.877198,
      0.851760, 0.846317, 0.883044, 0.843178, 0.856925, 0.857045, 0.860695,
      0.894345, 0.870391, 0.839519, 0.870541, 0.870573, 0.902951, 0.871798,
      0.818328, 0.871413, 0.921101, 0.863915, 0.793014, 0.936519, 0.888107,
      0.856968, 0.821018, 0.987345, 0.904846, 0.783447, 0.973613, 0.903628,
      0.875688, 0.931024, 0.992087, 0.806914, 1.050332, 0.942569, 0.800870,
      1.210426, 0.916555, 0.817352, 1.126946, 0.985119, 0.922530, 0.994633,
      0.959602, 0.381419, 1.879201, 2.078451, 0.475196, 0.952731, 1.709305,
      1.383894, 1.557669,
  };
  return kRotatorGains[i];
}

float SimpleDb(float energy) {
  static const float full_scale_sine_db = 78.3;
  return 10 * log10(energy + 1e-9) + full_scale_sine_db;
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
          for (int k = 0; k < kNumRotators; ++k) {
            channels[{out_ix}][k] =
                SimpleDb(scaling_for_downsampling * channels[{out_ix}][k]);
          }
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
    for (int k = 0; k < kNumRotators; ++k) {
      channels[{out_ix}][k] =
          SimpleDb(scaling_for_downsampling * channels[{out_ix}][k]);
    }
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
                   std::vector<float> filter_gains, const float sample_rate,
                   float global_gain) {
  channel.resize(num_channels);
  for (int i = 0; i < kNumRotators; ++i) {
    // The parameter relates to the frequency shape overlap and window length
    // of triple leaking integrator.
    float kWindow = 0.9996;
    float w40Hz = std::pow(kWindow, 128.0 / kNumRotators);  // at 40 Hz.
    float bw = CalculateBandwidth(
        i == 0 ? frequency[1] : frequency[i - 1], frequency[i],
        i + 1 == kNumRotators ? frequency[i - 1] : frequency[i + 1]);
    window[i] = std::pow(kWindow, bw * 0.7018);
    float windowM1 = 1.0f - window[i];
    float f = frequency[i] * 2.0f * M_PI / sample_rate;
    gain[i] = 2.0 * filter_gains[i] * global_gain * pow(windowM1, 3.0);
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
