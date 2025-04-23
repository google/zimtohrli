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

void LoudnessDb(hwy::AlignedNDArray<float, 2>& channels, size_t out_ix) {
  static const float kMul[128] = {
    0.69022, 0.68908, 0.69206, 0.68780, 0.68780, 0.68780, 0.68780, 0.68780,
    0.68780, 0.68780, 0.68780, 0.68913, 0.69045, 0.69310, 0.69575, 0.69565,
    0.69697, 0.70122, 0.72878, 0.79911, 0.85713, 0.88063, 0.88563, 0.87561,
    0.81948, 0.70435, 0.63479, 0.58382, 0.52065, 0.48390, 0.46452, 0.47952,
    0.52686, 0.63677, 0.75972, 0.89449, 0.97411, 1.01874, 1.01105, 0.99306,
    0.93613, 0.92825, 0.93149, 0.98687, 1.05782, 1.16461, 1.25028, 1.30768,
    1.31484, 1.28574, 1.23002, 1.15336, 1.08800, 1.01472, 0.94610, 0.91856,
    0.87797, 0.85825, 0.82836, 0.82198, 0.81394, 0.82724, 0.84235, 0.86009,
    0.88276, 0.89349, 0.92543, 0.94822, 0.98526, 0.99730, 1.02097, 1.04071,
    1.05254, 1.06462, 1.06872, 1.07382, 1.06739, 1.06331, 1.05118, 1.05002,
    1.04803, 1.06729, 1.09680, 1.15208, 1.22492, 1.32630, 1.42049, 1.50444,
    1.58735, 1.65199, 1.69488, 1.70748, 1.74525, 1.68760, 1.66818, 1.63401,
    1.55136, 1.49170, 1.42649, 1.33453, 1.28618, 1.26523, 1.24900, 1.24898,
    1.27864, 1.28723, 1.28455, 1.29777, 1.29637, 1.29687, 1.29853, 1.30319,
    1.30207, 1.26835, 1.25100, 1.24664, 1.24041, 1.17874, 1.07116, 0.97917,
  };
  static const float kBaseNoise = 912654.0;
  for (int k = 0; k < kNumRotators; ++k) {
    channels[{out_ix}][k] = log(channels[{out_ix}][k] + kBaseNoise) * kMul[k];
  }
}

// Ear drum and other receiving mass-spring objects are
// modeled through the Resonator. Resonator is a non-linear process
// and does complex spectral shifting of energy.
struct Resonator {
  float acc0 = 0;
  float acc1 = 0;
  float Update(float signal) {  // Resonate and attenuate.
    acc0 = 0.925323029 * acc0 - 0.040025628 * acc1 + signal;
    acc1 += acc0;
    return acc0;
  }
};

inline float Dot32(const float *a, const float *b) {
  // -ffast-math is helpful here, and clang can simdify this.
  float sum = 0;
  for (int i = 0; i < 32; ++i) sum += a[i] * b[i];
  return sum;
}

float Freq(int i) {
  // Center frequencies of the filter bank, plus one frequency in both ends.
  static const float kFreq[130] = {
    17.858, 24.349, 33.199, 42.359, 51.839, 61.651, 71.805, 82.315,
    93.192, 104.449, 116.099, 128.157, 140.636, 153.552, 166.919, 180.754,
    195.072, 209.890, 225.227, 241.099, 257.527, 274.528, 292.124, 310.336,
    329.183, 348.690, 368.879, 389.773, 411.398, 433.778, 456.941, 480.914,
    505.725, 531.403, 557.979, 585.484, 613.950, 643.411, 673.902, 705.459,
    738.119, 771.921, 806.905, 843.111, 880.584, 919.366, 959.503, 1001.04,
    1044.03, 1088.53, 1134.58, 1182.24, 1231.57, 1282.62, 1335.46, 1390.14,
    1446.73, 1505.31, 1565.93, 1628.67, 1693.60, 1760.80, 1830.35, 1902.34,
    1976.84, 2053.94, 2133.74, 2216.33, 2301.81, 2390.27, 2481.83, 2576.58,
    2674.65, 2776.15, 2881.19, 2989.91, 3102.43, 3218.88, 3339.40, 3464.14,
    3593.23, 3726.84, 3865.12, 4008.23, 4156.35, 4309.64, 4468.30, 4632.49,
    4802.43, 4978.31, 5160.34, 5348.72, 5543.70, 5745.49, 5954.34, 6170.48,
    6394.18, 6625.70, 6865.32, 7113.31, 7369.97, 7635.61, 7910.53, 8195.06,
    8489.53, 8794.30, 9109.73, 9436.18, 9774.04, 10123.7, 10485.6, 10860.1,
    11247.8, 11648.9, 12064.2, 12493.9, 12938.7, 13399.0, 13875.3, 14368.4,
    14878.7, 15406.8, 15953.4, 16519.1, 17104.5, 17710.4, 18337.6, 18986.6,
    19658.3, 20352.7,
  };
  return kFreq[i + 1];
}

double CalculateBandwidthInHz(int i) {
  return std::sqrt(Freq(i + 1) * Freq(i)) - std::sqrt(Freq(i - 1) * Freq(i));
}

void Rotators::OccasionallyRenormalize() {
  for (int i = 0; i < kNumRotators; ++i) {
    float norm = gain[i] / sqrt(rot[2][i] * rot[2][i] + rot[3][i] * rot[3][i]);
    rot[2][i] *= norm;
    rot[3][i] *= norm;
  }
}

void Rotators::IncrementAll(float signal) {
  for (int i = 0; i < kNumRotators; i++) {  // clang is great at simdifying this.
    const float w = window[i];
    for (int k = 0; k < 6; ++k) accu[k][i] *= w;
    accu[2][i] += accu[0][i];  // For an unknown reason this 
    accu[3][i] += accu[1][i];  // update order works best.
    accu[4][i] += accu[2][i];  // i.e., 2, 3, 4, 5, and finally 0, 1.
    accu[5][i] += accu[3][i];
    accu[0][i] += rot[2][i] * signal;
    accu[1][i] += rot[3][i] * signal;
    const float a = rot[2][i], b = rot[3][i];
    rot[2][i] = rot[0][i] * a - rot[1][i] * b;
    rot[3][i] = rot[0][i] * b + rot[1][i] * a;
  }
}

void Rotators::FilterAndDownsample(hwy::Span<const float> in,
                                   hwy::AlignedNDArray<float, 2>& out,
                                   int downsample) {
  static const float kSampleRate = 48000.0;
  static const float kHzToRad = 2.0f * M_PI / kSampleRate;
  static const double kWindow = 0.9996028710680265;
  static const double kBandwidthMagic = 0.7328516996032982;
  // A big value for normalization. Ideally 1.0, but this works slightly better.
  const float gainer = sqrt(905697397139.4474 / downsample);
  for (int i = 0; i < kNumRotators; ++i) {
    float bandwidth = CalculateBandwidthInHz(i);  // bandwidth of each filter.
    window[i] = std::pow(kWindow, bandwidth * kBandwidthMagic);
    float windowM1 = 1.0f - window[i];
    const float f = Freq(i) * kHzToRad;
    gain[i] = gainer * pow(windowM1, 3.0) * Freq(i) / bandwidth;
    rot[0][i] = float(std::cos(f));
    rot[1][i] = float(-std::sin(f));
    rot[2][i] = gain[i];
    rot[3][i] = 0.0f;
  }
  for (size_t zz = 0; zz < out.shape()[0]; zz++) {
    for (int k = 0; k < kNumRotators; ++k) {
      out[{zz}][k] = 0;
    }
  }
  std::vector<float> window(downsample);
  for (int i = 0; i < downsample; ++i) {
    window[i] = 1.0 / (1.0 + exp(7.9446 * ((2.0 / downsample) * (i + 0.5) - 1)));
  }
  Resonator resonator;
  size_t out_ix = 0;
  static const float reso_kernel[32] = {
    -0.00756885973, 0.00413482141,  -0.00000236200, 0.00619875373,
    -0.00283612301, -0.00000418032, -0.00653942799, -0.00697059266,
    0.00344293224,  0.00329933933,  -0.00298496041, 0.00350131041,
    0.00171017251,  -0.00154158276, 0.00404768079,  0.00127457555,
    -0.01171138281, -0.00010813847, -0.00152608046, -0.00838915828,
    -0.00640430929, -0.00086448874, -0.00720815920, 0.00344734180,
    -0.00294620320, 0.00079453551,  0.00067657883,  0.00185866424,
    0.00615985137,  -0.00236233239, -0.00680980952, 0.01082403830,
  };
  static const float linear_kernel[32] = {
    -0.10104347418, -0.11826972031, -0.06180710258, 0.07855591921,
    0.03670823911,  -0.01840452136, 0.10859856308,  0.16449286025,
    0.06054576192,  0.08362268315,  -0.00320242077, 0.17410886426,
    -0.13348931125, 0.12798560564,  0.02840772721,  0.01655141242,
    0.00565097497,  -0.39669214512, 0.25126981719,  0.29050002107,
    -0.34990576312, 0.13135342797,  1.09071850579,  -0.97998963695,
    -0.97386487573, 0.30687938104,  0.52811340907,  1.35094332106,
    0.35339301883,  -0.17657465769, 0.36698233014,  -0.39494225991,
  };
  for (size_t in_ix = 0; in_ix < in.size() && out_ix < out.shape()[0];
       ++out_ix) {
    OccasionallyRenormalize();
    for (int64_t zz = 0; zz < downsample && in_ix < in.size(); ++zz, ++in_ix) {
      float weight = window[zz];
      if (in_ix >= 32) {
	IncrementAll(resonator.Update(Dot32(&in[in_ix - 32], &reso_kernel[0])) +
		     Dot32(&in[in_ix - 32], &linear_kernel[0]));
      }
      if (out_ix + 1 < out.shape()[0]) {
	for (int k = 0; k < kNumRotators; ++k) {
	  float energy = accu[4][k] * accu[4][k] + accu[5][k] * accu[5][k];
	  out[{out_ix + 1}][k] += (1.0 - weight) * energy;
	  out[{out_ix}][k] += weight * energy;
	}
      } else {
	for (int k = 0; k < kNumRotators; ++k) {
	  float energy = accu[4][k] * accu[4][k] + accu[5][k] * accu[5][k];
	  out[{out_ix}][k] += energy;
	}
      }
    }
    LoudnessDb(out, out_ix);
  }
}

}  // namespace tabuli
