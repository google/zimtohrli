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

#ifndef CPP_ZIMT_ELLIPTIC_H_
#define CPP_ZIMT_ELLIPTIC_H_

#include <array>
#include <complex>
#include <ostream>
#include <vector>

namespace zimtohrli {

// Returns the Carlson elliptic function of the first kind.
// See scipy.special.ellipk.
double EllipticIntegral1(double y);

// Returns the Jacobian elliptic functions of parameter m between 0 and 1, and
// real argument u.
// See scipy.special.ellipj.
std::array<double, 3> EllipticJacobian(double u, double m);

// Contains zeroes, poles, and gain for a filter.
struct ZPKCoeffs {
  friend std::ostream& operator<<(std::ostream& os, const ZPKCoeffs& coeffs);
  std::vector<std::complex<double>> zeros;
  std::vector<std::complex<double>> poles;
  double gain;
};

// Computes the zeroes, poles, and gain of a prototype analog elliptic low pass
// filter with the given parameters.
// See scipy.signal.ellipap.
ZPKCoeffs AnalogPrototypeLowPass(int order, double pass_band_ripple,
                                 double stop_band_ripple);

// Computes the zeros, poles, and gain of an analog elliptic band pass filter
// based on the provided parameters.
// See scipy.signal.lp2bp_zpk.
ZPKCoeffs AnalogBandPassFromLowPass(const ZPKCoeffs& low_pass, double wo,
                                    double bw);

// Computes the zeros, poles, and gain of a digital elliptic band pass filter.
// See scipy.signal.bilinear_zpk.
ZPKCoeffs DigitalBandPassFromAnalog(const ZPKCoeffs& analog,
                                    double sample_rate);

// Contains the b-, and a-coeffs of a filter.
struct BACoeffs {
  friend std::ostream& operator<<(std::ostream& os, const BACoeffs& coeffs);
  std::vector<double> b_coeffs;
  std::vector<double> a_coeffs;
};

// Computes the b-, and a-coeffs of an filter based on the provided zeros,
// poles, and gain.
// See scipy.signal.zpk2tf.
BACoeffs BAFromZPK(const ZPKCoeffs& zpk);

// Computes b-, and a-coefficients for a sequence of third order digital
// elliptic filters that when applied in sequence defines an elliptic band pass
// filter.
// See scipy.signal.ellip(output='sos').
std::vector<BACoeffs> DigitalSOSBandPass(int order, double pass_band_ripple,
                                         double stop_band_ripple,
                                         double low_threshold,
                                         double high_threshold,
                                         double sample_rate);

}  // namespace zimtohrli

#endif  // CPP_ZIMT_ELLIPTIC_H_
