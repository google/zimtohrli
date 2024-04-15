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

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <list>
#include <type_traits>
#include <vector>

#include "absl/log/check.h"

namespace zimtohrli {

namespace {

constexpr int kMaxEllipticIntegralIter = 100;
constexpr double kEllipticIntegralEpsilon = 1e-12;

}  // namespace

std::ostream& operator<<(std::ostream& os, const ZPKCoeffs& coeffs) {
  os << coeffs.gain << " * (";
  for (size_t index = 0; index < coeffs.zeros.size(); ++index) {
    os << "(" << coeffs.zeros[index] << " - z)";
    if (index + 1 < coeffs.zeros.size()) {
      os << " * ";
    }
  }
  os << ") / (";
  for (size_t index = 0; index < coeffs.poles.size(); ++index) {
    os << "(" << coeffs.poles[index] << " - z)";
    if (index + 1 < coeffs.poles.size()) {
      os << " * ";
    }
  }
  os << ")";
  return os;
}

std::ostream& operator<<(std::ostream& os, const BACoeffs& coeffs) {
  os << "(";
  for (size_t index = 0; index < coeffs.b_coeffs.size(); ++index) {
    os << "(" << coeffs.b_coeffs[index] << " - z)";
    if (index + 1 < coeffs.b_coeffs.size()) {
      os << " * ";
    }
  }
  os << ") / (";
  for (size_t index = 0; index < coeffs.a_coeffs.size(); ++index) {
    os << "(" << coeffs.a_coeffs[index] << " - z)";
    if (index + 1 < coeffs.a_coeffs.size()) {
      os << " * ";
    }
  }
  os << ")";
  return os;
}

// See
// https://en.wikipedia.org/wiki/Carlson_symmetric_form#Numerical_evaluation.
double EllipticIntegral1(double y) {
  if (y >= 1) {
    return std::numeric_limits<double>::infinity();
  }
  double x = 0.0;
  double z = 1.0;
  y = 1.0 - y;
  double error = 1.0;
  for (int i = 0;
       i < kMaxEllipticIntegralIter && error > kEllipticIntegralEpsilon; ++i) {
    const double lambda = std::sqrt(x) * std::sqrt(y) +
                          std::sqrt(y) * std::sqrt(z) +
                          std::sqrt(z) * std::sqrt(x);
    x = (x + lambda) * 0.25;
    double next_y = (y + lambda) * 0.25;
    error = std::abs(next_y - y);
    y = next_y;
    z = (z + lambda) * 0.25;
  }
  return std::pow(y, -0.5);
}

namespace {

constexpr double kMinimumJacobianMThreshold = 1.0e-9;
constexpr double kMaximumJacobianMThreshold = 0.9999999999;
constexpr double kApproximateMachineEpsilon = 2e-53;
constexpr int kEllipticJacobianSteps = 9;

const std::array<double, 3> undefined_jacobian{
    std::numeric_limits<double>::quiet_NaN(),
    std::numeric_limits<double>::quiet_NaN(),
    std::numeric_limits<double>::quiet_NaN()};

}  // namespace

std::array<double, 3> EllipticJacobian(double u, double m) {
  // Check for special cases
  if (m < 0.0 || m > 1.0) {
    return undefined_jacobian;
  }

  if (m < kMinimumJacobianMThreshold) {
    const double t = std::sin(u);
    const double b = std::cos(u);
    const double ai = 0.25 * m * (u - t * b);
    return {t - ai * b, b + ai * t, 1.0 - 0.5 * m * t * t};
  }

  if (m >= kMaximumJacobianMThreshold) {
    const double b = std::cosh(u);
    const double t = std::tanh(u);
    const double phi = 1.0 / b;
    const double twon = b * std::sinh(u);
    const double ai = 0.25 * (1.0 - m) * t * phi;
    return {t + ai * (twon - u) / (b * b), phi - ai * (twon - u),
            phi + ai * (twon + u)};
  }

  double b, t, twon;
  double a[kEllipticJacobianSteps], c[kEllipticJacobianSteps];

  // A. G. M. scale
  a[0] = 1.0;
  b = std::sqrt(1.0 - m);
  c[0] = std::sqrt(m);
  twon = 1.0;
  int i;
  for (i = 0; std::abs(c[i] / a[i]) > kApproximateMachineEpsilon &&
              i < kEllipticJacobianSteps - 1;
       ++i) {
    c[i + 1] = (a[i] - b) / 2.0;
    t = std::sqrt(a[i] * b);
    a[i + 1] = (a[i] + b) / 2.0;
    b = t;
    twon *= 2.0;
  }

  double phi;

  // backward recurrence
  phi = twon * a[i] * u;
  do {
    t = c[i] * std::sin(phi) / a[i];
    b = phi;
    phi = (std::asin(t) + phi) / 2.0;
  } while (--i);

  t = std::cos(phi);
  return {std::sin(phi), t, t / std::cos(phi - b)};
}

namespace {

const std::vector<double> kEllipDegreeMNum = {0.0, 1.0, 2.0, 3.0,
                                              4.0, 5.0, 6.0, 7.0};
const std::vector<double> kEllipDegreeMden = {1.0, 2.0, 3.0, 4.0,
                                              5.0, 6.0, 7.0, 8.0};

// Solves "n * K(m) / K'(m) = K1(m1) / K1'(m1)" for m using nomes.
double EllipDegree(double n, double m1) {
  const double K1 = EllipticIntegral1(m1);
  const double K1p = EllipticIntegral1(1.0 - m1);
  const double q1 = std::exp(-M_PI * K1p / K1);
  const double q = pow(q1, 1.0 / n);
  double num = 0.0;
  for (double el : kEllipDegreeMNum) {
    num += pow(q, el * (el + 1.0));
  }
  double den = 1.0;
  for (double el : kEllipDegreeMden) {
    den += 2.0 * pow(q, el * el);
  }
  return 16.0 * q * pow((num / den), 4.0);
}

const double log10 = std::log(10);

// Returns 10 ** x - 1 for x near 0.
double pow10m1(double x) { return std::expm1(log10 * x); }

template <typename T>
T Complement(T kx) {
  return std::sqrt((1.0 - kx) * (1.0 + kx));
}

// Solves for z in w = sn(z, m).
std::complex<double> ArcJacSn(std::complex<double> w, double m) {
  const double k = std::sqrt(m);
  if (k > 1.0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  if (k == 1.0) {
    return std::atanh(w);
  }
  std::vector<double> ks = {k};
  int niter = 0;
  while (ks.back() != 0) {
    const double k_ = ks.back();
    const double k_p = Complement(k_);
    ks.push_back((1 - k_p) / (1 + k_p));
    ++niter;
    if (niter > 10) {
      return std::numeric_limits<double>::quiet_NaN();
    }
  }

  double K = M_PI * 0.5;
  for (int i = 1; i < ks.size(); ++i) {
    K *= 1 + ks[i];
  }

  std::vector<std::complex<double>> wns = {w};
  for (int i = 1; i < ks.size(); ++i) {
    const double kn = ks[i - 1];
    const double knext = ks[i];
    const std::complex<double> wn = wns.back();
    const std::complex<double> wnext =
        (2.0 * wn / ((1.0 + knext) * (1.0 + Complement(kn * wn))));
    wns.push_back(wnext);
  }

  return K * (2.0 / M_PI * std::asin(wns.back()));
}

constexpr double kMaxRealArcJac = 1e-14;

// Solves for z in w = sc(z, 1-m).
double ArcJacSc1(double w, double m) {
  std::complex<double> zcomplex = ArcJacSn(std::complex<double>(0.0, w), m);
  if (std::abs(zcomplex.real()) > kMaxRealArcJac) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return zcomplex.imag();
}

constexpr double kMinimumPoleZero = 2e-16;

}  // namespace

ZPKCoeffs AnalogPrototypeLowPass(int order, double pass_band_ripple,
                                 double stop_band_ripple) {
  ZPKCoeffs result;
  if (order == 0) {
    result.gain = std::pow(10, pass_band_ripple * -0.05);
    return result;
  }

  if (order == 1) {
    result.poles.push_back(-std::sqrt(1.0 / pow10m1(0.1 * pass_band_ripple)));
    result.gain = -result.poles[0].real();
    return result;
  }

  const double eps_sq = pow10m1(0.1 * pass_band_ripple);
  const double eps = std::sqrt(eps_sq);
  const double ck1_sq = eps_sq / pow10m1(0.1 * stop_band_ripple);
  CHECK_NE(ck1_sq, 0)
      << "Cannot design a filter with given ripple specifications";
  const std::vector<double> val = {EllipticIntegral1(ck1_sq),
                                   EllipticIntegral1(1.0 - ck1_sq)};
  const double m = EllipDegree(order, ck1_sq);
  const double capk = EllipticIntegral1(m);
  std::vector<double> j;
  for (int i = 1 - order % 2; i < order; i += 2) {
    j.push_back(i);
  }
  const int jj = j.size();

  std::vector<double> s(jj);
  std::vector<double> c(jj);
  std::vector<double> d(jj);
  std::vector<double> s_new;
  s_new.reserve(jj);
  for (int i = 0; i < jj; ++i) {
    const std::array<double, 3> jac = EllipticJacobian(j[i] * capk / order, m);
    s[i] = jac[0];
    c[i] = jac[1];
    d[i] = jac[2];
    if (std::abs(jac[0]) > kMinimumPoleZero) {
      s_new.push_back(jac[0]);
    }
  }
  result.zeros.reserve(s_new.size());
  for (int i = 0; i < s_new.size(); ++i) {
    result.zeros.push_back(
        std::complex<double>(0.0, 1.0 / (std::sqrt(m) * s_new[i])));
  }
  const int zeros_size = result.zeros.size();
  for (int i = 0; i < zeros_size; ++i) {
    result.zeros.push_back(std::conj(result.zeros[i]));
  }

  const double r = ArcJacSc1(1.0 / eps, ck1_sq);
  const double v0 = capk * r / (order * val[0]);

  std::array<double, 3> jac2 = EllipticJacobian(v0, 1.0 - m);

  const double sv = jac2[0];
  const double cv = jac2[1];
  const double dv = jac2[2];
  result.poles.reserve(jj);
  for (int i = 0; i < jj; ++i) {
    result.poles.push_back(
        -std::complex<double>(c[i] * d[i] * sv * cv, s[i] * dv) /
        (1.0 - std::pow(d[i] * sv, 2.0)));
  }
  if (order % 2) {
    std::complex<double> p_sum = 0.0;
    for (const auto& p_el : result.poles) {
      p_sum += p_el * std::conj(p_el);
    }
    const double minimum_p = std::sqrt(p_sum.real()) * kMinimumPoleZero;
    const size_t poles_size = result.poles.size();
    for (int i = 0; i < poles_size; ++i) {
      if (std::abs(result.poles[i].imag()) > minimum_p) {
        result.poles.push_back(std::conj(result.poles[i]));
      }
    }
  } else {
    const int poles_size = result.poles.size();
    for (int i = 0; i < poles_size; ++i) {
      result.poles.push_back(std::conj(result.poles[i]));
    }
  }

  std::complex<double> k_complex{1.0, 0.0};
  for (const auto& p_el : result.poles) {
    k_complex *= -p_el;
  }
  for (const auto& z_el : result.zeros) {
    k_complex /= -z_el;
  }
  result.gain = k_complex.real();

  if (order % 2 == 0) {
    result.gain /= std::sqrt(1 + eps_sq);
  }
  return result;
}

ZPKCoeffs AnalogBandPassFromLowPass(const ZPKCoeffs& low_pass, double wo,
                                    double bw) {
  ZPKCoeffs result;
  const int degree = low_pass.poles.size() - low_pass.zeros.size();
  CHECK_GE(degree, 0);
  std::vector<std::complex<double>> z_lp;
  z_lp.reserve(low_pass.zeros.size());
  for (const auto& zero : low_pass.zeros) {
    z_lp.push_back(zero * bw * 0.5);
  }
  std::vector<std::complex<double>> p_lp;
  p_lp.reserve(low_pass.poles.size());
  for (const auto& pole : low_pass.poles) {
    p_lp.push_back(pole * bw * 0.5);
  }
  result.zeros = std::vector<std::complex<double>>(z_lp.size() * 2);
  for (int i = 0; i < z_lp.size(); ++i) {
    const std::complex<double> z_lp_wo_norm =
        std::sqrt(z_lp[i] * z_lp[i] - wo * wo);
    result.zeros[i] = z_lp[i] + z_lp_wo_norm;
    result.zeros[i + z_lp.size()] = z_lp[i] - z_lp_wo_norm;
  }
  result.poles = std::vector<std::complex<double>>(p_lp.size() * 2);
  for (int i = 0; i < p_lp.size(); ++i) {
    const std::complex<double> p_lp_wo_norm =
        std::sqrt(p_lp[i] * p_lp[i] - wo * wo);
    result.poles[i] = p_lp[i] + p_lp_wo_norm;
    result.poles[i + p_lp.size()] = p_lp[i] - p_lp_wo_norm;
  }
  result.zeros.reserve(degree);
  for (int i = 0; i < degree; ++i) {
    result.zeros.push_back(0.0);
  }
  result.gain = low_pass.gain * std::pow(bw, degree);
  return result;
}

ZPKCoeffs DigitalBandPassFromAnalog(const ZPKCoeffs& analog,
                                    double sample_rate) {
  ZPKCoeffs result;
  const int degree = analog.poles.size() - analog.zeros.size();
  CHECK_GE(degree, 0);
  const double fs2 = 2.0 * sample_rate;
  result.zeros = std::vector<std::complex<double>>();
  result.zeros.reserve(analog.zeros.size());
  for (const auto& z : analog.zeros) {
    result.zeros.push_back((fs2 + z) / (fs2 - z));
  }
  result.poles = std::vector<std::complex<double>>();
  result.poles.reserve(analog.poles.size());
  for (const auto& p : analog.poles) {
    result.poles.push_back((fs2 + p) / (fs2 - p));
  }
  for (int i = 0; i < degree; ++i) {
    result.zeros.push_back(-1.0);
  }
  std::complex<double> zero_prod = 1.0;
  for (const auto& z : analog.zeros) {
    zero_prod *= (fs2 - z);
  }
  std::complex<double> pole_prod = 1.0;
  for (const auto& p : analog.poles) {
    pole_prod *= (fs2 - p);
  }
  result.gain = analog.gain * (zero_prod / pole_prod).real();
  return result;
}

namespace {

// Returns all combinations of the given options and size.
// See e.g.
// https://docs.python.org/3/library/itertools.html#itertools.combinations
std::list<std::vector<size_t>> Combinations(size_t options, size_t size) {
  CHECK_GE(options, size);

  std::list<std::vector<size_t>> result;

  // Create a selector for each option.
  std::vector<bool> selectors(options);
  // Mark the last ones as true, which will be the lexicographically first set
  // of selectors with this many true values.
  std::fill(selectors.end() - size, selectors.end(), true);

  // While there are lexicographically later permutations of the selectors,
  // pick the options that are selected and put in the combinations.
  do {
    std::vector<size_t> combination;
    combination.reserve(size);
    for (size_t option_index = 0; option_index < options; option_index++) {
      if (selectors[option_index]) {
        combination.push_back(option_index);
      }
    }
    result.push_back(combination);
  } while (std::next_permutation(selectors.begin(), selectors.end()));

  return result;
}

// Computes the coefficients of a polynomial, given the zeroes.
//
//   Assuming we have a filter H, with poles P and zeros Q:
//   H = g * np.prod(Q - z) / np.prod(P - z) = Y / X
//
//   Y / X = g * prod(Q * z^-1 - 1) / prod(P * z^-1 - 1)
//   Y = X * g * prod(Q * z^-1 - 1) / prod(P * z^-1 - 1)
//   Y * prod(P * z^-1 - 1) = X * g * prod(Q * z^-1 - 1)
//   Y * sum(Pc[num] * z^-num for num in range(len(P)+1)) =
//     X * g * sum(Qc[num] * z^-num for num in range(len(Q)+1))
//
//   coeffs_from_zeros computes Qc/Pc from Q/P.
//
std::vector<double> CoeffsFromZeros(
    const std::vector<std::complex<double>>& polynomial_zeros) {
  std::vector<std::complex<double>> result(polynomial_zeros.size() + 1);
  for (size_t num = 0; num < polynomial_zeros.size() + 1; ++num) {
    std::complex<double> s = 0;
    for (const std::vector<size_t>& parts :
         Combinations(polynomial_zeros.size(), num)) {
      std::complex<double> prod = 1;
      for (size_t part : parts) {
        prod *= -polynomial_zeros[part];
      }
      s += prod;
    }
    result[num] = s;
  }
  std::vector<double> real_result;
  real_result.reserve(result.size());
  for (const std::complex<double>& coeff : result) {
    real_result.push_back(coeff.real());
  }
  return real_result;
}

}  // namespace

BACoeffs BAFromZPK(const ZPKCoeffs& zpk) {
  BACoeffs result;
  result.b_coeffs = CoeffsFromZeros(zpk.zeros);
  for (double& numerator : result.b_coeffs) {
    numerator *= zpk.gain;
  }
  result.a_coeffs = CoeffsFromZeros(zpk.poles);
  return result;
}

namespace {

bool IsNegativeConjugate(const std::complex<double>& value) {
  return value.imag() < -100 * kApproximateMachineEpsilon;
}

int CountReal(const std::vector<std::complex<double>>& values) {
  int result = 0;
  for (const std::complex<double> value : values) {
    if (value.imag() == 0) {
      result++;
    }
  }
  return result;
}

struct kMustBeReal {};
struct kMustBeComplex {};
struct kRealOrComplex {};

constexpr double kBestQualityEpsilon = 1e-12;

template <typename T, typename requirement>
std::complex<double> PopBest(std::vector<std::complex<double>>& values,
                             T quality_metric, requirement req) {
  static_assert(std::is_same<requirement, kMustBeReal>::value ||
                std::is_same<requirement, kMustBeComplex>::value ||
                std::is_same<requirement, kRealOrComplex>::value);
  int best_index = -1;
  double best_quality = 0.0;
  for (int i = 0; i < values.size(); ++i) {
    const std::complex<double>& value = values[i];
    if (std::is_same<requirement, kRealOrComplex>::value ||
        (std::is_same<requirement, kMustBeReal>::value && value.imag() == 0) ||
        (std::is_same<requirement, kMustBeComplex>::value &&
         value.imag() != 0)) {
      const double quality = quality_metric(value);
      // Picking the last element with best quality to conform with how the
      // tested version of scipy does it.
      if (best_index == -1 || quality >= (best_quality - kBestQualityEpsilon)) {
        best_index = i;
        best_quality = quality;
      }
    }
  }
  CHECK_GE(best_index, 0);
  std::complex<double> result = values[best_index];
  values.erase(values.begin() + best_index);
  return result;
}

template <typename requirement>
std::complex<double> PopClosest(std::vector<std::complex<double>>& values,
                                const std::complex<double>& term,
                                requirement req) {
  return PopBest(
      values,
      [&](const std::complex<double>& value) {
        return -std::abs(value - term);
      },
      req);
}

template <typename requirement>
std::complex<double> PopWorst(std::vector<std::complex<double>>& values,
                              requirement req) {
  return PopBest(
      values,
      [&](const std::complex<double>& value) { return std::abs(value); }, req);
}

std::vector<BACoeffs> SOSSectionsFromZPK(const ZPKCoeffs& zpk) {
  CHECK_GT(zpk.zeros.size(), 0);
  CHECK_GT(zpk.poles.size(), 0);

  // Ensure equal and even number of poles and zeros.
  std::vector<std::complex<double>> zeros(zpk.zeros);
  std::vector<std::complex<double>> poles(zpk.poles);
  while (zeros.size() < poles.size()) {
    zeros.push_back(0.0);
  }
  while (poles.size() < zeros.size()) {
    poles.push_back(0.0);
  }
  if (zeros.size() & 1) {
    poles.push_back(0.0);
    zeros.push_back(0.0);
  }

  // Remove the negative conjugates so that we only have unique examples from
  // each conjugate pair.
  zeros.erase(std::remove_if(zeros.begin(), zeros.end(), IsNegativeConjugate),
              zeros.end());
  poles.erase(std::remove_if(poles.begin(), poles.end(), IsNegativeConjugate),
              poles.end());

  std::vector<BACoeffs> sections;

  while (!poles.empty()) {
    const std::complex<double> pole1 = PopWorst(poles, kRealOrComplex{});
    if (pole1.imag() == 0 && CountReal(poles) == 0) {
      if (!zeros.empty()) {
        const std::complex<double> zero1 =
            PopClosest(zeros, pole1, kMustBeReal{});
        sections.push_back(
            BAFromZPK({.zeros = {zero1}, .poles = {pole1}, .gain = 1}));
      } else {
        sections.push_back(
            BAFromZPK({.zeros = {}, .poles = {pole1}, .gain = 1}));
      }
    } else if (poles.size() + 1 == zeros.size() && pole1.imag() != 0 &&
               CountReal(poles) == 1 && CountReal(zeros) == 1) {
      const std::complex<double> zero1 =
          PopClosest(zeros, pole1, kMustBeComplex{});
      sections.push_back(BAFromZPK({.zeros = {zero1, std::conj(zero1)},
                                    .poles = {pole1, std::conj(pole1)},
                                    .gain = 1}));
    } else {
      const std::complex<double> pole2 =
          pole1.imag() == 0 ? PopWorst(poles, kMustBeReal{}) : std::conj(pole1);
      if (!zeros.empty()) {
        const std::complex<double> zero1 =
            PopClosest(zeros, pole1, kRealOrComplex{});
        if (zero1.imag() != 0) {
          sections.push_back(BAFromZPK({.zeros = {zero1, std::conj(zero1)},
                                        .poles = {pole1, pole2},
                                        .gain = 1}));
        } else {
          if (!zeros.empty()) {
            const std::complex<double> zero2 =
                PopClosest(zeros, pole1, kMustBeReal{});
            sections.push_back(BAFromZPK(
                {.zeros = {zero1, zero2}, .poles = {pole1, pole2}, .gain = 1}));
          } else {
            sections.push_back(BAFromZPK(
                {.zeros = {zero1}, .poles = {pole1, pole2}, .gain = 1}));
          }
        }
      } else {
        sections.push_back(
            BAFromZPK({.zeros = {}, .poles = {pole1, pole2}, .gain = 1}));
      }
    }
  }
  std::reverse(sections.begin(), sections.end());
  CHECK(zeros.empty());
  for (auto& coeff : sections[0].b_coeffs) {
    coeff *= zpk.gain;
  }

  return sections;
}

}  // namespace

std::vector<BACoeffs> DigitalSOSBandPass(int order, double pass_band_ripple,
                                         double stop_band_ripple,
                                         double low_threshold,
                                         double high_threshold,
                                         double sample_rate) {
  CHECK_GE(order, 0);
  CHECK_GT(pass_band_ripple, 0);
  CHECK_GT(stop_band_ripple, 0);
  CHECK_GT(low_threshold, 0);
  CHECK_GT(high_threshold, low_threshold);
  CHECK_GT(sample_rate, 0);
  CHECK_GE(sample_rate, 2 * high_threshold);
  const ZPKCoeffs low_pass =
      AnalogPrototypeLowPass(order, pass_band_ripple, stop_band_ripple);
  const double low_threshold_w = 2 * low_threshold / sample_rate;
  const double high_threshold_w = 2 * high_threshold / sample_rate;
  const double sample_rate_w = 2;
  const double low_threshold_warped =
      2 * sample_rate_w * tan(M_PI * low_threshold_w / sample_rate_w);
  const double high_threshold_warped =
      2 * sample_rate_w * tan(M_PI * high_threshold_w / sample_rate_w);
  const double bw = high_threshold_warped - low_threshold_warped;
  const double wo = sqrt(low_threshold_warped * high_threshold_warped);
  const ZPKCoeffs band_pass = AnalogBandPassFromLowPass(low_pass, wo, bw);
  const ZPKCoeffs digital = DigitalBandPassFromAnalog(band_pass, sample_rate_w);
  return SOSSectionsFromZPK(digital);
}

}  // namespace zimtohrli
