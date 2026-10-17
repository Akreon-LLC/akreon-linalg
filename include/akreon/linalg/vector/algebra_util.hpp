/*
 * Copyright © 2025-2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <cmath>
#include <limits>
#include <optional>

#include "akreon/linalg/constants.hpp"
#include "vector_concepts.hpp"

/**
 * @brief Maximum Newton/refinement iterations used by constexpr utilities.
 *
 * The cap keeps compile-time evaluation bounded on compilers with finite
 * constexpr step budgets.
 */
#define MAX_STEP_EVAL 256

/** @brief Whether the active standard library supports P1383R2 `<cmath>`. */
#if defined(__cpp_lib_constexpr_cmath) && __cpp_lib_constexpr_cmath >= 202306L
#define LINALG_HAS_CXX26_CONSTEXPR_CMATH 1
#else
#define LINALG_HAS_CXX26_CONSTEXPR_CMATH 0
#endif

/** @brief Whether the active standard library supports P1383R2 `<complex>`. */
#if defined(__cpp_lib_constexpr_complex) &&                                    \
    __cpp_lib_constexpr_complex >= 202306L
#define LINALG_HAS_CXX26_CONSTEXPR_COMPLEX_MATH 1
#else
#define LINALG_HAS_CXX26_CONSTEXPR_COMPLEX_MATH 0
#endif

namespace linalg {

/**
 * @brief Copy a fixed-size array into a tuple of the same length.
 * @param x Source array in index order.
 * @return Tuple containing `x[0]` through `x[N - 1]`.
 */
template <typename T, std::size_t N>
constexpr auto array_to_tuple(const std::array<T, N>& x) {
  auto to_tuple = [&]<std::size_t... Is>(const auto& arr,
                                         std::index_sequence<Is...>) {
    return std::make_tuple(arr[Is]...);
  };
  return to_tuple(x, std::make_index_sequence<N>{});
}

/**
 * @brief Check whether a complex value's imaginary component is below a bound.
 *
 * This function evaluates `z.imag() <= tol` exactly; it is primarily used where
 * the expected imaginary residual is nonnegative. It does not take the
 * absolute value of the imaginary component.
 *
 * @param z Complex value to inspect.
 * @param tol Upper bound for its imaginary component.
 */
template <Field T>
constexpr bool is_real(const std::complex<T>& z, real_type_t<T> tol = 1e-10) {
  return z.imag() <= tol;
}

/**
 * @brief Divide two real scalars with an explicit zero-denominator result.
 *
 * @param num numerator
 * @param den denominator
 *
 * @return num / den when denominator is nonzero, else nullopt
 */
template <Real T>
constexpr std::optional<T> constexpr_div(T num, T den) {
  if (!(den != T(0))) return std::nullopt;
  return num / den;
}

/**
 * @brief Divide two real scalars, returning a fallback for a zero denominator.
 *
 * @param num numerator
 * @param den denominator
 * @param fallback fallback value when denominator is zero
 *
 * @return scalar division result or fallback
 */
template <Real T>
constexpr T constexpr_div_safe(T num, T den, T fallback = T(0)) {
  if (const auto r = constexpr_div(num, den); r.has_value()) return *r;
  return fallback;
}

/**
 * @brief Raise a real scalar to a nonnegative integral power.
 *
 * @param x Real base.
 * @param power Nonnegative integral exponent; conventionally positive here.
 */
template <Real T>
constexpr T constexpr_pow(T x, std::size_t power) {
#if LINALG_HAS_CXX26_CONSTEXPR_CMATH
  return std::pow(x, power);
#else
  T res(1);
  for (std::size_t i = 0; i < power; ++i) res *= x;
  return res;
#endif
}

/**
 * @brief Approximate a nonnegative square root with Newton iteration.
 *
 * f(g) = g^2 - x to df/dg = 2 g
 * Newton formula : g_{k+1} = g_k - frac{g^2 - x} / {2 g_k} equivalent to 0.5 *
 * (g_n + x / g_n)
 *
 * @param x Nonnegative radicand; a negative value returns `std::nullopt`.
 * @param g Positive initial guess; nonpositive guesses are replaced with one.
 * @param tol Absolute squared-residual tolerance.
 * @return Approximation, or `std::nullopt` outside the real domain. The current
 * zero-input convention returns zero.
 */
template <Real T>
constexpr std::optional<T> constexpr_sqrt(T x, T g = 1e-10, T tol = 1e-10) {
  if (!(x >= T(0))) return std::nullopt;
  if (x == T(0)) return T(0);

#if LINALG_HAS_CXX26_CONSTEXPR_CMATH
  (void)g;
  (void)tol;
  return std::sqrt(x);
#else
  if (!(g > T(0))) g = T(1);

  auto abs = [](const auto& val) { return val < T(0) ? -val : val; };

  for (std::size_t k = 0; k < MAX_STEP_EVAL; ++k) {
    if (!(g > T(0))) g = T(1);
    if (abs(g * g - x) <= tol) return g;
    g = T(0.5) * (g + constexpr_div_safe(x, g, g));
  }
  return g;
#endif
}

/**
 * @brief Return a scalar square root with a sanitized fallback radicand.
 *
 * @param x input scalar
 * @param floor minimum scalar for fallback
 * @param g initial guess
 * @param tol convergence tolerance
 *
 * @return scalar sqrt with optional fallback handling
 */
template <Real T>
constexpr T constexpr_sqrt_safe(T x, T floor = T(0), T g = 1e-10,
                                T tol = 1e-10) {
  if (const auto r = constexpr_sqrt(x, g, tol); r.has_value()) return *r;

  const auto floor_safe = (floor >= T(0)) ? floor : T(0);
  if (const auto rf = constexpr_sqrt(floor_safe, g, tol); rf.has_value())
    return *rf;

  return T(0);
}

/**
 * @brief Compute real absolute value or complex magnitude.
 *
 * @param x Real or complex field scalar.
 * @return `|x|` in the associated real type.
 */
template <Real T>
constexpr T constexpr_abs(T x) {
#if LINALG_HAS_CXX26_CONSTEXPR_CMATH
  return std::abs(x);
#else
  return x < T(0) ? T(-1) * x : x;
#endif
}

template <Complex T>
constexpr auto constexpr_abs(const T& x) {
#if LINALG_HAS_CXX26_CONSTEXPR_COMPLEX_MATH
  return std::abs(x);
#else
  return constexpr_sqrt_safe(x.real() * x.real() + x.imag() * x.imag());
#endif
}

/**
 * @brief Approximate a real nth root with Newton iteration.
 *
 * f(g) = g^n - x to df/dg = n g^{n-1}
 * Newton formula : g_{k+1} = g_k - frac{g_k^n - x}{n g^{n-1}} equivalent to
 * g_{k+1} = 1/n ((n - 1) g_k + x / g_k^{n-1})
 *
 * @param x Real radicand. Negative values require an odd root degree.
 * @param root Root degree; zero returns `std::nullopt`.
 * @param g Positive initial guess.
 * @param tol Absolute reconstruction tolerance.
 * @return Approximation or `std::nullopt` for an invalid real-domain request.
 */
template <Real T>
constexpr std::optional<T> constexpr_nrt(T x, std::size_t root, T g = 1e-10,
                                         T tol = 1e-10) {
  if (root == 0) return std::nullopt;
  if (x == T(0)) return T(0);

  T sign = T(1);
  if (x < T(0)) {
    if (root % 2 == 0) return std::nullopt;
    sign = T(-1);
    x = -x;
  }

#if LINALG_HAS_CXX26_CONSTEXPR_CMATH
  (void)g;
  (void)tol;
  if (root == 1) return sign * x;
  if (root == 2) return sign * std::sqrt(x);
  if (root == 3) return sign * std::cbrt(x);
  return sign * std::pow(x, T(1) / T(root));
#else
  if (!(g > T(0))) g = T(1);

  auto pow_n = [&](T base, std::size_t n) {
    T pow(1);
    for (std::size_t i = 0; i < n; ++i) pow *= base;
    return pow;
  };

  for (std::size_t k = 0; k < MAX_STEP_EVAL; ++k) {
    const auto gkm1 = pow_n(g, root - 1);
    if (!(gkm1 > T(0))) return std::nullopt;

    g = (T(root - 1) * g + constexpr_div_safe(x, gkm1, g)) / T(root);
    if (constexpr_abs(pow_n(g, root) - x) <= tol) return sign * g;
  }
  return sign * g;
#endif
}

/**
 * @brief Return a scalar nth root with a sanitized fallback radicand.
 *
 * @param x input scalar
 * @param root root degree
 * @param floor fallback scalar
 * @param g initial guess
 * @param tol convergence tolerance
 *
 * @return scalar nth-root with optional fallback handling
 */
template <Real T>
constexpr T constexpr_nrt_safe(T x, std::size_t root, T floor = T(0),
                               T g = 1e-10, T tol = 1e-10) {
  if (const auto r = constexpr_nrt(x, root, g, tol); r.has_value()) return *r;

  const auto floor_safe = (floor >= T(0)) ? floor : T(0);
  if (const auto rf = constexpr_nrt(floor_safe, root, g, tol); rf.has_value())
    return *rf;

  return T(0);
}

/**
 * @brief Approximate sine with a truncated Taylor recurrence.
 *
 * sum_{k=0}^{infinity}frac{(-1)^{k}x^{2k+1}}{(2k+1)!}
 *
 * @param x Angle in radians.
 * @param terms Number of series terms.
 * @return Sine approximation.
 */
template <Real T>
constexpr auto constexpr_sin(T x, int terms = 16) {
#if LINALG_HAS_CXX26_CONSTEXPR_CMATH
  (void)terms;
  return std::sin(x);
#else
  T sum = x, term = x;
  for (int k = 1; k < terms; ++k) {
    term *= (-(x * x)) / (T(2 * k) * T(2 * k + 1));
    sum += term;
  }
  return sum;
#endif
}

/**
 * @brief Approximate cosine with a truncated Taylor recurrence.
 *
 * sum_{k=0}^{infinity}frac{(-1)^{k}x^{2k}}{(2k)!} = cos x
 *
 * @param x Angle in radians.
 * @param terms Number of series terms.
 * @return Cosine approximation.
 */
template <Real T>
constexpr auto constexpr_cos(T x, int terms = 16) {
#if LINALG_HAS_CXX26_CONSTEXPR_CMATH
  (void)terms;
  return std::cos(x);
#else
  T sum = T(1), term = T(1);
  for (int k = 1; k < terms; ++k) {
    term *= (-(x * x)) / (T(2 * k - 1) * T(2 * k));
    sum += term;
  }
  return sum;
#endif
}

/**
 * @brief atanh (inverse hyperbolic tangent function)
 *
 * atanh(x) = x + x^3 / 3 + x^5 / 5 + ... = sum_{n=0}^{infinity}
 * frac{x^{2n+1}}{2n+1}
 *
 * @param x in (-1, 1)
 * @param terms Number of odd-power series terms.
 * @return y in R
 */
template <Real T>
constexpr std::optional<T> constexpr_atanh(T x, int terms = 16) {
  if (constexpr_abs(x) >= T(1)) return std::nullopt;
#if LINALG_HAS_CXX26_CONSTEXPR_CMATH
  (void)terms;
  return std::atanh(x);
#else
  const T x2 = x * x;
  T term(x), sum(x);
  for (int k = 1; k < terms; ++k) {
    term *= x2;
    sum += (term / T(2 * k + 1));
  }
  return sum;
#endif
}

/**
 * @brief constexpr-safe atanh with fallback
 *
 * @param x input scalar
 * @param fallback fallback value when |x| >= 1
 * @param terms series terms
 *
 * @return scalar atanh result or fallback
 */
template <Real T>
constexpr T constexpr_atanh_safe(T x, T fallback = T(0), int terms = 16) {
  if (const auto r = constexpr_atanh(x, terms); r.has_value()) return *r;
  return fallback;
}

/**
 * @brief Approximate `exp(i * theta)` on the complex unit circle.
 *
 * @param theta Angle in radians.
 * @param terms Number of sine and cosine series terms.
 * @return Complex unit-circle approximation.
 */
template <Real T>
constexpr auto constexpr_expi(T theta, int terms = 16) {
#if LINALG_HAS_CXX26_CONSTEXPR_COMPLEX_MATH
  (void)terms;
  return std::exp(std::complex<T>(T(0), theta));
#else
  return std::complex<T>(constexpr_cos(theta, terms),
                         constexpr_sin(theta, terms));
#endif
}

/**
 * @brief Approximate the real exponential with a truncated power series.
 *
 * exp(x) = 1 + x + x^2/2! + x^3/3!... = sum_{n=0}^infinity
 * frac{x^n}{n!}
 *
 * @param x Real exponent.
 * @param terms Number of series terms.
 * @return Real exponential approximation.
 */
template <Real T>
constexpr auto constexpr_exp(T x, int terms = 16) {
#if LINALG_HAS_CXX26_CONSTEXPR_CMATH
  (void)terms;
  return std::exp(x);
#else
  T sum(1), term(1);
  for (int k = 1; k < terms; ++k) {
    term *= x / k;
    sum += term;
  }
  return sum;
#endif
}

/**
 * @brief Approximate the exponential of a complex scalar.
 *
 * @param lambda Complex exponent `alpha + i * omega`.
 * @param terms Number of series terms used by each component approximation.
 * @return Complex exponential approximation.
 */
template <Complex T>
constexpr auto constexpr_exp_lambda(const T& lambda, int terms = 16) {
#if LINALG_HAS_CXX26_CONSTEXPR_COMPLEX_MATH
  (void)terms;
  return std::exp(lambda);
#else
  return constexpr_exp(lambda.real()) * constexpr_expi(lambda.imag(), terms);
#endif
}

/**
 * @brief natural log
 *
 * [old 1] less numerically friendly
 * Taylor series expansion for 1 + t, t = 0
 * log(1 + t) = t - t^2 / 2 + t^3 / 3 - ... =
 * sum_{k=1}^{infinity}(-1)^{k+1}frac{t^k}{k}
 *
 * Restricting x in R_++, and let t = x - 1... log(1 + x - 1) = log(x)
 * log(x) approximately (x - 1) - (x - 1)^2 / 2 + (x - 1)^3 - ...
 *
 * [old 2] use atanh
 * log(x) approximately 2 * atan(y) where y = (x - 1) / (x + 1)
 *
 * [new] range reduce x = m * 2^e where m in [0.5, 2]
 * for all x > 0 we have x = m 2^e, m in [0.5, 1] and e in Z
 * ln(x) = ln(m) + e ln(2)
 * ** the above makes this much more numerically stable
 *
 * - perform atan(y) as before
 * - sum 2a with e*ln(2)
 *
 * @param x_in Positive real argument.
 * @param terms Number of inverse-hyperbolic-tangent series terms.
 * @return Logarithm approximation, or `std::nullopt` outside the positive
 * domain.
 */
template <Real T>
constexpr std::optional<T> constexpr_ln(T x_in, int terms = 16) {
  if (!(x_in > T(0))) return std::nullopt;

#if LINALG_HAS_CXX26_CONSTEXPR_CMATH
  (void)terms;
  return std::log(x_in);
#else
  T x = x_in;
  int e = 0;

  for (int i = 0; i < 64 && x > T(2); ++i) {
    x *= T(0.5);
    ++e;
  }
  for (int i = 0; i < 64 && x < T(0.5); ++i) {
    x *= T(2);
    --e;
  }

  const T y = constexpr_div_safe(x - T(1), x + T(1), T(0));
  const auto a = constexpr_atanh(y, terms);
  if (!a) return std::nullopt;

  return T(2) * (*a) + T(e) * ln2_v<T>;
#endif
}

/**
 * @brief natural log with domain sanitization and fallback
 *
 * Ensures no optional access is required at call sites that need a scalar.
 * Handles NaN/invalid by clamping to a positive floor.
 *
 * @param x input scalar
 * @param floor positive floor value used when x is invalid/non-positive
 * @param terms series terms
 *
 * @return ln(max(x, floor)) with NaN-safe clamping semantics
 */
template <Real T>
constexpr T constexpr_ln_safe(T x, T floor = std::numeric_limits<T>::min(),
                              int terms = 16) {
  const T floor_safe = (floor > T(0)) ? floor : std::numeric_limits<T>::min();
  const T x_safe = (x > floor_safe) ? x : floor_safe;
  const auto l = constexpr_ln(x_safe, terms);
  if (l.has_value()) return *l;

  const auto l_floor = constexpr_ln(floor_safe, terms);
  return l_floor.has_value() ? *l_floor : T(0);
}

/**
 * @brief constexpr for loop which recursively unrolls a dynamic for loop into
 * compile time recursive loop
 *
 * @tparam Start Inclusive compile-time start index.
 * @tparam End Exclusive compile-time end index.
 * @param f Callable receiving each index as `std::integral_constant`.
 */
template <std::size_t Start, std::size_t End, typename Func>
  requires(End - Start >= 0)
constexpr void constexpr_for(Func&& f) {
  if constexpr (Start < End) {
    f(std::integral_constant<std::size_t, Start>{});
    constexpr_for<Start + 1, End>(std::forward<Func>(f));
  }
}

/**
 * @brief Compare complex scalars by magnitude of their difference.
 *
 * @param a First complex scalar.
 * @param b Second complex scalar.
 * @param tol Strict magnitude tolerance.
 */
template <Complex T>
constexpr bool is_same(const T& a, const T& b, real_type_t<T> tol = 1e-10) {
  return constexpr_abs(a - b) < tol;
}

/**
 * @brief Return coordinate indices sorted by descending real value.
 *
 * @param x Fixed-size real vector.
 * @return `std::array<std::size_t, N>` containing a descending permutation.
 */
template <VectorContainer Vec>
  requires(Real<vector_element_t<Vec>>)
constexpr auto argsort_desc(const Vec& x) {
  static constexpr std::size_t N = vector_dim_v<Vec>;

  std::array<std::size_t, N> p;
  for (std::size_t i = 0; i < N; ++i) p[i] = i;
  for (std::size_t i = 0; i + 1 < N; ++i) {
    std::size_t largest = i;
    for (std::size_t j = i + 1; j < N; ++j)
      if (x(p[j]) > x(p[largest])) largest = j;
    if (largest != i) std::swap(p[i], p[largest]);
  }
  return p;
}

} // namespace linalg
