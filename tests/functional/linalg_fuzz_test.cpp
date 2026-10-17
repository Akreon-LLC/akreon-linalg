/*
 * Copyright © 2025-2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <charconv>
#include <cmath>
#include <string_view>
#include <cstdio>
#include <random>

#include "akreon/linalg/linalg.hpp"

namespace functional_test {

constexpr std::size_t DefaultTrials = 100;
constexpr std::size_t MaxOrder = 8;
constexpr unsigned Seed = 42;
constexpr std::size_t MaxDiagnosticsPerShape = 3;

using namespace linalg;

template <typename T>
void print_scalar(const T& x) {
  if constexpr (is_complex_v<T>)
    std::printf("(%g, %g) ", static_cast<double>(x.real()),
                static_cast<double>(x.imag()));
  else std::printf("%g ", static_cast<double>(x));
}

template <typename T, std::size_t Rows, std::size_t Cols>
void print_mat(const Matrix<T, Rows, Cols>& A) {
  for (std::size_t row = 0; row < Rows; ++row) {
    for (std::size_t col = 0; col < Cols; ++col) print_scalar(A(row, col));
    std::printf("\n");
  }
}

template <typename T, std::size_t Rows, std::size_t Cols>
auto make_A(std::mt19937& rng, std::uniform_real_distribution<T>& dist) {
  Matrix<T, Rows, Cols> A;
  for (std::size_t i = 0; i < Rows; ++i)
    for (std::size_t j = 0; j < Cols; ++j) A(i, j) = dist(rng);
  return A;
}

template <typename T, std::size_t N>
auto make_vec(std::mt19937& rng, std::uniform_real_distribution<T>& dist) {
  Vector<T, N> b;
  for (std::size_t i = 0; i < N; ++i) b(i) = dist(rng);
  return b;
}

template <typename T, std::size_t Rows, std::size_t Cols>
  requires(Field<T> && !is_complex_v<T>)
std::size_t check_decompositions(real_type_t<T> tol, std::size_t trials) {
  std::mt19937 rng(Seed);
  std::size_t failures = 0, failed_matrices = 0;
  std::uniform_real_distribution<T> dist(-10.0, 10.0);
  for (std::size_t i = 0; i < trials; ++i) {
    const auto A = make_A<T, Rows, Cols>(rng, dist);
    bool failed = false;
    auto fail = [&](const char* operation) {
      ++failures;
      failed = true;
      if (failed_matrices >= MaxDiagnosticsPerShape) return;
      std::fprintf(
          stderr,
          "seed=%u scalar=%s rows=%zu cols=%zu trial=%zu check=%s tol=%g\n",
          Seed, std::same_as<T, float> ? "float" : "double", Rows, Cols, i,
          operation, static_cast<double>(tol));
      print_mat(A);
    };
    const auto res = QR(A);
    const auto Q = res.Q();
    const auto R = res.R();
    static_assert(std::same_as<decltype(Q), const Matrix<T, Rows, Rows>>);
    static_assert(std::same_as<decltype(R), const Matrix<T, Rows, Cols>>);
    const auto A_prime = Q * R;
    if (!all_equal(A_prime, A, tol)) fail("QR reconstruction");

    if (!all_equal(transpose(Q) * Q, make_identity<T, Rows>(), tol))
      fail("QR orthogonality");

    if constexpr (Rows == Cols) {
      const auto eig_decomp = EigenDecomposition(A);
      if (!eig_decomp.converged) fail("eigen convergence");
      else {
        T scale = 1;
        for (const auto& value : A)
          scale = std::max(scale, constexpr_abs(value));
        if (!all_equal(as_complex_matrix(A) * eig_decomp.V,
                       eig_decomp.V * make_diag(eig_decomp.lambda),
                       tol * scale))
          fail("eigenpair residual");
      }
    }

    const auto svd = SVD(A);
    const auto U = svd.U;
    const auto S = svd.S;
    const auto VT = svd.VT;
    static_assert(
        std::same_as<decltype(U), const Matrix<std::complex<T>, Rows, Rows>>);
    static_assert(
        std::same_as<decltype(VT), const Matrix<std::complex<T>, Cols, Cols>>);
    static_assert(std::same_as<decltype(S), const Matrix<T, Rows, Cols>>);

    // Both successful convergence and reconstruction are required. The former
    // condition used && with a positive equality check, hiding bad SVD results.
    if (!svd.converged ||
        !all_equal(as_complex_matrix(A), U * (as_complex_matrix(S) * VT), tol))
      fail("SVD convergence or reconstruction");

    // S is rectangular: inspect its diagonal, not structural off-diagonal
    // zeros.
    for (std::size_t j = 0; j < std::min(Rows, Cols); ++j)
      if (!std::isfinite(S(j, j)) || S(j, j) < 0)
        fail("finite nonnegative singular values");

    const auto x = make_vec<T, Cols>(rng, dist);
    const auto b = A * x;

    const auto [x_tilde, flag] = res.solve(b);
    static_cast<void>(flag);
    const auto lsq_fit = A * x_tilde;
    const auto residual = euclidean_distance(lsq_fit, b);
    if (!std::isfinite(residual) || residual > tol)
      fail("least-squares residual");
    failed_matrices += failed;
  }
  std::printf("scalar=%s rows=%zu cols=%zu: %zu/%zu matrices passed, "
              "failed_checks=%zu\n",
              std::same_as<T, float> ? "float" : "double", Rows, Cols,
              trials - failed_matrices, trials, failures);

  return failures;
}

// Compare against a scale-aware bound and reject NaN/Inf explicitly.
template <typename A, typename B>
bool close_matrix(const A& actual, const B& expected, long double tol) {
  long double scale = 1, error = 0;
  for (std::size_t i = 0; i < mat_rows_v<A>; ++i) {
    for (std::size_t j = 0; j < mat_cols_v<A>; ++j) {
      const std::complex<long double> a(actual(i, j));
      const std::complex<long double> b(expected(i, j));
      if (!std::isfinite(a.real()) || !std::isfinite(a.imag()) ||
          !std::isfinite(b.real()) || !std::isfinite(b.imag()))
        return false;
      scale = std::max(scale, std::abs(b));
      error = std::max(error, std::abs(a - b));
    }
  }
  return error <= tol * scale;
}

template <typename T, std::size_t N>
struct SquareChecks {
  std::size_t failures = 0, cases = 0, failed_cases = 0;
  void check(const Matrix<T, N, N>& A, std::size_t trial, const char* family,
             const auto& verify) {
    bool failed = false;
    auto require = [&](bool passed, const char* operation) {
      if (passed) return;
      ++failures;
      failed = true;
      if (failed_cases >= MaxDiagnosticsPerShape) return;
      std::fprintf(
          stderr,
          "seed=%u scalar=%s%s order=%zu trial=%zu family=%s check=%s\n", Seed,
          is_complex_v<T> ? "complex-" : "",
          std::same_as<real_type_t<T>, float> ? "float" : "double", N, trial,
          family, operation);
      print_mat(A);
    };
    verify(require);
    ++cases;
    failed_cases += failed;
  }
  void report(const char* suite) const {
    std::printf(
        "%s scalar=%s%s order=%zu: %zu/%zu cases passed, failed_checks=%zu\n",
        suite, is_complex_v<T> ? "complex-" : "",
        std::same_as<real_type_t<T>, float> ? "float" : "double", N,
        cases - failed_cases, cases, failures);
  }
};

template <typename T, std::size_t N>
std::size_t check_symmetric_factors(T tol, std::size_t trials) {
  std::mt19937 rng(Seed);
  std::uniform_real_distribution<T> dist(T(-0.25), T(0.25));
  SquareChecks<T, N> checks;
  const auto I = make_identity<T, N>();
  for (std::size_t trial = 0; trial < trials; ++trial) {
    // Known, well-conditioned triangular factors exercise dense SPD matrices
    // over several scales without violating the unpivoted algorithms' contract.
    auto G = I;
    auto L = I;
    Matrix<T, N, N> D;
    T determinant = 1;
    const T scale = (trial / 3) % 3 == 0   ? T(0.01)
                    : (trial / 3) % 3 == 1 ? T(1)
                                           : T(100);
    for (std::size_t i = 0; i < N; ++i) {
      G(i, i) = T(1.5) + dist(rng);
      // Positive definite, negative definite, and mixed-sign inertia.
      const T sign = trial % 3 == 0   ? T(1)
                     : trial % 3 == 1 ? T(-1)
                                      : (i % 2 ? T(-1) : T(1));
      D(i, i) = scale * sign * (T(1.5) + dist(rng));
      determinant *= D(i, i);
      for (std::size_t j = 0; j < i; ++j) {
        G(i, j) = dist(rng);
        L(i, j) = dist(rng);
      }
    }
    const auto spd = scale * (G * transpose(G));
    checks.check(spd, trial, "SPD", [&](auto require) {
      const auto chol = Cholesky(spd);
      const auto recovered = chol.G();
      require(chol.is_valid() && chol.is_pd(),
              "Cholesky positive definiteness");
      require(close_matrix(recovered * transpose(recovered), spd, tol),
              "Cholesky reconstruction");
      // The constexpr square root uses a 1e-10 absolute radicand tolerance.
      require(
          close_matrix(recovered, std::sqrt(scale) * G, std::max(tol, T(1e-8))),
          "Cholesky known lower factor");
      for (std::size_t i = 0; i < N; ++i) {
        require(std::isfinite(recovered(i, i)) && recovered(i, i) > T(0),
                "Cholesky finite positive diagonal");
        for (std::size_t j = i + 1; j < N; ++j)
          require(recovered(i, j) == T(0),
                  "Cholesky lower triangular structure");
      }
    });
    const auto A = L * D * transpose(L);
    checks.check(A, trial, "symmetric signed pivots", [&](auto require) {
      const auto ldl = LDL(A);
      require(ldl.is_valid(), "LDL valid factorization");
      if (!ldl.is_valid()) return;
      require(trial % 3 == 0   ? ldl.is_pd()
              : trial % 3 == 1 ? ldl.is_nd()
                               : ldl.is_ind(),
              "LDL inertia");
      require(close_matrix(ldl.L() * ldl.D() * transpose(ldl.L()), A, tol),
              "LDL reconstruction");
      require(close_matrix(ldl.L(), L, tol) && close_matrix(ldl.D(), D, tol),
              "LDL known factors");
      require(std::isfinite(ldl.det()) &&
                  std::abs((ldl.det() - determinant) / determinant) <= tol,
              "LDL determinant");
      const auto x = make_vec<T, N>(rng, dist);
      const auto b = A * x;
      const auto [solved, event] = ldl.solve(b);
      require(event == NumericalEvent::NONE, "LDL solve status");
      require(close_matrix(as_col_matrix(solved), as_col_matrix(x), tol),
              "LDL known solution");
      require(close_matrix(as_col_matrix(A * solved), as_col_matrix(b), tol),
              "LDL solve residual");
      const auto [inverse, inverse_event] = ldl.apply(I);
      require(inverse_event == NumericalEvent::NONE &&
                  close_matrix(A * inverse, I, tol),
              "LDL multiple RHS solve");
    });
    // An exact zero final pivot must be rejected, even with random other
    // pivots.
    auto singular = D;
    singular(N - 1, N - 1) = 0;
    checks.check(singular, trial, "singular diagonal", [&](auto require) {
      require(!LDL(singular).is_valid(), "LDL singular rejection");
    });
  }
  checks.report("Cholesky/LDL");
  return checks.failures;
}

// Independent oracle: bounded inputs have ||A||_inf <= 2, so 48 Taylor terms
// in complex long double give ample margin over float/double error bounds.
// This reference uses neither eigendecomposition nor matrix_exp.
template <typename T, std::size_t N>
auto exp_series_reference(const Matrix<T, N, N>& A) {
  using C = std::complex<long double>;
  Matrix<C, N, N> input;
  for (std::size_t i = 0; i < N; ++i)
    for (std::size_t j = 0; j < N; ++j) input(i, j) = C(A(i, j));
  auto term = make_identity<C, N>();
  auto result = term;
  for (std::size_t k = 1; k <= 48; ++k) {
    term = (C(1) / C(k)) * (term * input);
    result = result + term;
  }
  return result;
}

template <typename T, std::size_t N>
std::size_t check_matrix_exponential(real_type_t<T> tol, std::size_t trials) {
  using R = real_type_t<T>;
  std::mt19937 rng(Seed);
  std::uniform_real_distribution<R> dist(R(-0.5), R(0.5));
  auto random_scalar = [&]() -> T {
    if constexpr (is_complex_v<T>) return T(dist(rng), dist(rng));
    else return dist(rng);
  };
  SquareChecks<T, N> checks;
  for (std::size_t trial = 0; trial < trials; ++trial) {
    Matrix<T, N, N> dense, diagonal, jordan;
    const auto lambda = random_scalar();
    for (std::size_t i = 0; i < N; ++i) {
      diagonal(i, i) = random_scalar();
      jordan(i, i) = lambda;
      if (i + 1 < N) jordan(i, i + 1) = random_scalar() / T(N);
      for (std::size_t j = 0; j < N; ++j) dense(i, j) = random_scalar() / T(N);
    }
    auto verify = [&](const auto& A, const char* family) {
      checks.check(A, trial, family, [&](auto require) {
        const auto result = matrix_exp(A);
        require(close_matrix(result, exp_series_reference(A), tol),
                "matrix exponential independent series reference");
        const auto inverse = matrix_exp(T(-1) * A);
        require(close_matrix(inverse, exp_series_reference(T(-1) * A), tol),
                "negative matrix exponential independent series reference");
        require(close_matrix(result * inverse,
                             make_identity<std::complex<R>, N>(),
                             R(16 * N) * tol),
                "exp(A) exp(-A) identity");
      });
    };
    verify(dense, "dense bounded");
    verify(diagonal, "diagonal");
    verify(jordan, "repeated eigenvalue Jordan chain");
    // Real skew-symmetric inputs include complex-conjugate eigenvalues;
    // Hermitian inputs cover orthogonal/unitary eigenvectors.
    if constexpr (!is_complex_v<T>) {
      verify(T(0.5) * (dense - transpose(dense)), "skew symmetric");
      verify(T(0.5) * (dense + transpose(dense)), "symmetric");
    }
  }
  checks.report("matrix_exp");
  return checks.failures;
}

} // namespace functional_test

int main(int argc, char** argv) {
  using namespace functional_test;
  std::size_t trials = DefaultTrials;
  if (argc > 2) return 2;
  if (argc == 2) {
    const std::string_view argument(argv[1]);
    const auto parsed = std::from_chars(
        argument.data(), argument.data() + argument.size(), trials);
    if (parsed.ec != std::errc{} ||
        parsed.ptr != argument.data() + argument.size() || trials == 0) {
      std::fprintf(stderr, "usage: linalg_fuzz_test [positive trial count]\n");
      return 2;
    }
  }
  std::size_t failures = 0, matrices = 0;
  auto check = [&]<std::size_t Rows, std::size_t Cols>() {
    failures += check_decompositions<double, Rows, Cols>(1e-10, trials);
    failures += check_decompositions<float, Rows, Cols>(1e-3f, trials);
    matrices += 2 * trials;
  };
  // Dense square matrices exercise general eigen convergence as well as QR,
  // SVD and consistent least-squares solves. Tall matrices skip eigen analysis.
  constexpr_for<2, MaxOrder>(
      [&]<std::size_t N>(std::integral_constant<std::size_t, N>) {
        check.template operator()<N, N>();
        failures += check_symmetric_factors<double, N>(1e-10, trials);
        failures += check_symmetric_factors<float, N>(1e-4f, trials);
      });
  constexpr_for<3, MaxOrder>(
      [&]<std::size_t N>(std::integral_constant<std::size_t, N>) {
        check.template operator()<N, 2>();
      });
  constexpr_for<1, MaxOrder>([&]<std::size_t N>(
                                 std::integral_constant<std::size_t, N>) {
    failures += check_matrix_exponential<double, N>(1e-8, trials);
    failures += check_matrix_exponential<float, N>(1e-3f, trials);
    failures += check_matrix_exponential<std::complex<double>, N>(1e-8, trials);
    failures += check_matrix_exponential<std::complex<float>, N>(1e-3f, trials);
  });
  std::printf(
      "linalg_fuzz_test: seed=%u decomposition_matrices=%zu failed_checks=%zu "
      "(QR, eigen convergence, SVD, least squares, Cholesky, LDL, matrix "
      "exp)\n",
      Seed, matrices, failures);
  return failures == 0 ? 0 : 1;
}
