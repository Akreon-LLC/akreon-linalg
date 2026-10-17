/*
 * Copyright © 2025-2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <iostream>

#include "akreon/linalg/linalg.hpp"
#include <akreon/linalg/test_helpers.hpp>

using namespace linalg;
using namespace unit_test;

TEST_CASE("Matrix exponential solves constant linear ODEs", "[dot(x) = Ax]") {
  SECTION("Zero matrix -> Identity") {
    constexpr auto Z = Matrix<double, 2, 2>{};
    constexpr auto res = matrix_exp(Z);
    constexpr auto I = make_identity<std::complex<double>, 2>();
    STATIC_REQUIRE(all_equal(res, I));
  }

  SECTION("Identity matrix -> e * I") {
    constexpr auto I = make_identity<std::complex<double>, 2>();
    constexpr auto res = matrix_exp(I);
    constexpr auto expected = constexpr_exp(1.0) * I;
    STATIC_REQUIRE(all_equal(res, expected));
  }

  SECTION("Matrix exponential of Jordan block") {
    constexpr auto A =
        Matrix<double, 2, 2>{Vectord<2>{2.0, 1.0}, Vectord<2>{0.0, 2.0}};
    constexpr double e2 = constexpr_exp(2.0);
    constexpr auto expected = Matrix<std::complex<double>, 2, 2>{
        Vectorcd<2>{e2, e2}, Vectorcd<2>{0.0, e2}};
    constexpr auto res = matrix_exp(A);
    STATIC_REQUIRE(all_equal(res, expected));
  }

  SECTION("Three-element Jordan chain retains the quadratic term") {
    constexpr auto A =
        Matrixd<3, 3>{Vectord<3>{0.0, 0.5, 0.0}, Vectord<3>{0.0, 0.0, -0.75},
                      Vectord<3>{0.0, 0.0, 0.0}};
    constexpr auto expected = Matrixcd<3, 3>{Vectorcd<3>{1.0, 0.5, -0.1875},
                                             Vectorcd<3>{0.0, 1.0, -0.75},
                                             Vectorcd<3>{0.0, 0.0, 1.0}};
    constexpr auto result = matrix_exp(A);
    STATIC_REQUIRE(all_equal(result, expected, 1e-8));
  }
}

TEST_CASE("Matrix-vector multiplication computes row inner products",
          "[y=Ax]") {
  constexpr auto x = Vectord<3>{3.0, -2.0, 1.0};
  constexpr auto A = Matrixd<3, 3>{x, x, x};
  constexpr auto y = A * x;
  STATIC_REQUIRE(y(0) == 14.0);
  STATIC_REQUIRE(y(1) == 14.0);
  STATIC_REQUIRE(y(2) == 14.0);
}

TEST_CASE("QR reconstructs real and complex matrices", "[A = QR]") {
  SECTION("Real A") {
    constexpr auto x1 = Vectord<3>{1.0, 1.0, 0.0};
    constexpr auto x2 = Vectord<3>{1.0, 0.0, 1.0};
    constexpr auto x3 = Vectord<3>{0.0, 1.0, 1.0};
    constexpr auto A = Matrix<double, 3, 3>{x1, x2, x3};
    constexpr auto qr = QR(A);
    constexpr auto Q = qr.Q();
    constexpr auto R = qr.R();
    constexpr auto A_prime = Q * R;
    STATIC_REQUIRE(all_equal(A_prime, A));

    STATIC_REQUIRE(R(1, 0) == 0);
    STATIC_REQUIRE(R(2, 0) == 0);
    STATIC_REQUIRE(R(2, 1) == 0);

    constexpr auto QTQ = transpose(Q) * Q;
    STATIC_REQUIRE(all_equal(QTQ, make_identity<double, 3>()));
  }

  SECTION("Real A m > n") {
    constexpr auto x1 = Vectord<2>{1.0, 1.0};
    constexpr auto x2 = Vectord<2>{1.0, 0.0};
    constexpr auto x3 = Vectord<2>{1.0, 0.0};
    constexpr auto A = Matrixd<3, 2>{x1, x2, x3};
    constexpr auto qr = QR(A);
    constexpr auto Q = qr.Q();
    constexpr auto R = qr.R();
    STATIC_REQUIRE(std::same_as<decltype(Q), const Matrixd<3, 3>>);
    STATIC_REQUIRE(std::same_as<decltype(R), const Matrixd<3, 2>>);
    constexpr auto A_prime = Q * R;
    STATIC_REQUIRE(all_equal(A_prime, A));

    constexpr auto rank = qr.rank();
    STATIC_REQUIRE(rank == 2);

    constexpr auto QTQ = transpose(Q) * Q;
    constexpr auto QTQr = get_n_cols<0, rank>(get_n_rows<0, rank>(QTQ));
    STATIC_REQUIRE(all_equal(QTQr, make_identity<double, 2>()));
  }

  SECTION("Real A m < n") {
    constexpr auto x1 = Vectord<3>{1.0, 1.0, 0.0};
    constexpr auto x2 = Vectord<3>{0.0, 1.0, 1.0};
    constexpr auto A = Matrixd<2, 3>{x1, x2};
    constexpr auto qr = QR(transpose(A));
    constexpr auto Q = transpose(qr.Q());
    constexpr auto L = transpose(qr.R());
    STATIC_REQUIRE(std::same_as<decltype(Q), const Matrixd<3, 3>>);
    STATIC_REQUIRE(std::same_as<decltype(L), const Matrixd<2, 3>>);
    constexpr auto A_prime = L * Q;
    STATIC_REQUIRE(all_equal(A_prime, A));

    constexpr auto rank = qr.rank();
    STATIC_REQUIRE(rank == 2);
  }

  SECTION("Complex A") {
    constexpr auto a = std::complex<double>(2.0, 0.0);
    constexpr auto b = std::complex<double>(3.0, 1.0);
    constexpr auto c = std::complex<double>(4.0, 0.0);
    constexpr auto x1 = Vectorcd<2>{a, b};
    constexpr auto x2 = Vectorcd<2>{std::conj(b), c};
    constexpr auto A = Matrixcd<2, 2>{x1, x2};
    constexpr auto qr = QR(A);
    constexpr auto Q = qr.Q();
    constexpr auto R = qr.R();
    constexpr auto A_prime = Q * R;
    STATIC_REQUIRE(all_equal(A_prime, A));

    constexpr auto zero = std::complex<double>(0.0, 0.0);
    STATIC_REQUIRE(R(1, 0) == zero);

    constexpr auto QTQ = conjugate_transpose(Q) * Q;
    STATIC_REQUIRE(all_equal(QTQ, make_identity<std::complex<double>, 2>()));
    STATIC_REQUIRE(2 == qr.rank());
  }
}

TEST_CASE("Least-squares solve recovers real and complex solutions",
          "[R x = Q^T b]") {
  SECTION("Real A") {
    constexpr auto x1 = Vectord<2>{1.0, 1.0};
    constexpr auto x2 = Vectord<2>{1.0, 2.0};
    constexpr auto x3 = Vectord<2>{1.0, 3.0};
    constexpr auto A = Matrixd<3, 2>{x1, x2, x3};
    constexpr auto qr = QR(A);
    constexpr auto Q = qr.Q();
    constexpr auto R = qr.R();
    STATIC_REQUIRE(std::same_as<decltype(Q), const Matrixd<3, 3>>);
    STATIC_REQUIRE(std::same_as<decltype(R), const Matrixd<3, 2>>);

    constexpr auto rank = qr.rank();
    STATIC_REQUIRE(rank == 2);

    constexpr auto A_prime = Q * R;
    STATIC_REQUIRE(all_equal(A_prime, A));

    constexpr auto QTQ = transpose(Q) * Q;
    STATIC_REQUIRE(std::same_as<decltype(QTQ), const Matrixd<3, 3>>);
    constexpr auto QTQr = get_n_cols<0, rank>(get_n_rows<0, rank>(QTQ));
    STATIC_REQUIRE(all_equal(QTQr, make_identity<double, 2>()));

    constexpr auto b = Vectord<3>{3.0, 5.0, 7.0};
    constexpr auto solve_res = qr.solve(b);
    constexpr auto x_tilde = std::get<0>(solve_res);
    constexpr auto flag = std::get<1>(solve_res);
    static_cast<void>(flag);

    STATIC_REQUIRE(euclidean_distance(x_tilde, Vectord<2>{1.0, 2.0}) <= eps);
  }

  SECTION("Complex A") {
    // (dakre) adding small eps to im part but otherwise same as before
    constexpr auto a = std::complex<double>(1.0, 1e-8);
    constexpr auto b = std::complex<double>(2.0, 0.0);
    constexpr auto c = std::complex<double>(3.0, 0.0);
    constexpr auto x1 = Vectorcd<2>{a, a};
    constexpr auto x2 = Vectorcd<2>{a, b};
    constexpr auto x3 = Vectorcd<2>{a, c};
    constexpr auto A = Matrixcd<3, 2>{x1, x2, x3};
    constexpr auto qr = QR(A);
    constexpr auto Q = qr.Q();
    constexpr auto R = qr.R();
    STATIC_REQUIRE(std::same_as<decltype(Q), const Matrixcd<3, 3>>);
    STATIC_REQUIRE(std::same_as<decltype(R), const Matrixcd<3, 2>>);

    constexpr auto rank = qr.rank();
    STATIC_REQUIRE(rank == 2);

    constexpr auto QTQ = transpose(Q) * Q;
    STATIC_REQUIRE(std::same_as<decltype(QTQ), const Matrixcd<3, 3>>);
    constexpr auto QTQr = get_n_cols<0, rank>(get_n_rows<0, rank>(QTQ));
    STATIC_REQUIRE(all_equal(QTQr, make_identity<std::complex<double>, 2>()));

    constexpr auto d = std::complex<double>(5.0, 0);
    constexpr auto e = std::complex<double>(7.0, 0);
    constexpr auto y = Vectorcd<3>{c, d, e};
    constexpr auto solve_res = qr.solve(y);
    constexpr auto x_tilde = std::get<0>(solve_res);
    constexpr auto flag = std::get<1>(solve_res);
    static_cast<void>(flag);
    STATIC_REQUIRE(euclidean_distance(x_tilde, Vectorcd<2>{1.0, 2.0}) <= eps);
  }

  SECTION("1x1 case") {
    constexpr auto X = Vector<double, 1>{0.0};
    constexpr auto A = Matrix<double, 1, 1>{X};
    constexpr auto qr = QR(A);
    constexpr auto Q = qr.Q();
    constexpr auto R = qr.R();
    STATIC_REQUIRE(Q(0, 0) == 1.0);
    STATIC_REQUIRE(R(0, 0) == 0.0);
  }
}

TEST_CASE(
    "Eigen decomposition recovers eigenpairs of real and complex matrices",
    "[A X = Lambda X]") {
  SECTION("Diagonal A") {
    constexpr auto diag = Vectord<3>{1.0, 2.0, 3.0};
    constexpr auto A = make_diag<double, 3>(diag);
    constexpr auto result = EigenDecomposition(A);
    STATIC_REQUIRE(result.converged);

    constexpr auto lambdas = result.lambda;
    STATIC_REQUIRE(lambdas(0).real() == diag(0));
    STATIC_REQUIRE(lambdas(1).real() == diag(1));
    STATIC_REQUIRE(lambdas(2).real() == diag(2));

    constexpr auto V = result.V;
    STATIC_REQUIRE(all_equal(V, make_identity<std::complex<double>, 3>()));

    constexpr auto rho = result.spectral_radius();
    STATIC_REQUIRE(rho == std::complex<double>(3.0));
  }

  SECTION("Symmetric A") {
    constexpr auto x1 = Vectord<2>{2.0, 1.0};
    constexpr auto x2 = Vectord<2>{1.0, 2.0};
    constexpr auto A = Matrixd<2, 2>{x1, x2};
    constexpr auto result = EigenDecomposition(A);
    constexpr auto expected =
        make_diag(Vectorcd<2>{std::complex(3.0, 0.0), std::complex(1.0, 0.0)});
    STATIC_REQUIRE(result.converged);

    constexpr auto V = result.V;
    constexpr auto VTV = conjugate_transpose(V) * V;
    STATIC_REQUIRE(all_equal(VTV, make_identity<std::complex<double>, 2>()));

    constexpr auto VTAV = conjugate_transpose(V) * (as_complex_matrix(A) * V);

    constexpr auto lambdas = result.lambda;
    STATIC_REQUIRE(all_equal(make_diag(lambdas), VTAV));

    constexpr auto VLVT = V * (make_diag(lambdas) * conjugate_transpose(V));
    STATIC_REQUIRE(all_equal(as_complex_matrix(A), VLVT));

    constexpr auto rho = result.spectral_radius();
    STATIC_REQUIRE(constexpr_abs(rho - expected(0, 0)) <= eps);
  }

  SECTION("Real A") {
    constexpr auto x1 = Vectord<3>{1.0, 2.0, 3.0};
    constexpr auto x2 = Vectord<3>{4.0, 5.0, 6.0};
    constexpr auto x3 = Vectord<3>{7.0, 8.0, 9.0};
    constexpr auto A = Matrixd<3, 3>{x1, x2, x3};
    constexpr auto result = EigenDecomposition(A);
    STATIC_REQUIRE(result.converged);

    constexpr auto expected = make_diag(
        Vectorcd<3>{std::complex(1.61168440e+01), std::complex(-1.11684397e+00),
                    std::complex(0.0)});
    constexpr auto lambdas = result.lambda;
    STATIC_REQUIRE(all_equal(make_diag(lambdas), expected));

    constexpr auto rho = result.spectral_radius();
    STATIC_REQUIRE(rho == lambdas(0));
  }

  SECTION("Complex hermitian A") {
    constexpr auto a = std::complex<double>(2.0, 0.0);
    constexpr auto b = std::complex<double>(3.0, 1.0);
    constexpr auto c = std::complex<double>(4.0, 0.0);
    constexpr auto x1 = Vectorcd<2>{a, b};
    constexpr auto x2 = Vectorcd<2>{std::conj(b), c};
    constexpr auto A = Matrixcd<2, 2>{x1, x2};

    constexpr auto res = EigenDecomposition(A);
    constexpr auto lambdas = res.lambda;
    STATIC_REQUIRE(res.converged);

    constexpr auto d = std::complex<double>(3 + constexpr_sqrt_safe(11.0));
    constexpr auto e = std::complex<double>(3 - constexpr_sqrt_safe(11.0));
    constexpr auto expected = make_diag(Vectorcd<2>{d, e});
    constexpr auto result = make_diag(lambdas);
    STATIC_REQUIRE(all_equal(result, expected));

    constexpr auto V = res.V;
    constexpr auto V1 = as_col_vector(V.row(0));
    constexpr auto V2 = as_col_vector(V.row(1));
    constexpr auto V1TV2 = conjugate_transpose(V1) * V2;
    STATIC_REQUIRE((V1TV2 * V1TV2)(0, 0).real() < eps);
    STATIC_REQUIRE((V1TV2 * V1TV2)(0, 0).imag() < eps);

    constexpr auto VTV = conjugate_transpose(V) * V;
    STATIC_REQUIRE(all_equal(VTV, make_identity<std::complex<double>, 2>()));

    constexpr auto VTAV = conjugate_transpose(V) * (as_complex_matrix(A) * V);
    STATIC_REQUIRE(all_equal(make_diag(lambdas), VTAV));

    constexpr auto VLVT = V * (result * conjugate_transpose(V));
    STATIC_REQUIRE(all_equal(A, VLVT));

    constexpr auto rho = res.spectral_radius();
    STATIC_REQUIRE(rho == lambdas(0));
  }
}

TEST_CASE("SVD reconstructs matrices and reports singular values",
          "[A = U Sigma VT]") {
  SECTION("Known SVD") {
    constexpr auto x1 = Vectord<2>{1.0, 0.0};
    constexpr auto x2 = Vectord<2>{0.0, 2.0};
    constexpr auto x3 = Vectord<2>{0.0, 0.0};
    constexpr auto A = Matrixd<3, 2>{x1, x2, x3};
    constexpr auto result = SVD(A);

    constexpr auto U = result.U;
    constexpr auto S = result.S;
    constexpr auto VT = result.VT;
    STATIC_REQUIRE(std::same_as<decltype(U), const Matrixcd<3, 3>>);
    STATIC_REQUIRE(std::same_as<decltype(S), const Matrixd<3, 2>>);
    STATIC_REQUIRE(std::same_as<decltype(VT), const Matrixcd<2, 2>>);
    STATIC_REQUIRE(euclidean_distance(get_diag(S), Vectord<2>{2.0, 1.0}) <=
                   eps);

    constexpr auto UTU = conjugate_transpose(U) * U;
    STATIC_REQUIRE(all_equal(get_n_rows<0, 2>(get_n_cols<0, 2>(UTU)),
                             make_identity<std::complex<double>, 2>()));

    constexpr auto VTV = VT * conjugate_transpose(VT);
    STATIC_REQUIRE(all_equal(VTV, make_identity<std::complex<double>, 2>()));

    constexpr auto USVT = U * (as_complex_matrix(S) * VT);
    STATIC_REQUIRE(all_equal(as_complex_matrix(A), USVT));
  }

  SECTION("Zeros case") {
    constexpr auto Z1 = make_zeros<double, 2, 2>();
    constexpr auto all_zeros = [](const auto& Z) {
      bool rc = true;
      for (const auto& elem : Z) rc &= elem == 0;
      return rc;
    };
    STATIC_REQUIRE(all_zeros(Z1));
    constexpr auto result = SVD(Z1);
    constexpr auto U = result.U;
    constexpr auto S = result.S;
    constexpr auto VT = result.VT;
    STATIC_REQUIRE(std::same_as<decltype(U), const Matrixcd<2, 2>>);
    STATIC_REQUIRE(std::same_as<decltype(S), const Matrixd<2, 2>>);
    STATIC_REQUIRE(std::same_as<decltype(VT), const Matrixcd<2, 2>>);
    STATIC_REQUIRE(euclidean_distance(get_diag(S), Vectord<2>{0.0, 0.0}) <=
                   eps);

    constexpr auto V1 = as_col_vector(VT.row(0));
    constexpr auto V2 = as_col_vector(VT.row(1));
    constexpr auto V1TV2 = conjugate_transpose(V1) * V2;
    STATIC_REQUIRE((V1TV2 * V1TV2)(0, 0).real() < eps);
    STATIC_REQUIRE((V1TV2 * V1TV2)(0, 0).imag() < eps);

    constexpr auto U1 = as_col_vector(U.row(0));
    constexpr auto U2 = as_col_vector(U.row(1));
    constexpr auto U1TU2 = conjugate_transpose(U1) * U2;
    STATIC_REQUIRE((U1TU2 * U1TU2)(0, 0).real() < eps);
    STATIC_REQUIRE((U1TU2 * U1TU2)(0, 0).imag() < eps);
  }

  SECTION("Complex hermitian A") {
    constexpr auto a = std::complex<double>(2.0, 0.0);
    constexpr auto b = std::complex<double>(3.0, 1.0);
    constexpr auto c = std::complex<double>(4.0, 0.0);
    constexpr auto x1 = Vectorcd<2>{a, b};
    constexpr auto x2 = Vectorcd<2>{std::conj(b), c};
    constexpr auto A = Matrixcd<2, 2>{x1, x2};

    constexpr auto d = double(3 + constexpr_sqrt_safe(11.0));
    constexpr auto e = constexpr_abs(double(3 - constexpr_sqrt_safe(11.0)));
    constexpr auto expected = make_diag(Vectord<2>{d, e});

    constexpr auto result = SVD(A);
    constexpr auto U = result.U;
    constexpr auto S = result.S;
    constexpr auto VT = result.VT;
    STATIC_REQUIRE(std::same_as<decltype(U), const Matrixcd<2, 2>>);
    STATIC_REQUIRE(std::same_as<decltype(S), const Matrixd<2, 2>>);
    STATIC_REQUIRE(std::same_as<decltype(VT), const Matrixcd<2, 2>>);

    // (dakre) since A is hermitian its singular values are equivalent to the
    // abs value of the eigenvalues of A
    STATIC_REQUIRE(all_equal(S, expected));

    constexpr auto I = make_identity<std::complex<double>, 2>();

    constexpr auto UTU = conjugate_transpose(U) * U;
    STATIC_REQUIRE(all_equal(UTU, I));

    constexpr auto VTV = VT * conjugate_transpose(VT);
    STATIC_REQUIRE(all_equal(VTV, I));

    constexpr auto USVT = U * (as_complex_matrix(S) * VT);
    STATIC_REQUIRE(all_equal(A, USVT));
  }
}

TEST_CASE(
    "Eigenvectors use their computed eigenvalues during back substitution",
    "[eigen][regression]") {
  constexpr auto A =
      Matrixd<3, 3>{Vectord<3>{2.0, 1.0, -1.0}, Vectord<3>{0.0, 3.0, 2.0},
                    Vectord<3>{0.0, 0.0, 5.0}};
  constexpr auto result = EigenDecomposition(A);
  STATIC_REQUIRE(result.converged);
  STATIC_REQUIRE(all_equal(as_complex_matrix(A) * result.V,
                           result.V * make_diag(result.lambda)));
}

TEST_CASE("Eigen QR deflates subdiagonals before reporting convergence",
          "[eigen][regression]") {
  // This matrix is already Hessenberg, but its nonzero subdiagonals mean
  // Hessenberg reduction alone cannot establish spectral convergence.
  constexpr auto A =
      Matrixd<3, 3>{Vectord<3>{2.0, 1.0, 0.0}, Vectord<3>{1.0, 2.0, 1.0},
                    Vectord<3>{0.0, 1.0, 2.0}};
  constexpr auto result = EigenDecomposition(A);
  STATIC_REQUIRE(result.converged);
  STATIC_REQUIRE(all_equal(as_complex_matrix(A) * result.V,
                           result.V * make_diag(result.lambda)));
  STATIC_REQUIRE(all_equal(conjugate_transpose(result.V) * result.V,
                           make_identity<std::complex<double>, 3>()));
  constexpr auto root2 = constexpr_sqrt_safe(2.0);
  STATIC_REQUIRE(constexpr_abs(result.lambda(0).real() - (2 + root2)) < eps);
  STATIC_REQUIRE(constexpr_abs(result.lambda(1).real() - 2) < eps);
  STATIC_REQUIRE(constexpr_abs(result.lambda(2).real() - (2 - root2)) < eps);

  constexpr config::FactorConfig<double> no_iterations{.steps = 0};
  constexpr auto unfinished = EigenDecomposition(A, no_iterations);
  STATIC_REQUIRE(!unfinished.converged);
  constexpr config::FactorConfig<double> configured{
      .steps = 128, .tol = 1e-12, .floor_eps = 1e-14};
  constexpr auto explicit_result = EigenDecomposition(A, configured);
  STATIC_REQUIRE(explicit_result.converged);
  STATIC_REQUIRE(
      all_equal(as_complex_matrix(A) * explicit_result.V,
                explicit_result.V * make_diag(explicit_result.lambda)));
}

TEST_CASE("Real Schur conjugate blocks preserve eigenpair residuals",
          "[eigen][regression]") {
  constexpr auto A =
      Matrixd<3, 3>{Vectord<3>{0.0, -2.0, 1.0}, Vectord<3>{2.0, 0.0, 3.0},
                    Vectord<3>{0.0, 0.0, 4.0}};
  constexpr auto result = EigenDecomposition(A);
  STATIC_REQUIRE(result.converged);
  STATIC_REQUIRE(all_equal(as_complex_matrix(A) * result.V,
                           result.V * make_diag(result.lambda)));
  STATIC_REQUIRE(constexpr_abs(result.lambda(0).imag() - 2.0) < eps);
  STATIC_REQUIRE(constexpr_abs(result.lambda(1).imag() + 2.0) < eps);
}

TEST_CASE("Scalar eigen decomposition needs no Hessenberg reduction",
          "[eigen][regression]") {
  constexpr auto A = Matrixd<1, 1>{Vectord<1>{-3.0}};
  constexpr auto result = EigenDecomposition(A);
  STATIC_REQUIRE(result.converged);
  STATIC_REQUIRE(result.lambda(0) == std::complex<double>(-3.0));
  STATIC_REQUIRE(result.V == make_identity<std::complex<double>, 1>());
}

TEST_CASE("QR preserves complex reflector phases in its packed representation",
          "[qr][regression]") {
  constexpr auto A =
      Matrixcd<3, 2>{Vectorcd<2>{std::complex<double>(1.0, 2.0),
                                 std::complex<double>(3.0, -1.0)},
                     Vectorcd<2>{std::complex<double>(-2.0, 1.0),
                                 std::complex<double>(0.0, 4.0)},
                     Vectorcd<2>{std::complex<double>(2.0, -3.0),
                                 std::complex<double>(1.0, 1.0)}};
  constexpr auto result = QR(A);
  STATIC_REQUIRE(all_equal(result.Q() * result.R(), A));
  STATIC_REQUIRE(all_equal(conjugate_transpose(result.Q()) * result.Q(),
                           make_identity<std::complex<double>, 3>()));
}

TEST_CASE("Complex triangular eigenvectors retain independent complex roots",
          "[eigen][regression]") {
  constexpr auto A =
      Matrixcd<2, 2>{Vectorcd<2>{std::complex<double>(1.0, -2.0),
                                 std::complex<double>(3.0, 1.0)},
                     Vectorcd<2>{std::complex<double>(0.0, 0.0),
                                 std::complex<double>(4.0, 3.0)}};
  constexpr auto result = EigenDecomposition(A);
  STATIC_REQUIRE(result.converged);
  STATIC_REQUIRE(all_equal(A * result.V, result.V * make_diag(result.lambda)));
}

TEST_CASE("Eigen deflation tolerance scales with the input matrix",
          "[eigen][regression]") {
  const auto A =
      Matrixd<3, 3>{Vectord<3>{2.0, 1.0, 0.0}, Vectord<3>{1.0, 2.0, 1.0},
                    Vectord<3>{0.0, 1.0, 2.0}};
  for (const double scale : {1e-6, 1.0, 1e6}) {
    CAPTURE(scale);
    const auto result = EigenDecomposition(scale * A);
    REQUIRE(result.converged);
    REQUIRE(all_equal(as_complex_matrix(A) * result.V,
                      result.V * make_diag((1.0 / scale) * result.lambda)));
  }
}

TEST_CASE(
    "Eigen decomposition rejects invalid configuration and nonfinite input",
    "[eigen][regression]") {
  constexpr auto A = make_identity<double, 2>();
  constexpr auto invalid =
      EigenDecomposition(A, config::FactorConfig<double>{.tol = -1.0});
  STATIC_REQUIRE(!invalid.converged);
  auto nonfinite = A;
  nonfinite(0, 0) = std::numeric_limits<double>::infinity();
  REQUIRE(!EigenDecomposition(nonfinite).converged);
}

TEST_CASE("Float SVD retains a small nonzero singular direction",
          "[svd][regression]") {
  // Forming A^T A can round its smallest eigenvalue to zero or below, even
  // though A has full rank. Recovering sigma from A*v preserves this direction.
  constexpr auto A = Matrix<float, 2, 2>{Vector<float, 2>{1.0f, 1.0f},
                                         Vector<float, 2>{1.0f, 1.0001f}};
  constexpr auto result = SVD(A);
  STATIC_REQUIRE(result.converged);
  STATIC_REQUIRE(result.S(1, 1) > 0.0f);
  STATIC_REQUIRE(all_equal(as_complex_matrix(A),
                           result.U * (as_complex_matrix(result.S) * result.VT),
                           1e-5f));
}

TEST_CASE(
    "Eigen shifts resolve a small coupled block beside a large eigenvalue",
    "[eigen][regression]") {
  constexpr auto A =
      Matrixd<3, 3>{Vectord<3>{2e-5, 1e-5, 0.0}, Vectord<3>{1e-5, 2e-5, 0.0},
                    Vectord<3>{0.0, 0.0, 4.0}};
  constexpr auto result = EigenDecomposition(A);
  STATIC_REQUIRE(result.converged);
  STATIC_REQUIRE(all_equal(as_complex_matrix(A) * result.V,
                           result.V * make_diag(result.lambda)));
  STATIC_REQUIRE(constexpr_abs(result.lambda(1).real() - 3e-5) < 1e-12);
  STATIC_REQUIRE(constexpr_abs(result.lambda(2).real() - 1e-5) < 1e-12);
}

TEST_CASE("Complex QR shifts resolve a conjugate spectrum off the real axis",
          "[eigen][regression]") {
  constexpr auto A = Matrixcd<2, 2>{
      Vectorcd<2>{std::complex<double>(0.0), std::complex<double>(-1.0)},
      Vectorcd<2>{std::complex<double>(1.0), std::complex<double>(0.0)}};
  constexpr auto result = EigenDecomposition(A);
  STATIC_REQUIRE(result.converged);
  STATIC_REQUIRE(all_equal(A * result.V, result.V * make_diag(result.lambda)));
}

TEST_CASE("Complex QR preserves a small leading phase", "[qr][regression]") {
  constexpr auto A =
      Matrixcd<2, 2>{Vectorcd<2>{0.003965974050843096, -0.9999921354939889},
                     Vectorcd<2>{0.9999921354939888, 0.003965974050843096}};
  constexpr auto qr = QR(A);
  constexpr auto Q = qr.Q();
  STATIC_REQUIRE(all_equal(conjugate_transpose(Q) * Q,
                           make_identity<std::complex<double>, 2>(), 1e-9));
  STATIC_REQUIRE(all_equal(Q * qr.R(), A, 1e-9));
  constexpr auto inverse = MatrixInverse<decltype(A)>()(A);
  STATIC_REQUIRE(std::get<1>(inverse) == NumericalEvent::NONE);
  STATIC_REQUIRE(all_equal(A * std::get<0>(inverse),
                           make_identity<std::complex<double>, 2>(), 1e-9));
}
