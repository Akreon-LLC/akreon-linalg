/*
 * Copyright © 2025-2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <iostream>

#include "akreon/linalg/linalg.hpp"
#include "akreon/linalg/operator/matrix_pencil.hpp"
#include <akreon/linalg/test_helpers.hpp>

using namespace linalg;
using namespace unit_test;

TEST_CASE(
    "Field accepts floating-point and complex scalars and rejects integers",
    "[field]") {
  STATIC_REQUIRE(Field<double>);
  STATIC_REQUIRE(Field<float>);
  STATIC_REQUIRE(Field<std::complex<double>>);
  STATIC_REQUIRE(!Field<int>);
  STATIC_REQUIRE(!Field<unsigned>);
}

TEST_CASE("Constexpr math evaluates known values and rejects invalid domains",
          "[math]") {
  constexpr auto res1 = constexpr_sqrt_safe(64.0) - 8.0;
  STATIC_REQUIRE(res1 < eps);
  STATIC_REQUIRE(constexpr_sqrt(0.0).value() == 0.0);
  STATIC_REQUIRE(constexpr_sqrt_safe(0.0) == 0.0);
  STATIC_REQUIRE(!constexpr_sqrt(-1.0).has_value());

  STATIC_REQUIRE(constexpr_pow(2.0, 0) == 1.0);
  STATIC_REQUIRE(constexpr_pow(2.0, 3) == 8.0);

  STATIC_REQUIRE(constexpr_nrt_safe<double>(0.0, 3) == 0.0);
  STATIC_REQUIRE(constexpr_abs(constexpr_nrt_safe<double>(1.0, 3) - 1.0) < eps);
  STATIC_REQUIRE(
      constexpr_abs(constexpr_nrt_safe<double>(3.0, 3) - 1.44224957031) < eps);
  STATIC_REQUIRE(constexpr_abs(constexpr_nrt_safe<double>(-8.0, 3) + 2.0) <
                 eps);
  STATIC_REQUIRE(!constexpr_nrt<double>(-8.0, 2).has_value());
  STATIC_REQUIRE(!constexpr_nrt<double>(8.0, 0).has_value());

  STATIC_REQUIRE(constexpr_abs(-2.0) == 2.0);

  STATIC_REQUIRE(constexpr_abs(constexpr_cos(0.0) - 1.0) < eps);
  STATIC_REQUIRE(constexpr_sin(0.0) < eps);

  STATIC_REQUIRE(constexpr_cos(M_PI / 2) < eps);
  STATIC_REQUIRE(constexpr_abs(constexpr_sin(M_PI / 2) - 1.0) < eps);

  STATIC_REQUIRE(constexpr_sin(M_PI) < eps);

  STATIC_REQUIRE(constexpr_sin(M_PI / 4) - constexpr_cos(M_PI / 4) < eps);

  constexpr auto a = constexpr_expi(M_PI / 6);
  STATIC_REQUIRE(a.real() == constexpr_cos(M_PI / 6));
  STATIC_REQUIRE(a.imag() == constexpr_sin(M_PI / 6));

  STATIC_REQUIRE(!constexpr_atanh(2.0).has_value());
  STATIC_REQUIRE(!constexpr_atanh(-1.0).has_value());
  STATIC_REQUIRE(!constexpr_atanh(1.0).has_value());
  STATIC_REQUIRE(constexpr_atanh(0.0).value() == 0.0);
  STATIC_REQUIRE(
      constexpr_abs(constexpr_atanh(0.5).value() - 0.549306144224055) < eps);

  constexpr auto b = std::complex<double>(1.0, -2.0);
  constexpr auto expected = 2.3607;
  STATIC_REQUIRE(L2<Vector<std::complex<double>, 1>>()(b) - expected < eps);

  constexpr auto V = Vectord<5>{1.0, 2.0, 3.0, 4.0, 5.0};
  constexpr auto P = argsort_desc(V);
  STATIC_REQUIRE(P == std::array<std::size_t, 5>{4, 3, 2, 1, 0});

  STATIC_REQUIRE(!constexpr_ln(-1.0).has_value());
  STATIC_REQUIRE(constexpr_ln(1.0).has_value());
  STATIC_REQUIRE(constexpr_ln(1.0).value() == 0.0);
  STATIC_REQUIRE(constexpr_abs(constexpr_ln(constexpr_exp(1.0)).value() - 1.0) <
                 eps);
  STATIC_REQUIRE(constexpr_ln(1e-10).has_value());
}

TEST_CASE("Vector and matrix arithmetic preserve values and shapes",
          "[vector_space]") {
  SECTION("Static vector checks") {
    constexpr auto A = Vectord<2>{1.0, 2.0};
    constexpr auto B = Vectord<2>{-1.0, 5.0};
    constexpr auto C = add(A, B);
    constexpr auto D = A + B;
    STATIC_REQUIRE(C == D);
    STATIC_REQUIRE(C(0) == 0.0);
    STATIC_REQUIRE(C(1) == 7.0);

    constexpr auto split = splice<1>(A);
    STATIC_REQUIRE(std::get<0>(split)(0) == 1.0);
    STATIC_REQUIRE(std::get<1>(split)(0) == 2.0);

    constexpr double alpha = 2.0;
    constexpr auto E = alpha * A;
    constexpr auto F = scalar_multiplication(alpha, A);
    STATIC_REQUIRE(E == F);
    STATIC_REQUIRE(E(0) == 2.0);
    STATIC_REQUIRE(E(1) == 4.0);

    STATIC_REQUIRE(9.0 == inner_product(A, B));

    constexpr std::complex<double> c(1.0, 2.0);
    constexpr auto X = Vectorcd<5>{c, c, c, c, c};
    constexpr auto result = inner_product(X, X);
    STATIC_REQUIRE(result == std::complex<double>{25, 0});

    constexpr auto G = A - B;
    STATIC_REQUIRE(G(0) == 2.0);
    STATIC_REQUIRE(G(1) == -3.0);
    STATIC_REQUIRE(inner_product(G, G) == 13.0);
    STATIC_REQUIRE(L2<Vector<double, 2>>()(G) > 0);
    STATIC_REQUIRE(L2<Vector<double, 2>>()(G) == euclidean_distance(A, B));
    STATIC_REQUIRE(L1<Vector<double, 2>>()(G) > 0);
    STATIC_REQUIRE(L1<Vector<double, 2>>()(G) == manhattan_distance(A, B));
    STATIC_REQUIRE(Linf<Vector<double, 2>>()(G) == 3.0);

    constexpr auto GT = transpose(G);
    STATIC_REQUIRE(GT(0, 0) == 2.0);
    STATIC_REQUIRE(GT(1, 0) == -3.0);
    constexpr auto GTT = transpose(GT);
    STATIC_REQUIRE(as_vector(GTT) == G);

    constexpr auto G_col_vec = as_col_vector(G);
    constexpr auto G_row_vec = as_row_vector(G);
    STATIC_REQUIRE(std::same_as<decltype(G_col_vec), const ColVectord<2>>);
    STATIC_REQUIRE(std::same_as<decltype(G_row_vec), const RowVectord<2>>);

    constexpr auto G_outer_prod = outer_product(G_col_vec, G_row_vec);
    STATIC_REQUIRE(G_outer_prod(0, 0) == 4.0);
    STATIC_REQUIRE(G_outer_prod(0, 1) == -6.0);
    STATIC_REQUIRE(G_outer_prod(1, 0) == -6.0);
    STATIC_REQUIRE(G_outer_prod(1, 1) == 9.0);

    constexpr auto Zeros = Vectord<3>(0.0, 0.0, 0.0);
    STATIC_REQUIRE(cross(Zeros, Zeros) == Zeros);

    constexpr auto ex = Vectord<3>(1.0, 0.0, 0.0);
    constexpr auto ey = Vectord<3>(0.0, 1.0, 0.0);
    constexpr auto ez = Vectord<3>(0.0, 0.0, 1.0);
    STATIC_REQUIRE(cross(ex, ey) == ez);
    STATIC_REQUIRE(cross(ey, ex) == -1.0 * ez);
    STATIC_REQUIRE(cross(ez, ex) == ey);
    STATIC_REQUIRE(cross(ex, ez) == -1.0 * ey);
    STATIC_REQUIRE(cross(ey, ez) == ex);
    STATIC_REQUIRE(cross(ez, ey) == -1.0 * ex);

    STATIC_REQUIRE(
        cross(Vectord<3>(1.0, 2.0, 3.0), Vectord<3>(4.0, 5.0, 6.0)) ==
        Vectord<3>(-3.0, 6.0, -3.0));
  }

  SECTION("Static matrix checks") {
    constexpr auto A = Vectord<2>{1.0, 2.0};
    constexpr auto B = Matrixd<2, 2>{A, A};
    constexpr auto C = B + B;
    constexpr auto D = add(B, B);

    STATIC_REQUIRE(min(A) == 1.0);
    STATIC_REQUIRE(max(A) == 2.0);
    STATIC_REQUIRE(arg_min(A) == 0);
    STATIC_REQUIRE(arg_max(A) == 1);

    constexpr auto B_inv_res = MatrixInverse<decltype(B)>()(B);
    constexpr auto B_inv = std::get<0>(B_inv_res);
    constexpr auto B_inv_event = std::get<1>(B_inv_res);
    STATIC_REQUIRE(B_inv_event ==
                   linalg::NumericalEvent::NEAR_SINGULAR_TRIANGULAR_SOLVE);
    constexpr auto B_B_inv = B * B_inv;
    STATIC_REQUIRE(B_B_inv == Matrixd<2, 2>{});

    // (dakre) this case acts like true inverse where Z is unique
    constexpr auto Z =
        Matrixd<2, 2>{Vectord<2>{1.0, 2.0}, Vectord<2>{3.0, 4.0}};
    constexpr auto Z_inv_res = MatrixInverse<decltype(Z)>()(Z);
    constexpr auto Z_inv = std::get<0>(Z_inv_res);
    constexpr auto Z_inv_event = std::get<1>(Z_inv_res);
    STATIC_REQUIRE(Z_inv_event == linalg::NumericalEvent::NONE);
    constexpr auto Z_Z_inv = Z * Z_inv;
    STATIC_REQUIRE(all_equal(Z_Z_inv, make_identity<double, 2>()));

    constexpr auto S =
        Matrixd<2, 2>{Vectord<2>{4.0, 2.0}, Vectord<2>{2.0, 3.0}};
    constexpr auto S_inv_res = SymmetricMatrixInverse<decltype(S)>()(S);
    constexpr auto S_inv = std::get<0>(S_inv_res);
    constexpr auto S_inv_event = std::get<1>(S_inv_res);
    STATIC_REQUIRE(S_inv_event == linalg::NumericalEvent::NONE);
    constexpr auto S_inv_exp =
        Matrixd<2, 2>{Vectord<2>{0.375, -0.25}, Vectord<2>{-0.25, 0.5}};
    STATIC_REQUIRE(all_equal(S_inv, S_inv_exp));
    STATIC_REQUIRE(all_equal(S * S_inv, make_identity<double, 2>()));
    STATIC_REQUIRE(all_equal(S_inv * S, make_identity<double, 2>()));
    STATIC_REQUIRE(all_equal(S_inv, transpose(S_inv)));

    constexpr auto S_1x1 = Matrixd<1, 1>{Vectord<1>{4.0}};
    constexpr auto S_1x1_inv_res =
        SymmetricMatrixInverse<decltype(S_1x1)>()(S_1x1);
    constexpr auto S_1x1_inv = std::get<0>(S_1x1_inv_res);
    constexpr auto S_1x1_inv_event = std::get<1>(S_1x1_inv_res);
    STATIC_REQUIRE(S_1x1_inv_event == linalg::NumericalEvent::NONE);
    STATIC_REQUIRE(S_1x1_inv(0, 0) == 0.25);

    constexpr auto S_1x1_zero = Matrixd<1, 1>{Vectord<1>{0.0}};
    constexpr auto S_1x1_zero_inv_res =
        SymmetricMatrixInverse<decltype(S_1x1_zero)>()(S_1x1_zero);
    constexpr auto S_1x1_zero_inv_event = std::get<1>(S_1x1_zero_inv_res);
    STATIC_REQUIRE(S_1x1_zero_inv_event ==
                   linalg::NumericalEvent::FACTORIZATION_FAILED);

    constexpr auto S_singular =
        Matrixd<2, 2>{Vectord<2>{1.0, 0.0}, Vectord<2>{0.0, 0.0}};
    constexpr auto S_singular_inv_res =
        SymmetricMatrixInverse<decltype(S_singular)>()(S_singular);
    constexpr auto S_singular_inv_event = std::get<1>(S_singular_inv_res);
    STATIC_REQUIRE(S_singular_inv_event ==
                   linalg::NumericalEvent::FACTORIZATION_FAILED);

    constexpr auto B_11 = get_block<0, 1, 0, 1>(B);
    STATIC_REQUIRE(B_11(0, 0) == 1.0);

    constexpr auto B_new =
        set_block<0, 1, 0, 1>(B, Matrix<double, 1, 1>{Vector<double, 1>{-2.0}});
    STATIC_REQUIRE(B_new(0, 0) == -2.0);
    STATIC_REQUIRE(B_new(0, 1) == 2.0);
    STATIC_REQUIRE(B_new(1, 0) == 1.0);
    STATIC_REQUIRE(B_new(1, 1) == 2.0);

    STATIC_REQUIRE(all_equal(C, D));
    STATIC_REQUIRE(C(0, 0) == 2.0);
    STATIC_REQUIRE(C(0, 1) == 4.0);
    STATIC_REQUIRE(C(1, 0) == 2.0);
    STATIC_REQUIRE(C(1, 1) == 4.0);

    STATIC_REQUIRE(as_vector(get_n_rows<0, 1>(B)) == get_row(B, 0));
    STATIC_REQUIRE(all_equal(get_n_rows<0, 2>(B), B));
    STATIC_REQUIRE(get_n_cols<0, 1>(B) == transpose(get_col(B, 0)));
    STATIC_REQUIRE(all_equal(get_n_cols<0, 2>(B), B));

    constexpr auto BR = get_upper_right_triangular_matrix(B);
    constexpr auto BL = get_lower_left_triangular_matrix(B);
    STATIC_REQUIRE(BR(1, 0) == 0.0);
    STATIC_REQUIRE(BL(0, 1) == 0.0);
    STATIC_REQUIRE(get_diag(BR) == get_diag(BL));

    constexpr auto B_col_0 = get_col(B, 0);
    STATIC_REQUIRE(B_col_0(0) == B(0, 0));
    STATIC_REQUIRE(B_col_0(1) == B(1, 0));
    constexpr auto B_col_1 = get_col(B, 1);
    STATIC_REQUIRE(B_col_1(0) == B(0, 1));
    STATIC_REQUIRE(B_col_1(1) == B(1, 1));
    constexpr auto B_row_0 = get_row(B, 0);
    STATIC_REQUIRE(B_row_0 == B.row(0));
    constexpr auto B_row_1 = get_row(B, 1);
    STATIC_REQUIRE(B_row_1 == B.row(1));

    constexpr double alpha = 3.0;
    constexpr auto E = scalar_multiplication(alpha, B);
    constexpr auto F = alpha * B;
    STATIC_REQUIRE(E == F);
    STATIC_REQUIRE(E(0, 0) == 3.0);
    STATIC_REQUIRE(E(0, 1) == 6.0);
    STATIC_REQUIRE(E(1, 0) == 3.0);
    STATIC_REQUIRE(E(1, 1) == 6.0);

    constexpr auto BT = transpose(B);
    STATIC_REQUIRE(rows_equal_cols(BT, B));

    constexpr auto G = Matrixd<1, 3>{Vectord<3>{1.0, 2.0, 3.0}};
    constexpr auto H =
        Matrixd<3, 1>{Vectord<1>{1.0}, Vectord<1>{2.0}, Vectord<1>{3.0}};

    constexpr auto GT = transpose(G);
    constexpr auto HT = transpose(H);
    STATIC_REQUIRE(G == HT);
    STATIC_REQUIRE(GT == H);

    STATIC_REQUIRE(trace(B) == 3.0);

    constexpr auto c = std::complex<double>(1.0, 2.0);  // 1 + 2i
    constexpr auto c_conj = std::conj(c);               // 1 - 2i
    constexpr auto d = std::complex<double>(5.0, -1.0); // 5 - 1i
    constexpr auto d_conj = std::conj(d);               // 5 + 1i
    constexpr auto J = Vectorcd<2>{c, d};
    constexpr auto K = Matrixcd<2, 2>{J, J};
    // K = [[c d]; [c d]] -> K^* = [[\bar{c} \bar{c}] [\bar{d} \bar{d}]]
    constexpr auto KH = conjugate_transpose(K);
    STATIC_REQUIRE(KH(0, 0) == c_conj);
    STATIC_REQUIRE(KH(0, 1) == c_conj);
    STATIC_REQUIRE(KH(1, 0) == d_conj);
    STATIC_REQUIRE(KH(1, 1) == d_conj);

    constexpr auto L = K * KH;
    STATIC_REQUIRE(L(0, 0) == c * c_conj + d * d_conj);
    STATIC_REQUIRE(L(0, 1) == c * c_conj + d * d_conj);
    STATIC_REQUIRE(L(1, 0) == c * c_conj + d * d_conj);
    STATIC_REQUIRE(L(1, 1) == c * c_conj + d * d_conj);
    STATIC_REQUIRE(Fro<Matrix<std::complex<double>, 2, 2>>()(K) ==
                   constexpr_sqrt_safe(trace(L).real()));

    constexpr auto I = make_identity<double, 3>();
    constexpr auto I_diag = get_diag(I);
    STATIC_REQUIRE(I_diag == Vectord<3>{1.0, 1.0, 1.0});
    constexpr auto off_diag = I - make_diag(I_diag);
    constexpr auto Zeros = make_zeros<double, 3, 3>();
    STATIC_REQUIRE(all_equal(off_diag, Zeros));

    STATIC_REQUIRE(I * I == I);

    STATIC_REQUIRE(Fro<Matrix<double, 3, 3>>()(I) ==
                   constexpr_sqrt_safe<double>(3));
  }
}

TEST_CASE("Householder vectors", "[householder]") {
  SECTION("Real householder vector") {
    constexpr Vector<double, 4> x{1.0, 2.0, 3.0, 4.0};
    constexpr auto h = HouseholderVector(x);
    constexpr auto beta = h.beta;
    constexpr auto v = h.v;

    constexpr auto I = make_identity<double, 4>();
    constexpr auto P =
        I - beta * outer_product(as_col_vector(v), as_row_vector(v));
    STATIC_REQUIRE(all_equal(transpose(P) * P, I));

    constexpr auto Px = P * x;
    constexpr auto expected = Vectord<4>{-5.47723, 0.0, 0.0, 0.0};
    STATIC_REQUIRE(euclidean_distance(Px, expected) <= eps);
  }

  SECTION("Complex householder vector (all real)") {
    constexpr std::complex<double> a(1.0, 0.0);
    constexpr std::complex<double> b(2.0, 0.0);
    constexpr std::complex<double> c(3.0, 0.0);
    constexpr std::complex<double> d(4.0, 0.0);
    constexpr Vector<std::complex<double>, 4> x{a, b, c, d};

    constexpr auto h = HouseholderVector(x);
    constexpr auto beta = h.beta;
    constexpr auto v = h.v;

    constexpr auto I = make_identity<std::complex<double>, 4>();
    constexpr auto P =
        I - beta * outer_product(as_col_vector(v), as_row_vector(conjugate(v)));
    STATIC_REQUIRE(all_equal(conjugate_transpose(P) * P, I));

    // (dakre) complex case reflects in opp dir as reals maybe monitor more as
    // time goes on
    constexpr auto Px = P * x;
    constexpr std::complex<double> e(-5.47723, 0.0);
    constexpr std::complex<double> f(0.0, 0.0);
    constexpr auto expected = Vectorcd<4>{e, f, f, f};
    STATIC_REQUIRE(euclidean_distance(Px, expected) <= eps);
  }

  SECTION("Complex householder vector") {
    constexpr std::complex<double> a(1.0, 2.0);
    constexpr std::complex<double> b(3.0, 4.0);
    constexpr Vector<std::complex<double>, 2> x{a, b};

    constexpr auto h = HouseholderVector(x);
    constexpr auto beta = h.beta;
    constexpr auto v = h.v;

    constexpr auto I = make_identity<std::complex<double>, 2>();
    constexpr auto P =
        I - beta * outer_product(as_col_vector(v), as_row_vector(conjugate(v)));
    STATIC_REQUIRE(all_equal(conjugate_transpose(P) * P, I));
  }
}

TEST_CASE("Hessenberg matrices", "[hessenberg]") {
  constexpr auto a = Vectord<3>{1.0, 2.0, 3.0};
  constexpr auto b = Vectord<3>{4.0, 5.0, 6.0};
  constexpr auto c = Vectord<3>{7.0, 8.0, 9.0};
  constexpr auto A = Matrixd<3, 3>{a, b, c};
  constexpr auto H_mat = Hessenberg(A);
  constexpr auto H = H_mat.H();
  constexpr auto beta = H_mat.beta;
  STATIC_REQUIRE(constexpr_abs(beta(0)) > 0.0);
  STATIC_REQUIRE(constexpr_abs(beta(1)) == 0.0);
  STATIC_REQUIRE(constexpr_abs(beta(2)) == 0.0);
  STATIC_REQUIRE(H(2, 0) == 0.0);
  STATIC_REQUIRE(H(0, 0) == H_mat(0, 0));
  STATIC_REQUIRE(H(0, 1) == H_mat(0, 1));
  STATIC_REQUIRE(H(0, 2) == H_mat(0, 2));
  STATIC_REQUIRE(H(1, 0) == H_mat(1, 0));
  STATIC_REQUIRE(H(1, 1) == H_mat(1, 1));
  STATIC_REQUIRE(H(1, 2) == H_mat(1, 2));
  STATIC_REQUIRE(H(0, 2) == H_mat(0, 2));
  STATIC_REQUIRE(H(1, 2) == H_mat(1, 2));
  STATIC_REQUIRE(H(2, 2) == H_mat(2, 2));

  constexpr auto U = H_mat.U();
  constexpr auto UTU = transpose(U) * U;
  STATIC_REQUIRE(all_equal(UTU, make_identity<double, 3>()));

  STATIC_REQUIRE(all_equal(H, transpose(U) * A * U));
}

TEST_CASE("S^n via QR factorization", "[symmetric_matrices]") {
  SECTION("Positive Definite Matrices") {
    constexpr auto A =
        Matrixd<2, 2>{Vectord<2>{2.0, 1.0}, Vectord<2>{1.0, 2.0}};
    constexpr auto A_smat = SymmetricMatrix(A);
    STATIC_REQUIRE(A_smat.type == SymmetricOperatorType::PD);
    STATIC_REQUIRE(A(0, 0) == A_smat(0, 0));
    STATIC_REQUIRE(A(0, 1) == A_smat(0, 1));
    STATIC_REQUIRE(A(1, 0) == A_smat(1, 0));
    STATIC_REQUIRE(A(1, 1) == A_smat(1, 1));

    constexpr auto B =
        Matrixd<2, 2>{Vectord<2>{4.0, -1.0}, Vectord<2>{-1.0, 3.0}};
    constexpr auto B_smat = SymmetricMatrix(B);
    STATIC_REQUIRE(B_smat.type == SymmetricOperatorType::PD);

    constexpr auto I = make_identity<double, 3>();
    constexpr auto I_smat = SymmetricMatrix(I);
    STATIC_REQUIRE(I_smat.type == SymmetricOperatorType::PD);

    constexpr auto I2 = linalg::make_smat<3>(I_smat.svec);
    STATIC_REQUIRE(linalg::all_equal(I, I2));

    constexpr Matrix<double, 2, 2> C{Vectord<2>{1.0, 1.0},
                                     Vectord<2>{1.0, 1.0}};
    constexpr auto C_smat = SymmetricMatrix(C);
    STATIC_REQUIRE(C_smat.type == SymmetricOperatorType::PSD);
  }

  SECTION("Positive semidefinite Matrices") {
    constexpr auto Z_mat = mat_traits<Matrixd<2, 2>>();
    constexpr auto Z = Z_mat(1.0, 0.0, 0.0, 0.0);
    constexpr auto Z_smat = SymmetricMatrix(Z);
    STATIC_REQUIRE(Z_smat.type != SymmetricOperatorType::PD);
    STATIC_REQUIRE(Z_smat.type == SymmetricOperatorType::PSD);
  }

  SECTION("Negative semidefinite Matrices") {
    constexpr auto Z_mat = mat_traits<Matrixd<2, 2>>();
    constexpr auto Z = Z_mat(-1.0, 0.0, 0.0, 0.0);
    constexpr auto Z_smat = SymmetricMatrix(Z);
    STATIC_REQUIRE(Z_smat.type == SymmetricOperatorType::NSD);
  }

  SECTION("Negative definite Matrices") {
    constexpr auto I = -1.0 * make_identity<double, 3>();
    constexpr auto I_smat = SymmetricMatrix(I);
    STATIC_REQUIRE(I_smat.type == SymmetricOperatorType::ND);
  }

  SECTION("Indefinite matrix") {
    constexpr auto I_mat = mat_traits<Matrixd<3, 3>>();
    constexpr auto I = I_mat(-1.0, 2.0, 0.0, 2.0, -1.0, 0.0, 0.0, 0.0, -1.0);
    constexpr auto I_smat = SymmetricMatrix(I);
    STATIC_REQUIRE(I_smat.type == SymmetricOperatorType::IND);
  }

  SECTION("Invalid matrix") {
    constexpr auto A =
        Matrix<double, 2, 2>{Vectord<2>{1.0, 2.0}, Vectord<2>{3.0, 4.0}};
    constexpr auto A_smat = SymmetricMatrix(A);
    STATIC_REQUIRE(A_smat.type == SymmetricOperatorType::INV);
  }
}

TEST_CASE("Matrix square root", "[matrix_sqrt]") {
  SECTION("2x2 diagonal") {
    constexpr auto A =
        Matrixd<2, 2>{Vectord<2>{4.0, 0.0}, Vectord<2>{0.0, 9.0}};
    constexpr auto A_sqrt_res = MatrixSqrt<decltype(A)>()(A);
    constexpr auto A_sqrt = std::get<0>(A_sqrt_res);
    constexpr auto A_inv_sqrt = std::get<1>(A_sqrt_res);
    constexpr auto A_sqrt_exp =
        Matrixd<2, 2>{Vectord<2>{2.0, 0.0}, Vectord<2>{0.0, 3.0}};
    constexpr auto A_inv_sqrt_exp =
        Matrixd<2, 2>{Vectord<2>{0.5, 0.0}, Vectord<2>{0.0, 1.0 / 3.0}};

    STATIC_REQUIRE(all_equal(A_sqrt, A_sqrt_exp, 1e-5));
    STATIC_REQUIRE(all_equal(A_inv_sqrt, A_inv_sqrt_exp, 1e-5));
    STATIC_REQUIRE(all_equal(A_sqrt * A_sqrt, A, 1e-5));
    STATIC_REQUIRE(
        all_equal(A_inv_sqrt * A_sqrt, make_identity<double, 2>(), 1e-5));
  }

  SECTION("3x3 diagonal") {
    constexpr auto A_mat = mat_traits<Matrixd<3, 3>>();
    constexpr auto A = A_mat(1.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 16.0);
    constexpr auto A_sqrt_res = MatrixSqrt<decltype(A)>()(A);
    constexpr auto A_sqrt = std::get<0>(A_sqrt_res);
    constexpr auto A_inv_sqrt = std::get<1>(A_sqrt_res);
    constexpr auto A_sqrt_exp =
        A_mat(1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 4.0);
    constexpr auto A_inv_sqrt_exp =
        A_mat(1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.25);

    STATIC_REQUIRE(all_equal(A_sqrt, A_sqrt_exp, 1e-5));
    STATIC_REQUIRE(all_equal(A_inv_sqrt, A_inv_sqrt_exp, 1e-5));
    STATIC_REQUIRE(all_equal(A_sqrt * A_sqrt, A, 1e-5));
    STATIC_REQUIRE(
        all_equal(A_inv_sqrt * A_sqrt, make_identity<double, 3>(), 1e-5));
  }

  SECTION("2x2 SPD") {
    constexpr auto A_mat = mat_traits<Matrixd<2, 2>>();
    constexpr auto A = A_mat(4.0, 2.0, 2.0, 3.0);
    constexpr auto A_sqrt_res = MatrixSqrt<decltype(A)>()(A);
    constexpr auto A_sqrt = std::get<0>(A_sqrt_res);
    constexpr auto A_inv_sqrt = std::get<1>(A_sqrt_res);

    STATIC_REQUIRE(all_equal(A_sqrt * A_sqrt, A, 1e-5));
    STATIC_REQUIRE(all_equal(A_sqrt, transpose(A_sqrt), 1e-5));
    STATIC_REQUIRE(
        all_equal(A_inv_sqrt * A_sqrt, make_identity<double, 2>(), 1e-5));
  }

  SECTION("3x3 SPD") {
    constexpr auto A_mat = mat_traits<Matrixd<3, 3>>();
    constexpr auto A = A_mat(4.0, 2.0, 0.0, 2.0, 3.0, 0.0, 0.0, 0.0, 9.0);
    constexpr auto A_sqrt_res = MatrixSqrt<decltype(A)>()(A);
    constexpr auto A_sqrt = std::get<0>(A_sqrt_res);
    constexpr auto A_inv_sqrt = std::get<1>(A_sqrt_res);

    STATIC_REQUIRE(all_equal(A_sqrt * A_sqrt, A, 1e-5));
    STATIC_REQUIRE(all_equal(A_sqrt, transpose(A_sqrt), 1e-5));
    STATIC_REQUIRE(
        all_equal(A_inv_sqrt * A_sqrt, make_identity<double, 3>(), 1e-5));
  }

  SECTION("4x4 diagonal") {
    constexpr auto A_mat = mat_traits<Matrixd<4, 4>>();
    constexpr auto A = A_mat(1.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0,
                             9.0, 0.0, 0.0, 0.0, 0.0, 16.0);
    constexpr auto A_sqrt_res = MatrixSqrt<decltype(A)>()(A);
    constexpr auto A_sqrt = std::get<0>(A_sqrt_res);
    constexpr auto A_inv_sqrt = std::get<1>(A_sqrt_res);
    constexpr auto A_sqrt_exp = A_mat(1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0,
                                      0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 4.0);
    constexpr auto A_inv_sqrt_exp =
        A_mat(1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0 / 3.0, 0.0,
              0.0, 0.0, 0.0, 0.25);

    STATIC_REQUIRE(all_equal(A_sqrt, A_sqrt_exp, 1e-5));
    STATIC_REQUIRE(all_equal(A_inv_sqrt, A_inv_sqrt_exp, 1e-5));
    STATIC_REQUIRE(all_equal(A_sqrt * A_sqrt, A, 1e-5));
    STATIC_REQUIRE(
        all_equal(A_inv_sqrt * A_sqrt, make_identity<double, 4>(), 1e-5));
  }

  SECTION("4x4 SPD") {
    constexpr auto A_mat = mat_traits<Matrixd<4, 4>>();
    constexpr auto A = A_mat(6.0, 2.0, 1.0, 0.0, 2.0, 5.0, 2.0, 1.0, 1.0, 2.0,
                             5.0, 2.0, 0.0, 1.0, 2.0, 6.0);
    constexpr auto A_sqrt_res = MatrixSqrt<decltype(A)>()(A);
    constexpr auto A_sqrt = std::get<0>(A_sqrt_res);
    constexpr auto A_inv_sqrt = std::get<1>(A_sqrt_res);

    STATIC_REQUIRE(all_equal(A_sqrt * A_sqrt, A, 1e-5));
    STATIC_REQUIRE(all_equal(A_sqrt, transpose(A_sqrt), 1e-5));
    STATIC_REQUIRE(
        all_equal(A_inv_sqrt * A_sqrt, make_identity<double, 4>(), 1e-5));
  }
}

TEST_CASE("svec/smat preserve symmetric geometry", "[symmetric_matrices]") {
  constexpr auto A = Matrixd<3, 3>{
      Vectord<3>{2.0, -1.0, 3.0},
      Vectord<3>{-1.0, 4.0, 0.5},
      Vectord<3>{3.0, 0.5, 5.0},
  };

  constexpr auto svec = get_svec(A);
  constexpr auto A_roundtrip = make_smat<3>(svec);

  STATIC_REQUIRE(all_equal(A_roundtrip, A));
  STATIC_REQUIRE(all_equal(make_svec(A_roundtrip), svec));
  STATIC_REQUIRE(constexpr_abs(inner_product(svec, svec) - trace(A * A)) <
                 1e-10);
}

TEST_CASE("LDL^T Factorization") {
  SECTION("Identity matrix factors exactly") {
    constexpr auto I = make_identity<double, 10>();
    constexpr auto ldl = LDL(I);
    STATIC_REQUIRE(ldl.is_valid());
    STATIC_REQUIRE(ldl.is_psd());
    STATIC_REQUIRE(get_diag(ldl.D()) == vector_traits<Vectord<10>>().fill(1.0));
    STATIC_REQUIRE(ldl.L() == make_identity<double, 10>());
  }

  SECTION("Zero matrix is rejected by LDL") {
    constexpr auto Z = Matrixd<6, 6>{};
    constexpr auto ldl = LDL(Z);
    STATIC_REQUIRE(!ldl.is_valid());
  }

  SECTION("Diagonal indefinite matrix factors and solves") {
    constexpr auto A_mat = mat_traits<Matrixd<4, 4>>();
    constexpr auto A = A_mat(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                             -1.0, 0.0, 0.0, 0.0, 0.0, -1.0);
    constexpr auto ldl = LDL(A);
    STATIC_REQUIRE(ldl.is_valid());
    STATIC_REQUIRE(ldl.is_ind());
    STATIC_REQUIRE(get_diag(ldl.D()) == Vectord<4>{1.0, 1.0, -1.0, -1.0});
    STATIC_REQUIRE(ldl.L() == make_identity<double, 4>());

    constexpr auto b = Vectord<4>{1.0, 1.0, 1.0, 1.0};
    constexpr auto solve_res = ldl.solve(b);
    constexpr auto x = std::get<0>(solve_res);
    constexpr auto event = std::get<1>(solve_res);
    STATIC_REQUIRE(event == linalg::NumericalEvent::NONE);
    constexpr auto y = A * x;
    STATIC_REQUIRE(y == b);
  }

  SECTION("2x2 SPD") {
    constexpr auto A_mat = mat_traits<Matrixd<2, 2>>();
    constexpr auto A = A_mat(4.0, 2.0, 2.0, 3.0);
    constexpr auto ldl = LDL(A);
    STATIC_REQUIRE(ldl.is_valid());
    STATIC_REQUIRE(ldl.is_psd());
    STATIC_REQUIRE(get_diag(ldl.D()) == Vectord<2>{4.0, 2.0});
    STATIC_REQUIRE(A == ldl.L() * (ldl.D() * transpose(ldl.L())));

    constexpr auto b = Vectord<2>{1.0, 2.0};
    constexpr auto solve_res = ldl.solve(b);
    constexpr auto x = std::get<0>(solve_res);
    constexpr auto event = std::get<1>(solve_res);
    STATIC_REQUIRE(event == linalg::NumericalEvent::NONE);
    constexpr auto y = A * x;
    STATIC_REQUIRE(y == b);
  }

  SECTION("4x4 SPD") {
    constexpr auto A_mat = mat_traits<Matrixd<4, 4>>();
    constexpr auto A = A_mat(6.0, 2.0, 1.0, 0.0, 2.0, 5.0, 2.0, 1.0, 1.0, 2.0,
                             5.0, 2.0, 0.0, 1.0, 2.0, 6.0);
    constexpr auto ldl = LDL(A);
    STATIC_REQUIRE(ldl.is_valid());
    STATIC_REQUIRE(ldl.is_psd());
    STATIC_REQUIRE(L2<Vectord<4>>()(get_diag(ldl.D()) -
                                    Vectord<4>{6.0, 4.3333333333, 4.1923076923,
                                               5.146789}) < eps);
    STATIC_REQUIRE(A == ldl.L() * (ldl.D() * transpose(ldl.L())));

    constexpr auto b = Vectord<4>{1.0, 2.0, 3.0, 4.0};
    constexpr auto solve_res = ldl.solve(b);
    constexpr auto x = std::get<0>(solve_res);
    constexpr auto event = std::get<1>(solve_res);
    STATIC_REQUIRE(event == linalg::NumericalEvent::NONE);
    constexpr auto y = A * x;
    STATIC_REQUIRE(y == b);
  }

  SECTION("Determinant from LDL diagonal") {
    constexpr auto A_mat = mat_traits<Matrixd<4, 4>>();
    constexpr auto A = A_mat(1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0,
                             -3.0, 0.0, 0.0, 0.0, 0.0, 4.0);
    constexpr auto ldl = LDL(A);

    STATIC_REQUIRE(ldl.is_valid());
    STATIC_REQUIRE(get_diag(ldl.D()) == Vectord<4>{1.0, 2.0, -3.0, 4.0});
    STATIC_REQUIRE(ldl.det() == -24.0);
  }

  SECTION("Banded SPD") {
    constexpr auto A_mat = mat_traits<Matrixd<5, 5>>();
    constexpr auto A = A_mat(2.0, -1.0, 0.0, 0.0, 0.0, -1.0, 2.0, -1.0, 0.0,
                             0.0, 0.0, -1.0, 2.0, -1.0, 0.0, 0.0, 0.0, -1.0,
                             2.0, -1.0, 0.0, 0.0, 0.0, -1.0, 2.0);
    constexpr auto ldl = LDL(A);
    STATIC_REQUIRE(ldl.is_valid());
    STATIC_REQUIRE(ldl.is_psd());
    STATIC_REQUIRE(A == ldl.L() * (ldl.D() * transpose(ldl.L())));

    constexpr auto b = Vectord<5>{1.0, 2.0, 3.0, 4.0, 5.0};
    constexpr auto solve_res = ldl.solve(b);
    constexpr auto x = std::get<0>(solve_res);
    constexpr auto event = std::get<1>(solve_res);
    STATIC_REQUIRE(event == linalg::NumericalEvent::NONE);
    constexpr auto y = A * x;
    STATIC_REQUIRE(L2<Vectord<5>>()(y - b) < eps);
  }

  SECTION(
      "Coupled indefinite matrix reconstructs and solves within tolerance") {
    constexpr auto A_mat = mat_traits<Matrixd<4, 4>>();
    constexpr auto A = A_mat(2.0, 0.4, -0.2, 0.1, 0.4, -0.92, -0.38, 0.21, -0.2,
                             -0.38, 2.89, 0.91, 0.1, 0.21, 0.91, -1.8625);
    constexpr auto ldl = LDL(A);
    STATIC_REQUIRE(ldl.is_valid());
    STATIC_REQUIRE(ldl.is_ind());
    constexpr auto res = ldl.L() * (ldl.D() * transpose(ldl.L()));
    /// (dakre) we can see no longer == but equals within tolerance
    STATIC_REQUIRE(all_equal(A, res));

    constexpr auto b = Vectord<4>{-1.0, -2.0, -3.0, -4.0};
    constexpr auto solve_res = ldl.solve(b);
    constexpr auto x = std::get<0>(solve_res);
    constexpr auto event = std::get<1>(solve_res);
    STATIC_REQUIRE(event == linalg::NumericalEvent::NONE);
    constexpr auto y = A * x;
    STATIC_REQUIRE(L2<Vectord<4>>()(y - b) < eps);
  }

  SECTION("Tiny leading pivot is rejected") {
    constexpr auto A_mat = mat_traits<Matrixd<3, 3>>();
    constexpr auto A = A_mat(1e-12, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, -1.0);
    constexpr auto ldl = LDL(A);
    STATIC_REQUIRE(!ldl.is_valid());
  }

  SECTION("Near-singular matrix is rejected") {
    constexpr auto A_mat = mat_traits<Matrixd<3, 3>>();
    constexpr auto A =
        A_mat(1.0, 0.999999, 0.0, 0.999999, 0.999998, 0.0, 0.0, 0.0, 1.0);
    constexpr auto ldl = LDL(A);
    STATIC_REQUIRE(!ldl.is_valid());
  }
}

TEST_CASE("Matrix Pencil") {
  SECTION("Symmetric basis expansion") {
    using Mat = Matrixd<2, 2>;
    constexpr linalg::MatrixPencil<Mat, 3> pencil;
    constexpr auto F_blks = pencil.template apply_sym_mat<Mat>(
        [](const Mat& E) constexpr { return E; });

    constexpr auto inv_s2 = inv_sqrt2_v<double>;
    constexpr auto E11 = Mat{Vectord<2>{1.0, 0.0}, Vectord<2>{0.0, 0.0}};
    constexpr auto E12 = Mat{Vectord<2>{0.0, inv_s2}, Vectord<2>{inv_s2, 0.0}};
    constexpr auto E22 = Mat{Vectord<2>{0.0, 0.0}, Vectord<2>{0.0, 1.0}};

    STATIC_REQUIRE(F_blks(0) == E11);
    STATIC_REQUIRE(F_blks(1) == E12);
    STATIC_REQUIRE(F_blks(2) == E22);
  }

  SECTION("Symmetric basis expansion follows svec order for 3x3") {
    using Mat = Matrixd<3, 3>;
    constexpr linalg::MatrixPencil<Mat, 6> pencil;
    constexpr auto F_blks = pencil.template apply_sym_mat<Mat>(
        [](const Mat& E) constexpr { return E; });

    constexpr auto inv_s2 = inv_sqrt2_v<double>;
    constexpr auto E11 =
        Mat{Vectord<3>{1.0, 0.0, 0.0}, Vectord<3>{0.0, 0.0, 0.0},
            Vectord<3>{0.0, 0.0, 0.0}};
    constexpr auto E21 =
        Mat{Vectord<3>{0.0, inv_s2, 0.0}, Vectord<3>{inv_s2, 0.0, 0.0},
            Vectord<3>{0.0, 0.0, 0.0}};
    constexpr auto E22 =
        Mat{Vectord<3>{0.0, 0.0, 0.0}, Vectord<3>{0.0, 1.0, 0.0},
            Vectord<3>{0.0, 0.0, 0.0}};
    constexpr auto E31 =
        Mat{Vectord<3>{0.0, 0.0, inv_s2}, Vectord<3>{0.0, 0.0, 0.0},
            Vectord<3>{inv_s2, 0.0, 0.0}};
    constexpr auto E32 =
        Mat{Vectord<3>{0.0, 0.0, 0.0}, Vectord<3>{0.0, 0.0, inv_s2},
            Vectord<3>{0.0, inv_s2, 0.0}};
    constexpr auto E33 =
        Mat{Vectord<3>{0.0, 0.0, 0.0}, Vectord<3>{0.0, 0.0, 0.0},
            Vectord<3>{0.0, 0.0, 1.0}};

    STATIC_REQUIRE(F_blks(0) == E11);
    STATIC_REQUIRE(F_blks(1) == E21);
    STATIC_REQUIRE(F_blks(2) == E22);
    STATIC_REQUIRE(F_blks(3) == E31);
    STATIC_REQUIRE(F_blks(4) == E32);
    STATIC_REQUIRE(F_blks(5) == E33);
  }

  SECTION("Full basis expansion") {
    using Block = Matrixd<2, 2>;
    using FullMat = Matrixd<2, 2>;
    constexpr linalg::MatrixPencil<Block, 4> pencil;
    constexpr auto F_blks = pencil.template apply_full<FullMat>(
        [](const FullMat& E) constexpr { return E + 2.0 * transpose(E); });

    constexpr auto F00 = Block{Vectord<2>{3.0, 0.0}, Vectord<2>{0.0, 0.0}};
    constexpr auto F01 = Block{Vectord<2>{0.0, 1.0}, Vectord<2>{2.0, 0.0}};
    constexpr auto F10 = Block{Vectord<2>{0.0, 2.0}, Vectord<2>{1.0, 0.0}};
    constexpr auto F11 = Block{Vectord<2>{0.0, 0.0}, Vectord<2>{0.0, 3.0}};

    STATIC_REQUIRE(F_blks(0) == F00);
    STATIC_REQUIRE(F_blks(1) == F01);
    STATIC_REQUIRE(F_blks(2) == F10);
    STATIC_REQUIRE(F_blks(3) == F11);
  }
}

TEST_CASE("Cholesky Factorization") {
  SECTION("Identity matrix factors exactly") {
    constexpr auto I = make_identity<double, 5>();
    constexpr auto chol = Cholesky(I);
    STATIC_REQUIRE(chol.is_valid());
    STATIC_REQUIRE(chol.is_psd());
    STATIC_REQUIRE(all_equal(chol.G(), make_identity<double, 5>()));
    STATIC_REQUIRE(all_equal(I, chol.G() * transpose(chol.G())));
  }

  SECTION("2x2 SPD") {
    constexpr auto A_mat = mat_traits<Matrixd<2, 2>>();
    constexpr auto A = A_mat(4.0, 2.0, 2.0, 3.0);
    constexpr auto chol = Cholesky(A);
    constexpr auto G_exp = A_mat(2.0, 0.0, 1.0, constexpr_sqrt_safe(2.0));

    STATIC_REQUIRE(chol.is_valid());
    STATIC_REQUIRE(chol.is_psd());
    STATIC_REQUIRE(all_equal(chol.G(), G_exp));
    STATIC_REQUIRE(all_equal(A, chol.G() * transpose(chol.G())));
  }

  SECTION("4x4 SPD") {
    constexpr auto A_mat = mat_traits<Matrixd<4, 4>>();
    constexpr auto A = A_mat(6.0, 2.0, 1.0, 0.0, 2.0, 5.0, 2.0, 1.0, 1.0, 2.0,
                             5.0, 2.0, 0.0, 1.0, 2.0, 6.0);
    constexpr auto chol = Cholesky(A);
    constexpr auto G_exp =
        A_mat(constexpr_sqrt_safe(6.0), 0.0, 0.0, 0.0, 0.8164965809,
              2.0816659995, 0.0, 0.0, 0.4082482905, 0.8006407690, 2.0475125622,
              0.0, 0.0, 0.4803844614, 0.7889497946, 2.2686535634);

    STATIC_REQUIRE(chol.is_valid());
    STATIC_REQUIRE(chol.is_psd());
    STATIC_REQUIRE(all_equal(chol.G(), G_exp, 1e-6));
    STATIC_REQUIRE(all_equal(A, chol.G() * transpose(chol.G()), 1e-6));
  }

  SECTION("Banded SPD") {
    constexpr auto A_mat = mat_traits<Matrixd<5, 5>>();
    constexpr auto A = A_mat(2.0, -1.0, 0.0, 0.0, 0.0, -1.0, 2.0, -1.0, 0.0,
                             0.0, 0.0, -1.0, 2.0, -1.0, 0.0, 0.0, 0.0, -1.0,
                             2.0, -1.0, 0.0, 0.0, 0.0, -1.0, 2.0);
    constexpr auto chol = Cholesky(A);
    constexpr auto G_exp =
        A_mat(constexpr_sqrt_safe(2.0), 0.0, 0.0, 0.0, 0.0, -0.7071067812,
              1.2247448714, 0.0, 0.0, 0.0, 0.0, -0.8164965809, 1.1547005384,
              0.0, 0.0, 0.0, 0.0, -0.8660254038, 1.1180339887, 0.0, 0.0, 0.0,
              0.0, -0.8944271910, 1.0954451150);

    STATIC_REQUIRE(chol.is_valid());
    STATIC_REQUIRE(chol.is_psd());
    STATIC_REQUIRE(all_equal(chol.G(), G_exp, 1e-6));
    STATIC_REQUIRE(all_equal(A, chol.G() * transpose(chol.G()), 1e-6));
  }
}

TEST_CASE("Householder reflection of a negative axis vector is orthogonal",
          "[householder][regression]") {
  constexpr auto x = Vectord<3>{-2.0, 0.0, 0.0};
  constexpr auto reflector = HouseholderVector(x);
  constexpr auto identity = make_identity<double, 3>();
  constexpr auto P =
      identity - reflector.beta * outer_product(as_col_vector(reflector.v),
                                                as_row_vector(reflector.v));
  STATIC_REQUIRE(reflector.beta == 2.0);
  STATIC_REQUIRE(P * x == Vectord<3>{2.0, 0.0, 0.0});
  STATIC_REQUIRE(transpose(P) * P == identity);
}

TEST_CASE("Householder remains orthogonal with a tiny tail near deflation",
          "[householder][regression]") {
  // Squaring the old positive-leading cancellation term underflowed in float.
  constexpr auto x = Vector<float, 3>{1.0f, 1e-12f, -2e-12f};
  constexpr auto reflector = HouseholderVector(x);
  constexpr auto identity = make_identity<float, 3>();
  constexpr auto P =
      identity - reflector.beta * outer_product(as_col_vector(reflector.v),
                                                as_row_vector(reflector.v));
  STATIC_REQUIRE(all_equal(transpose(P) * P, identity, 1e-6f));
  STATIC_REQUIRE(all_equal(P * x, Vector<float, 3>{-1.0f, 0.0f, 0.0f}, 1e-6f));
}
