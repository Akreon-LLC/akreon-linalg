/*
 * Copyright © 2025-2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "akreon/linalg/assert.hpp"
#include "akreon/linalg/numerical_config.hpp"
#include "akreon/linalg/linalg_util.hpp"
#include "akreon/linalg/vector/householder.hpp"

namespace linalg {
/**
 * @brief Factors A = Q R using Householder reflections.
 *
 * Successive reflections eliminate each column below its diagonal. The upper
 * triangle stores R; reflector tails and coefficients reconstruct the full
 * orthogonal or unitary Q. For wide matrices, factor A^* to obtain an LQ form.
 *
 * @tparam Mat Input matrix type with at least as many rows as columns.
 */
template <MatrixContainer Mat>
struct QR {
  // @brief alias for underlying type
  using T = matrix_scalar_t<Mat>;
  using ScalarT = real_type_t<T>;

  // @brief get rows and cols from matrix type
  static constexpr std::size_t Rows = mat_rows_v<Mat>;
  static constexpr std::size_t Cols = mat_cols_v<Mat>;

  static_assert(Rows >= Cols && Rows > 0,
                "QR Factorization requires [square | overdetermined] system");

  /**
   * @brief Factor A using Householder reflections.
   *
   * @param[in] A Input matrix with Rows >= Cols.
   */
  constexpr QR(const Mat& A) { solve_householder(A); }

  /**
   * @brief Factor A with an explicit solve configuration.
   *
   * @param[in] A Input matrix with Rows >= Cols.
   * @param[in] config_in Factorization settings and triangular pivot floor.
   */
  constexpr QR(const Mat& A, const config::FactorConfig<ScalarT>& config_in)
      : config(config_in) {
    solve_householder(A);
  }

  /**
   * @brief Reconstruct the full orthogonal or unitary factor.
   *
   * @return Matrix<T, Rows, Rows> containing Q in A = Q R.
   */
  constexpr auto Q() const { return get_unitary_matrix(); }

  /**
   * @brief Extract the upper-trapezoidal factor.
   *
   * @return Matrix<T, Rows, Cols> containing R in A = Q R.
   */
  constexpr auto R() const { return get_upper_right_triangular_matrix(H); }

  /**
   * @brief Estimate rank from the diagonal of R.
   *
   * @param[in] tol Absolute threshold for counting nonzero diagonal entries.
   * @return Number of diagonal magnitudes greater than tol (std::size_t).
   */
  constexpr auto rank(ScalarT tol = 1e-6) const {
    const auto LT = R();
    const std::size_t mn = std::min(Rows, Cols);
    std::size_t rank = 0;
    for (std::size_t i = 0; i < mn; ++i)
      if (L2<Vector<T, 1>>()(Vector<T, 1>{LT(i, i)}) > tol) rank += 1;
    return rank;
  }

  /**
   * @brief Solve A x = b in the least-squares sense.
   *
   * Project c = Q^* b, then back-substitute through the leading block of R.
   *
   * @param[in] b Right-hand side with Rows entries.
   * @return Tuple (Vector<T, Cols> x, NumericalEvent event).
   */
  constexpr auto solve(const Vector<T, Rows>& b) const {
    Vector<T, Rows> c;

    const auto U = Q();
    const auto LT = R();

    if constexpr (Complex<T>) c = conjugate_transpose(U) * b;
    else c = transpose(U) * b;

    return back_solve(LT, c);
  }

private:
  /**
   * @brief Reduce A to R and pack its Householder reflectors.
   *
   * @param[in] A Input matrix copied into the packed factor storage.
   */
  constexpr void solve_householder(Mat A) {
    H = A;
    constexpr_for<0, Cols>([&]<std::size_t j>(
                               std::integral_constant<std::size_t, j>) {
      const auto& Q_jm = get_n_rows<j, Rows>(H);
      const auto& Q_jm_j = get_col(Q_jm, j);

      // Construct P_j to eliminate rows j+1,... in column j.
      const auto& q = HouseholderVector(Q_jm_j);
      beta(j) = q.beta;
      const auto& v = as_col_vector(q.v);

      // Apply P_j = I - beta*v*v^* to the trailing block from the left.
      auto Q_jm_jn = get_n_cols<j, Cols>(Q_jm);
      if constexpr (Complex<T>)
        Q_jm_jn = Q_jm_jn - (beta(j) * v) * (conjugate_transpose(v) * Q_jm_jn);
      else Q_jm_jn = Q_jm_jn - (beta(j) * v) * (transpose(v) * Q_jm_jn);

      for (std::size_t k = j, a = 0; k < Rows; ++k, ++a)
        for (std::size_t l = j, b = 0; l < Cols; ++l, ++b)
          H(k, l) = Q_jm_jn(a, b);

      // Store the reflector tail in the eliminated entries; v(0) is implicit.
      for (std::size_t k = j + 1, a = 1; k < Rows; ++k, ++a) H(k, j) = v(a, 0);
    });
  }

  /**
   * @brief Reconstruct Q by applying packed reflectors in reverse order.
   *
   * @return Matrix<T, Rows, Rows> containing the full factor Q.
   */
  constexpr auto get_unitary_matrix() const {
    auto U = make_identity<T, Rows>();
    constexpr_for<0, Cols>([&]<std::size_t jj>(
                               std::integral_constant<std::size_t, jj>) {
      const std::size_t j = Cols - jj - 1;
      Vector<T, Rows - j> v;
      v(0) = T(1);
      for (std::size_t i = j + 1, a = 1; i < Rows; ++i, ++a) v(a) = H(i, j);

      const auto& v_col = as_col_vector(v);

      auto Q_block = get_n_rows<j, Rows>(U);
      if constexpr (Complex<T>)
        Q_block = Q_block -
                  (beta(j) * v_col) * (conjugate_transpose(v_col) * Q_block);
      else Q_block = Q_block - (beta(j) * v_col) * (transpose(v_col) * Q_block);

      for (std::size_t r = j, a = 0; r < Rows; ++r, ++a)
        for (std::size_t c = 0; c < Rows; ++c) U(r, c) = Q_block(a, c);
    });
    return U;
  }

  /**
   * @brief Back-substitute through the leading triangular block of R.
   *
   * @param[in] R Upper-trapezoidal factor.
   * @param[in] rhs Projected right-hand side; uses the first Cols entries.
   * @return Tuple (Vector<T, Cols> x, NumericalEvent event).
   */
  constexpr auto back_solve(const Matrix<T, Rows, Cols>& R,
                            const Vector<T, Rows>& rhs) const {
    Vector<T, Cols> x;
    constexpr auto mn = std::min(Rows, Cols);
    linalg::NumericalEvent event{linalg::NumericalEvent::NONE};
    for (int i = static_cast<int>(mn) - 1; i >= 0; --i) {
      T sum = 0;
      for (std::size_t j = i + 1; j < Cols; ++j) sum += (R(i, j) * x(j));

      AK_ASSERT(R(i, i) != T(0) &&
                "QR back_solve requires non-zero diagonal in R");
      // Floor a small pivot while preserving its sign or complex phase.
      T d = R(i, i);
      ScalarT mag = constexpr_abs(d);
      if (mag <= config.floor_eps) {
        d = (mag > ScalarT(0)) ? d * config.floor_eps / mag
                               : T(config.floor_eps);
        event = linalg::NumericalEvent::NEAR_SINGULAR_TRIANGULAR_SOLVE;
      }
      x(i) = (rhs(i) - sum) / d;
    }
    return std::tuple{x, event};
  }

  /** @brief QR decomposition for matrix input A */
  Mat H; ///< R in the upper triangle; packed reflector tails below it.
  Vector<T, Rows> beta;                 ///< beta scalars
  config::FactorConfig<ScalarT> config; ///< Factor config
};

} // namespace linalg
