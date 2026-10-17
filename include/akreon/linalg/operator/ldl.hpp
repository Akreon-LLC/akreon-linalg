/*
 * Copyright © 2025-2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>

#include "akreon/linalg/assert.hpp"
#include "akreon/linalg/linalg_util.hpp"
#include "symmetric_operator.hpp"

namespace linalg {

/**
 * @brief Factors a real symmetric matrix as A = L D L^T without pivoting.
 *
 * @tparam Mat Square symmetric matrix type.
 */
template <SquareMatrix Mat>
struct LDL : SymmetricOperator<Mat, LDL<Mat>> {
  /// @brief alias for symmetric matrix
  using SymmetricBase = SymmetricOperator<Mat, LDL<Mat>>;
  using typename SymmetricBase::ComplexT;
  using typename SymmetricBase::ScalarT;
  using typename SymmetricBase::T;

  /// @brief helpers for symmetric matrix
  using SymmetricBase::DIM;
  using SymmetricBase::N;

  static_assert(N > 1, "LDL^T requires at least a 2x2 matrix");

  /**
   * @brief Factor a symmetric matrix as A = L D L^T.
   *
   * @param[in] A Square symmetric input matrix.
   */
  constexpr explicit LDL(const Mat& A) : SymmetricBase(A) { factor(A); }

  /**
   * @brief Factor a symmetric matrix with explicit pivot tolerance.
   *
   * @param[in] A Square symmetric input matrix.
   * @param[in] config_in Factorization settings, including pivot tolerance.
   */
  constexpr explicit LDL(const Mat& A, const config::FactorConfig<T>& config_in)
      : SymmetricBase(A), config(config_in) {
    factor(A);
  }

  /**
   * @brief Reconstruct the diagonal factor.
   *
   * @return Matrix<T, N, N> containing D.
   */
  constexpr auto D() const { return make_diag(D_vec); }

  /**
   * @brief Reconstruct the unit lower-triangular factor.
   *
   * @return Matrix<T, N, N> containing L.
   */
  constexpr auto L() const { return make_unit_lower_triangle<Mat>(L_vec); }

  /**
   * @brief Solve A x = b using the stored LDL^T factors.
   *
   * Solve L y = b, D z = y, then L^T x = z.
   *
   * @param[in] b Right-hand side with N entries.
   * @return Tuple (Vector<T, N> x, NumericalEvent event).
   */
  constexpr auto solve(const Vector<T, N>& b) const {
    const auto L_mat = L();
    const auto y = forward_solve(L_mat, b);
    const auto [z, event] = diag_solve(y);
    if (is_fatal_event(event)) return std::tuple(z, event);

    return std::tuple(back_solve(transpose(L_mat), z),
                      linalg::NumericalEvent::NONE);
  }

  /**
   * @brief Compute the determinant from the diagonal pivots.
   *
   * @return Scalar T equal to the product of the entries of D.
   */
  constexpr T det() const {
    T res{1};
    std::for_each(D_vec.begin(), D_vec.end(), [&](const auto& d) { res *= d; });
    return res;
  }

  /**
   * @brief Solve A X = Y one column at a time.
   *
   * @tparam MatIn Matrix type sharing the factorization scalar type.
   * @param[in] Y Right-hand-side matrix with N rows.
   * @return Tuple (solution matrix X, NumericalEvent event); X = A^-1 Y on
   * success.
   */
  template <linalg::MatrixContainer MatIn>
    requires(std::same_as<linalg::matrix_scalar_t<MatIn>, T>)
  constexpr auto apply(const MatIn& Y) const {
    constexpr std::size_t YRows = linalg::mat_rows_v<MatIn>;
    constexpr std::size_t YCols = linalg::mat_cols_v<MatIn>;
    static_assert(YRows == N, "Codom of Y must match expected dom of A");

    linalg::Matrix<T, N, YCols> A_inv_Y{};
    for (std::size_t i = 0; i < YCols; ++i) {
      const auto y_col = get_col(Y, i);
      const auto [res, event] = solve(y_col);
      if (linalg::is_fatal_event(event)) return std::tuple(Y, event);
      for (std::size_t j = 0; j < YRows; ++j) A_inv_Y(j, i) = res(j);
    }
    return std::tuple(A_inv_Y, linalg::NumericalEvent::NONE);
  }

private:
  /**
   * @brief Compute and pack the unpivoted LDL^T factors.
   *
   * @param[in] A Symmetric input matrix copied for in-place column updates.
   */
  constexpr void factor(Mat A) {
    const auto& tol = config.tol;
    auto ldl_t = [&]<std::size_t j = 0>(this auto&& self) -> bool {
      // Advance one column at a time until L and D are complete.
      if constexpr (j >= SymmetricBase::N) return true;
      else {
        // The first pivot has no contributions from earlier columns.
        constexpr std::size_t next = j + 1;
        if constexpr (j == 0) {
          if (constexpr_abs(A(0, 0)) < tol) return false;

          const auto A_col = get_block<next, SymmetricBase::N, 0, next>(A);
          const auto col_up = (T(1) / A(0, 0)) * A_col;

          A = set_block<next, SymmetricBase::N, 0, next>(A, col_up);
        } else {
          // Form v_k = L(j,k)*D(k,k) and d_j = A(j,j) - L(j,:)*v.
          Vector<T, j> v{};
          for (std::size_t k = 0; k < j; ++k) v(k) = A(j, k) * A(k, k);

          const auto A_row = get_block<j, next, 0, j>(A);
          A(j, j) = A(j, j) - (A_row * v)(0);

          if (constexpr_abs(A(j, j)) < tol) return false;

          if constexpr (next < SymmetricBase::N) {
            const auto A_col = get_block<next, SymmetricBase::N, j, next>(A);
            const auto A_blk = get_block<next, SymmetricBase::N, 0, j>(A);

            // Remove prior-column contributions, then divide by the new pivot.
            const auto col_up =
                T(1) / A(j, j) * (A_col - as_col_vector(A_blk * v));

            A = set_block<next, SymmetricBase::N, j, next>(A, col_up);
          }
        }
        return self.template operator()<next>();
      }
    };

    if (!ldl_t()) {
      this->type = SymmetricOperatorType::INV;
      return;
    }

    // Pack unit-lower L separately from diagonal D; D determines the inertia.
    L_vec = get_lower_triangle(A);
    D_vec = get_diag(A);

    this->compute_definiteness(D_vec, config.tol);
  }

  /**
   * @brief Solve the unit lower-triangular system L y = b.
   *
   * @param[in] L_mat Unit lower-triangular factor.
   * @param[in] b Right-hand side.
   * @return Vector<T, N> containing y.
   */
  constexpr auto forward_solve(const Mat& L_mat, const Vector<T, N>& b) const {
    Vector<T, N> y{};
    for (std::size_t i = 0; i < N; ++i) {
      T sum(0);
      for (std::size_t j = 0; j < i; ++j) sum += L_mat(i, j) * y(j);
      y(i) = b(i) - sum;
    }
    return y;
  }

  /**
   * @brief Solve the diagonal system D z = y.
   *
   * @param[in] y Right-hand side from forward substitution.
   * @return Tuple (Vector<T, N> z, NumericalEvent event).
   */
  constexpr auto diag_solve(const Vector<T, N>& y) const {
    Vector<T, N> z{};

    AK_ASSERT(this->is_valid() &&
              "LDL::diag_solve requires a valid factorization (non-singular)");
    if (!this->is_valid())
      return std::tuple(z, linalg::NumericalEvent::FACTORIZATION_FAILED);

    for (std::size_t i = 0; i < N; ++i) z(i) = y(i) / D_vec(i);

    return std::tuple(z, linalg::NumericalEvent::NONE);
  }

  /**
   * @brief Solve the unit upper-triangular system L^T x = z.
   *
   * @param[in] L_mat Transpose of the stored unit lower-triangular factor.
   * @param[in] z Right-hand side from the diagonal solve.
   * @return Vector<T, N> containing x.
   */
  constexpr auto back_solve(const Mat& L_mat, const Vector<T, N>& z) const {
    Vector<T, N> x{};
    for (int i = static_cast<int>(N) - 1; i >= 0; --i) {
      T sum(0);
      for (int j = i + 1; j < static_cast<int>(N); ++j)
        sum += L_mat(i, j) * x(j);
      x(i) = z(i) - sum;
    }
    return x;
  }

  /** @brief member variables */
  config::FactorConfig<T> config; ///< factor config
  Vector<T, DIM - N> L_vec;       ///< lower left triangle of factorization
  Vector<T, N> D_vec;             ///< diag of factorization
};

} // namespace linalg
