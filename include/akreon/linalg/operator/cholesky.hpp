/*
 * Copyright © 2025-2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "akreon/linalg/assert.hpp"
#include "akreon/linalg/linalg_util.hpp"
#include "symmetric_operator.hpp"

namespace linalg {

/**
 * @brief Factors a real positive-definite matrix as A = G G^T.
 *
 * @tparam Mat Real square matrix type.
 */
template <SquareMatrix Mat>
struct Cholesky : SymmetricOperator<Mat, Cholesky<Mat>> {
  /// @brief alias for symmetric operators
  using SymmetricBase = SymmetricOperator<Mat, Cholesky<Mat>>;
  using typename SymmetricBase::ComplexT;
  using typename SymmetricBase::ScalarT;
  using typename SymmetricBase::T;

  /// @brief helpers for symmetric matrix
  using SymmetricBase::DIM;
  using SymmetricBase::N;

  static_assert(N > 1, "GG^T requires as least 2x2 matrix");

  /**
   * @brief Factor A as G G^T using Cholesky column updates.
   *
   * @param[in] A Real symmetric positive-definite input matrix.
   */
  constexpr explicit Cholesky(const Mat& A) : SymmetricBase(A) { factor(A); }

  /**
   * @brief Compute Cholesky factors with explicit settings.
   *
   * @param[in] A Real symmetric positive-definite input matrix.
   * @param[in] config_in Factorization settings and definiteness tolerance.
   */
  constexpr explicit Cholesky(const Mat& A,
                              const config::FactorConfig<T>& config_in)
      : SymmetricBase(A), config(config_in) {
    factor(A);
  }

  /**
   * @brief Reconstruct the lower-triangular Cholesky factor.
   *
   * @return Matrix<T, N, N> containing G such that A = G G^T.
   */
  constexpr auto G() const { return make_lower_triangle<Mat>(G_vec); }

private:
  /**
   * @brief Compute and pack the lower Cholesky factor.
   *
   * @param[in] A Positive-definite input matrix copied for column updates.
   */
  constexpr void factor(Mat A) {
    auto chol = [&]<std::size_t j = 0>(this auto&& self) {
      if constexpr (j >= SymmetricBase::N) {
        return;
      } else {
        constexpr std::size_t next = j + 1;
        // Subtract contributions of the previously computed columns of G.
        if constexpr (j > 0) {
          const auto A_jn_j = get_block<j, SymmetricBase::N, j, next>(A);
          const auto A_jn_0j = get_block<j, SymmetricBase::N, 0, j>(A);
          const auto A_j_0j = get_block<j, next, 0, j>(A);
          const auto lhs = A_jn_j - A_jn_0j * transpose(A_j_0j);
          A = set_block<j, SymmetricBase::N, j, next>(A, lhs);
        }
        const auto A_jn_j = get_block<j, SymmetricBase::N, j, next>(A);
        // Take the positive pivot root, then scale the remaining column.
        const auto a_jj = constexpr_sqrt_safe(A(j, j));
        A = set_block<j, SymmetricBase::N, j, next>(A, T(1) / a_jj * A_jn_j);
        return self.template operator()<next>();
      }
    };

    chol();

    // Pack the completed lower factor, including its diagonal.
    G_vec = get_lower_triangle_with_diag(A);

    this->compute_definiteness(get_diag(A), config.tol);
  }

  /** @brief member vars */
  config::FactorConfig<T> config; ///< factor config
  Vector<T, DIM> G_vec;           ///< lower left triangle of chol
};

} // namespace linalg
