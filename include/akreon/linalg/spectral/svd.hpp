/*
 * Copyright © 2025-2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "akreon/linalg/assert.hpp"
#include "eig_decomp.hpp"

namespace linalg {

/**
 * @brief Computes A = U Sigma V^* from the eigendecomposition of A^* A.
 *
 * Right singular vectors come from A^* A; each product A*v_i yields its
 * singular value and left singular vector. Singular triplets are sorted
 * by decreasing singular value.
 *
 * @tparam Mat Input matrix type with at least as many rows as columns.
 */
template <MatrixContainer Mat>
struct SVD {
  // @brief aliasing complex and real types for eigen decomposition
  using T = matrix_scalar_t<Mat>;
  using Real = real_type_t<T>;
  using Complex = std::conditional_t<linalg::Complex<T>, T, std::complex<Real>>;

  // @brief helpers for Rows & Cols
  static constexpr std::size_t Rows = mat_rows_v<Mat>;
  static constexpr std::size_t Cols = mat_cols_v<Mat>;

  /**
   * @brief Compute singular triplets for A.
   *
   * @param[in] A Input matrix with Rows >= Cols.
   */
  constexpr SVD(const Mat& A) : converged(false) { svd(A); }

  /** @brief SVD member variables */
  Matrix<Complex, Cols, Cols> VT; ///< right singular vectors
  Matrix<Complex, Rows, Rows> U;  ///< left singular vectors
  Matrix<Real, Rows, Cols> S;     ///< singular values
  bool converged;                 ///< converged

private:
  /**
   * @brief Recover singular triplets from the Gram matrix.
   *
   * @param[in] A Input matrix used to populate U, S, VT, and converged.
   */
  constexpr void svd(const Mat& A) {
    // Form the Hermitian Gram matrix M = A^* A.
    Matrix<T, Cols, Cols> M;
    if constexpr (linalg::Real<T>) M = transpose(A) * A;
    else M = conjugate_transpose(A) * A;

    // Its orthonormal eigenvectors provide the right singular basis V.
    const auto result = EigenDecomposition(M);
    converged = result.converged;
    if (!converged) return;

    const auto& V = result.V;
    U = make_identity<Complex, Rows>();
    for (std::size_t i = 0; i < Cols; ++i) {
      // Compute sigma_i = ||A*v_i|| with scaling, then u_i = A*v_i / sigma_i.
      const auto column = as_complex_matrix(A) * get_col(V, i);
      Real scale = 0;
      for (const auto& value : column)
        scale = std::max(scale, constexpr_abs(value));
      if (scale == Real(0)) continue;
      auto normalized = column;
      for (std::size_t j = 0; j < Rows; ++j) normalized(j) /= scale;
      S(i, i) = scale * L2<Vector<Complex, Rows>>()(normalized);
      for (std::size_t j = 0; j < Rows; ++j) U(j, i) = column(j) / S(i, i);
    }

    // Store the right factor V^* for reconstruction.
    VT = conjugate_transpose(V);

    // Apply one descending permutation to sigma_i, u_i, and v_i.
    sort_svd_desc();
  }

  /**
   * @brief Reorder singular values and their vectors in descending order.
   */
  constexpr void sort_svd_desc() {
    // Determine the order from the diagonal of Sigma.
    const auto P = argsort_desc(get_diag(S));

    // Permute U columns and V^* rows together with their singular values.
    Matrix<Real, Rows, Cols> S_tmp;
    Matrix<Complex, Rows, Rows> U_tmp;
    Matrix<Complex, Cols, Cols> VT_tmp;
    for (std::size_t i = 0; i < Cols; ++i) {
      const auto idx = P[i];
      S_tmp(i, i) = S(idx, idx);

      for (std::size_t r = 0; r < Rows; ++r) U_tmp(r, i) = U(r, idx);

      for (std::size_t c = 0; c < Cols; ++c) VT_tmp(i, c) = VT(idx, c);
    }

    S = S_tmp;
    U = U_tmp;
    VT = VT_tmp;
  }
};

} // namespace linalg
