/*
 * Copyright © 2025-2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "akreon/linalg/linalg_util.hpp"
#include "akreon/linalg/spectral/eig_decomp.hpp"
#include "symmetric_operator.hpp"

namespace linalg {

/**
 * @brief SymmetricMatrix definition
 *
 * This class depends on square like matrix type and tests if it's symmetric (A
 * = A^T or A = A^H) then stores in reduced flattend vector for look up
 */
template <SquareMatrix Mat>
struct SymmetricMatrix : SymmetricOperator<Mat, SymmetricMatrix<Mat>> {
  /// @brief alias for symmetric matrix
  using SymmetricBase = SymmetricOperator<Mat, SymmetricMatrix<Mat>>;
  using typename SymmetricBase::ComplexT;
  using typename SymmetricBase::ScalarT;
  using typename SymmetricBase::T;

  /// @brief helpers for symmetric matrix
  using SymmetricBase::DIM;
  using SymmetricBase::N;

  /**
   * @brief CTOR
   *
   * Creates symmetric matrix with matrix A type T N x N elements
   * - Checks if A is symmetric and sets data elements to vector with N * (N +
   * 1) / 2 elements
   * - Checks eigenvalue signs and sets type
   */
  constexpr explicit SymmetricMatrix(const Mat& A, ScalarT tol = 1e-8)
      : SymmetricBase(A) {
    factor(A, tol);
  }

private:
  /**
   * @brief QR factorization
   *
   * @param Mat A
   * @param tol
   *
   * @return void
   */
  constexpr void factor(const Mat& A, const ScalarT& tol) {
    // (0) First see if the incoming matrix is symmetric
    bool valid = false;
    if constexpr (Real<T>) valid = all_equal(A, transpose(A));
    else valid = all_equal(A, conjugate_transpose(A));

    // (1) No need to set data for invalid one and set type
    // likewise if it is symmetric we can store it temporarily
    // as indefinite one
    if (!valid) {
      this->type = SymmetricOperatorType::INV;
      return;
    }

    // (2) Next calculate the eigenvalues of the matrix and check signs of the
    // eigenvalues by running count
    const auto eig_decomp = EigenDecomposition(A);
    if (!eig_decomp.converged) {
      this->type = SymmetricOperatorType::INV;
      return;
    }

    // (3) Compute definiteness of matrix
    this->compute_definiteness(eig_decomp.lambda, tol);
  }
};

} // namespace linalg
