/*
 * Copyright © 2025-2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "ldl.hpp"
#include "akreon/linalg/linalg_util.hpp"
#include "akreon/linalg/spectral/qr.hpp"

namespace linalg {

/**
 * @brief Matrix inverse via QR
 *
 * @tparam MatIn Square input matrix type.
 */
template <MatrixContainer MatIn>
struct MatrixInverse {
  using Mat = linalg::remove_cvref_t<MatIn>;
  using T = matrix_scalar_t<Mat>;
  static constexpr std::size_t N = mat_rows_v<Mat>;
  static_assert(N >= 1, "inverse requires matrix dim of at least 1");

  /**
   * @brief Compute the inverse by solving against identity columns.
   *
   * @param[in] A Square input matrix.
   * @return Tuple (Mat inverse, NumericalEvent event); inverse is complete
   * only when event is NONE.
   */
  constexpr auto operator()(const Mat& A) const { return apply(A); }

  /**
   * @brief Compute the inverse by solving against identity columns.
   *
   * @param[in] A Square input matrix.
   * @return Tuple (Mat inverse, NumericalEvent event); inverse is complete
   * only when event is NONE.
   */
  constexpr auto apply(const Mat& A) const {
    // Factor once, then solve A*x_i = e_i for each inverse column.
    const auto qr = QR(A);
    Mat A_inv{};
    for (std::size_t i = 0; i < N; ++i) {
      Vector<T, N> I_vec{};
      I_vec(i) = T(1);
      const auto [a_col, event] = qr.solve(I_vec);
      if (event != linalg::NumericalEvent::NONE)
        return std::tuple(A_inv, event);

      for (std::size_t j = 0; j < N; ++j) A_inv(j, i) = a_col(j);
    }
    return std::tuple(A_inv, linalg::NumericalEvent::NONE);
  }
};

/**
 * @brief Symmetric matrix inverse via LDL^T
 *
 * @tparam MatIn Square symmetric input matrix type.
 */
template <SquareMatrix MatIn>
struct SymmetricMatrixInverse {
  using Mat = linalg::remove_cvref_t<MatIn>;
  using T = matrix_scalar_t<Mat>;
  static constexpr std::size_t N = mat_rows_v<Mat>;
  static_assert(N >= 1, "inverse requires matrix dim of at least 1");

  /**
   * @brief Compute the inverse by solving against identity columns.
   *
   * @param[in] A Square symmetric input matrix.
   * @return Tuple (Mat inverse, NumericalEvent event); inverse is complete
   * only when event is NONE.
   */
  constexpr auto operator()(const Mat& A) const { return apply(A); }

  /**
   * @brief Compute the inverse by solving against identity columns.
   *
   * @param[in] A Square symmetric input matrix.
   * @return Tuple (Mat inverse, NumericalEvent event); inverse is complete
   * only when event is NONE.
   */
  constexpr auto apply(const Mat& A) const {
    Mat A_inv{};
    if constexpr (N == 1) {
      if (constexpr_abs(A(0, 0)) == T(0))
        return std::tuple(A_inv, linalg::NumericalEvent::FACTORIZATION_FAILED);
      A_inv(0, 0) = T(1) / A(0, 0);
      return std::tuple(A_inv, linalg::NumericalEvent::NONE);
    } else {
      // Reuse LDL^T for the identity-column solves A*x_i = e_i.
      const auto ldl = LDL(A);

      if (!ldl.is_valid())
        return std::tuple(A_inv, linalg::NumericalEvent::FACTORIZATION_FAILED);

      for (std::size_t i = 0; i < N; ++i) {
        Vector<T, N> I_vec{};
        I_vec(i) = T(1);
        const auto [a_col, event] = ldl.solve(I_vec);
        if (linalg::is_fatal_event(event)) return std::tuple(A_inv, event);
        for (std::size_t j = 0; j < N; ++j) A_inv(j, i) = a_col(j);
      }
      return std::tuple(A_inv, linalg::NumericalEvent::NONE);
    }
  }
};

} // namespace linalg
