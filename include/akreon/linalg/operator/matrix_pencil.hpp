/*
 * Copyright © 2025-2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "akreon/linalg/linalg_util.hpp"

namespace linalg {

template <SquareMatrix Mat, std::size_t Dim>
struct MatrixPencil {
  /// @brief extracting alias and types
  using T = matrix_scalar_t<Mat>;
  using Fs = Vector<Mat, Dim>;

  /**
   * @brief apply_full
   *
   * Fs = { F_i }
   *
   * @param op (Fn)
   *
   * @return Fs
   */
  template <MatrixContainer FullMat, typename Fn>
  constexpr auto apply_full(const Fn& op) const {
    static constexpr std::size_t Rows = mat_rows_v<FullMat>;
    static constexpr std::size_t Cols = mat_cols_v<FullMat>;
    Fs F_blks{};
    for (std::size_t i = 0, idx = 0; i < Rows; ++i) {
      for (std::size_t j = 0; j < Cols; ++j, ++idx) {
        FullMat E{};
        E(i, j) = T(1);
        F_blks(idx) = op(E);
      }
    }
    return F_blks;
  }

  /**
   * @brief apply symmetric matrix basis
   *
   * Fs = { F_i }
   *
   * @param op (Fn)
   *
   * @return Fs
   */
  template <SquareMatrix SymMat, typename Fn>
  constexpr auto apply_sym_mat(const Fn& op) const {
    static constexpr std::size_t N = mat_rows_v<SymMat>;

    Fs F_blks{};
    for (std::size_t i = 0, idx = 0; i < N; ++i) {
      for (std::size_t j = 0; j <= i; ++j) {
        SymMat E{};
        if (i == j) E(i, j) = T(1);
        else E(i, j) = E(j, i) = linalg::inv_sqrt2_v<T>;

        F_blks(idx++) = op(E);
      }
    }
    return F_blks;
  }
};

} // namespace linalg
