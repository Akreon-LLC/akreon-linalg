/*
 * Copyright © 2025-2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "akreon/linalg/assert.hpp"
#include "akreon/linalg/linalg_ops.hpp"
#include "norm_traits.hpp"

namespace linalg {

/**
 * @brief compute the frobenius norm of a matrix
 *
 * sqrt(tr(A^T A)) if A in R else sqrt(tr(A^* A)) where A in C
 *
 * @param matrix A Rows x Cols elements of type T (must be square)
 *
 * @return scalar
 */
template <MatrixContainer Mat>
struct Fro {
  using domain_type = remove_cvref_t<Mat>;
  using scalar_type = matrix_scalar_t<domain_type>;
  using codomain_type = real_type_t<scalar_type>;

  constexpr codomain_type operator()(const domain_type& matrix) const {
    codomain_type result = 0;
    for (std::size_t row = 0; row < mat_rows_v<domain_type>; ++row)
      for (std::size_t col = 0; col < mat_cols_v<domain_type>; ++col) {
        if constexpr (Real<scalar_type>)
          result += matrix(row, col) * matrix(row, col);
        else result += (std::conj(matrix(row, col)) * matrix(row, col)).real();
      }
    return constexpr_sqrt_safe(result);
  }
};

template <MatrixContainer Mat>
struct norm_traits<Fro<Mat>> : std::true_type {
  using domain_type = remove_cvref_t<Mat>;
  using scalar_type = matrix_scalar_t<domain_type>;
  using codomain_type = real_type_t<scalar_type>;
  static constexpr std::size_t dim =
      mat_rows_v<domain_type> * mat_cols_v<domain_type>;
};

} // namespace linalg
