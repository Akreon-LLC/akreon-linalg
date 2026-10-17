/*
 * Copyright © 2025-2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstring>
#include <iostream>

#include <akreon/linalg/linalg.hpp>

namespace unit_test {

/// @brief eps for testing
constexpr double eps = 1e-8;

/**
 * @brief log mat
 *
 * @param A (mat)
 *
 * @return void
 */
template <linalg::MatrixContainer Mat>
inline constexpr void log_mat(const Mat& A) {
  constexpr std::size_t M = linalg::mat_rows_v<Mat>;
  constexpr std::size_t N = linalg::mat_cols_v<Mat>;
  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) std::cout << A(i, j) << " ";
    std::cout << std::endl;
  }
}

/**
 * @brief log vector
 *
 * @param b (vec)
 *
 * @return void
 */
template <linalg::VectorContainer Vec>
inline constexpr void log_vec(const Vec& b) {
  constexpr std::size_t N = linalg::vector_dim_v<Vec>;
  for (std::size_t i = 0; i < N; ++i) std::cout << b(i) << " ";
  std::cout << std::endl;
}

} // namespace unit_test
