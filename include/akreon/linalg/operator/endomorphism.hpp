/*
 * Copyright © 2025-2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "operator_traits.hpp"

namespace linalg {

/**
 * @brief Endomorphism operator CRTP base
 */
template <SquareMatrix Mat, typename Derived>
struct Endomorphism {
  /// @brief alias helpers for types
  using T = matrix_scalar_t<Mat>;
  using ScalarT = real_type_t<T>;
  using ComplexT = std::complex<ScalarT>;

  /// @brief helper to extract N
  static constexpr std::size_t N = mat_rows_v<Mat>;

  /**
   * @brief get element at (i,j) index
   *
   * @param row index
   * @param col index
   *
   * @return element
   */
  constexpr const T& operator()(std::size_t i, std::size_t j) const {
    return self()(i, j);
  }

  /**
   * @brief get element at (i,j) index
   *
   * @param row index
   * @param col index
   *
   * @return element (mutable)
   */
  constexpr T& operator()(std::size_t i, std::size_t j) { return self()(i, j); }

  /**
   * @brief self
   *
   * @return object
   */
  constexpr Derived& self() { return static_cast<Derived&>(*this); }

  /**
   * @brief self
   *
   * @return const object
   */
  constexpr const Derived& self() const {
    return static_cast<const Derived&>(*this);
  }
};

} // namespace linalg
