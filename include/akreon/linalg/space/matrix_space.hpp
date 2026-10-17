/*
 * Copyright © 2025-2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "coordinate_map.hpp"

namespace linalg {

/**
 * @brief Space of all `Rows x Cols` matrices over field `T`.
 *
 * This descriptor adds explicit space and coordinate semantics to the plain
 * `Matrix` container. Its coordinate dimension is `Rows * Cols`.
 */
template <Field T, std::size_t Rows, std::size_t Cols>
struct MatrixSpace {};

/** @brief Space metadata for the full rectangular matrix space. */
template <Field T, std::size_t Rows, std::size_t Cols>
struct space_traits<MatrixSpace<T, Rows, Cols>> {
  /** @brief Scalar field. */
  using scalar_type = T;
  /** @brief Canonical matrix element representation. */
  using element_type = Matrix<T, Rows, Cols>;
  /** @brief Flattened row-major coordinate representation. */
  using coordinate_type = Vector<T, Rows * Cols>;
  /** @brief Number of independent scalar coordinates. */
  static constexpr std::size_t dim = Rows * Cols;
  /** @brief Compatibility spelling of the coordinate dimension. */
  static constexpr std::size_t DIM = dim;
};

/**
 * @brief Row-major coordinate isomorphism for `MatrixSpace`.
 *
 * Matrix entry `(row, col)` maps to coordinate `row * Cols + col`.
 */
template <Field T, std::size_t Rows, std::size_t Cols>
struct coordinate_map<MatrixSpace<T, Rows, Cols>> {
  static constexpr auto to_coordinates(const Matrix<T, Rows, Cols>& element) {
    Vector<T, Rows * Cols> coordinate;
    for (std::size_t row = 0; row < Rows; ++row)
      for (std::size_t col = 0; col < Cols; ++col)
        coordinate(row * Cols + col) = element(row, col);
    return coordinate;
  }

  static constexpr auto from_coordinates(
      const Vector<T, Rows * Cols>& coordinate) {
    Matrix<T, Rows, Cols> element;
    for (std::size_t row = 0; row < Rows; ++row)
      for (std::size_t col = 0; col < Cols; ++col)
        element(row, col) = coordinate(row * Cols + col);
    return element;
  }
};

} // namespace linalg
