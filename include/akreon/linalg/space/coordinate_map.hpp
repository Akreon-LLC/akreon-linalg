/*
 * Copyright © 2025-2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <concepts>

#include "space_traits.hpp"

namespace linalg {

/**
 * @brief Customization point mapping space elements to finite coordinates.
 *
 * A specialization provides static `to_coordinates(element)` and
 * `from_coordinates(coordinate)` operations consistent with the element and
 * coordinate types registered in `space_traits`. The primary template is
 * intentionally undefined so coordinate semantics are always explicit.
 */
template <class Space>
struct coordinate_map;

/**
 * @brief Identity coordinate map for one-row or one-column matrices.
 *
 * Their registered space element and coordinate representations are both the
 * corresponding `Vector<T, Rows * Cols>` rather than the matrix container.
 */
template <Field T, std::size_t Rows, std::size_t Cols>
  requires(Rows == 1 || Cols == 1)
struct coordinate_map<Matrix<T, Rows, Cols>> {
  static constexpr auto to_coordinates(const Vector<T, Rows * Cols>& element) {
    return element;
  }

  static constexpr auto from_coordinates(
      const Vector<T, Rows * Cols>& coordinate) {
    return coordinate;
  }
};

/**
 * @brief Row-major coordinate map for general rectangular matrices.
 *
 * Entry `(row, col)` maps to coordinate `row * Cols + col`. The inverse uses
 * the same ordering, yielding a canonical flatten/unflatten round trip.
 */
template <Field T, std::size_t Rows, std::size_t Cols>
  requires(Rows > 1 && Cols > 1)
struct coordinate_map<Matrix<T, Rows, Cols>> {
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

/** @brief Coordinate-map specialization selected after cv/ref normalization. */
template <class Space>
using coordinate_map_t = coordinate_map<remove_cvref_t<Space>>;

/**
 * @brief Convert a registered space element to its coordinate representation.
 * @param element Element in the canonical representation of `Space`.
 * @return Coordinate vector registered by `space_traits<Space>`.
 */
template <class Space>
constexpr auto coordinates(const space_element_t<Space>& element) {
  return coordinate_map_t<Space>::to_coordinates(element);
}

/**
 * @brief Reconstruct a registered space element from its coordinates.
 * @param coordinate Coordinate representation registered for `Space`.
 * @return Element in the canonical representation of `Space`.
 */
template <class Space>
constexpr auto from_coordinates(const space_coordinate_t<Space>& coordinate) {
  return coordinate_map_t<Space>::from_coordinates(coordinate);
}

} // namespace linalg
