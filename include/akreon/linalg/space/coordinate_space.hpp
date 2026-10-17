/*
 * Copyright © 2025-2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "coordinate_map.hpp"
#include "akreon/linalg/scalar/scalar_concepts.hpp"

namespace linalg {

/**
 * @brief Canonical `Dim`-dimensional coordinate space over field `T`.
 *
 * Elements and coordinates are both `Vector<T, Dim>`, making this the identity
 * model of a finite-dimensional vector space.
 */
template <Field T, std::size_t Dim>
struct CoordinateSpace {};

/** @brief Space metadata for `CoordinateSpace`. */
template <Field T, std::size_t Dim>
struct space_traits<CoordinateSpace<T, Dim>> {
  /** @brief Scalar field. */
  using scalar_type = T;
  /** @brief Canonical element representation. */
  using element_type = Vector<T, Dim>;
  /** @brief Coordinate representation, identical to the element type. */
  using coordinate_type = Vector<T, Dim>;
  /** @brief Finite coordinate dimension. */
  static constexpr std::size_t dim = Dim;
  /** @brief Compatibility spelling of the finite coordinate dimension. */
  static constexpr std::size_t DIM = dim;
};

/** @brief Identity coordinate map for `CoordinateSpace`. */
template <Field T, std::size_t Dim>
struct coordinate_map<CoordinateSpace<T, Dim>> {
  static constexpr auto to_coordinates(const Vector<T, Dim>& element) {
    return element;
  }

  static constexpr auto from_coordinates(const Vector<T, Dim>& coordinate) {
    return coordinate;
  }
};

} // namespace linalg
