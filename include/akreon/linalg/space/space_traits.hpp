/*
 * Copyright © 2025-2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "akreon/linalg/matrix/matrix_traits.hpp"

namespace linalg {

/**
 * @brief Customization point for finite-dimensional space metadata.
 *
 * Modern space descriptors expose `scalar_type`, `element_type`,
 * `coordinate_type`, and `DIM`. Legacy vector-space consumers additionally use
 * `value_type` and `Point`; those aliases are supplied by the direct matrix
 * registrations below. The primary template is intentionally undefined so a
 * type acquires space semantics only through explicit registration.
 */
template <class Space>
struct space_traits;

/** @brief Space registration for a scalar-shaped `1 x 1` matrix. */
template <typename T>
struct space_traits<Matrix<T, 1, 1>> {
  /** @brief Legacy scalar-field alias. */
  using value_type = T;
  /** @brief Legacy point representation. */
  using Point = Vector<T, 1>;
  /** @brief Scalar field. */
  using scalar_type = T;
  /** @brief Canonical element representation. */
  using element_type = Point;
  /** @brief Canonical coordinate representation. */
  using coordinate_type = Point;
  /** @brief Finite coordinate dimension. */
  static constexpr std::size_t DIM = 1;
};

/** @brief Space registration for a matrix-shaped row vector. */
template <typename T, std::size_t Dim>
struct space_traits<RowVector<T, Dim>> {
  using value_type = T;
  using Point = Vector<T, Dim>;
  using scalar_type = T;
  using element_type = Point;
  using coordinate_type = Point;
  static constexpr std::size_t DIM = Dim;
};

/** @brief Space registration for a matrix-shaped column vector. */
template <typename T, std::size_t Dim>
struct space_traits<ColVector<T, Dim>> {
  using value_type = T;
  using Point = Vector<T, Dim>;
  using scalar_type = T;
  using element_type = Point;
  using coordinate_type = Point;
  static constexpr std::size_t DIM = Dim;
};

/**
 * @brief Space registration for a general rectangular matrix container.
 *
 * The mathematical element is the matrix itself and its coordinates are a
 * row-major vector of length `Rows * Cols`. `Point` remains the legacy action
 * domain used by matrix/vector-space consumers and has length `Cols`.
 */
template <typename T, std::size_t Rows, std::size_t Cols>
  requires(Rows > 1 && Cols > 1)
struct space_traits<Matrix<T, Rows, Cols>> {
  using value_type = T;
  using Point = Vector<T, Cols>;
  using scalar_type = T;
  using element_type = Matrix<T, Rows, Cols>;
  using coordinate_type = Vector<T, Rows * Cols>;
  static constexpr std::size_t DIM = Rows * Cols;
};

/** @brief Legacy scalar-field alias after cv/ref normalization. */
template <class Space>
using space_type_t = typename space_traits<remove_cvref_t<Space>>::value_type;

/** @brief Legacy point/domain alias after cv/ref normalization. */
template <class Space>
using space_point_t = typename space_traits<remove_cvref_t<Space>>::Point;

/** @brief Finite coordinate dimension after cv/ref normalization. */
template <class Space>
inline constexpr std::size_t space_dim_v =
    space_traits<remove_cvref_t<Space>>::DIM;

/** @brief Scalar field registered for a space. */
template <class Space>
using space_scalar_t =
    typename space_traits<remove_cvref_t<Space>>::scalar_type;

/** @brief Canonical element representation registered for a space. */
template <class Space>
using space_element_t =
    typename space_traits<remove_cvref_t<Space>>::element_type;

/** @brief Finite coordinate representation registered for a space. */
template <class Space>
using space_coordinate_t =
    typename space_traits<remove_cvref_t<Space>>::coordinate_type;

} // namespace linalg
