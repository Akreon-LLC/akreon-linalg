/*
 * Copyright © 2025-2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <complex>
#include <cstddef>
#include <utility>

#include "matrix.hpp"

namespace linalg {

/** @brief Matrix-shaped representation of a `Dim`-element column vector. */
template <typename T, std::size_t Dim>
using ColVector = Matrix<T, Dim, 1>;

/** @brief Matrix-shaped representation of a `Dim`-element row vector. */
template <typename T, std::size_t Dim>
using RowVector = Matrix<T, 1, Dim>;

/**
 * @brief Customization point for compile-time matrix metadata.
 *
 * A specialization registers a matrix representation by exposing
 * `value_type`, `ROWS`, and `COLS`. The primary template is intentionally
 * undefined: an unregistered type must not accidentally satisfy matrix
 * concepts. Public aliases below normalize cv/ref qualification before looking
 * up a specialization.
 *
 * @tparam Mat Unqualified matrix representation being described.
 */
template <class Mat>
struct mat_traits;

/**
 * @brief Traits for the SDK's owning `Matrix` container.
 *
 * In addition to scalar and shape metadata, this specialization provides
 * constexpr construction helpers used heavily by tests and compile-time model
 * definitions. `operator()` consumes exactly `Rows * Cols` coordinates in
 * flattened row-major order; `fill()` assigns the same value to every element.
 */
template <typename T, std::size_t Rows, std::size_t Cols>
struct mat_traits<Matrix<T, Rows, Cols>> {
  /** @brief Scalar field stored by the matrix. */
  using value_type = T;
  /** @brief Compile-time row count. */
  static constexpr std::size_t ROWS = Rows;
  /** @brief Compile-time column count. */
  static constexpr std::size_t COLS = Cols;

  /**
   * @brief Construct a matrix from `Rows * Cols` row-major coordinates.
   *
   * @param xs Scalar values ordered by row, then column.
   */
  template <typename... Ts>
  static constexpr auto operator()(Ts&&... xs) {
    static_assert(sizeof...(xs) == Rows * Cols,
                  "mat_traits::operator() requires Rows*Cols coordinates");
    Matrix<T, Rows, Cols> matrix;
    std::size_t i = 0;
    auto set_element = [&](const T& x) {
      const auto col = i % Cols;
      const auto row = i / Cols;
      matrix(row, col) = x;
      ++i;
    };
    ((void)set_element(std::forward<Ts>(xs)), ...);
    return matrix;
  }

  /** @brief Construct a matrix whose elements are all equal to `x`. */
  static constexpr auto fill(const T& x) {
    Matrix<T, Rows, Cols> matrix;
    for (std::size_t i = 0; i < Rows; ++i)
      for (std::size_t j = 0; j < Cols; ++j) matrix(i, j) = x;
    return matrix;
  }
};

/** @brief Registered matrix scalar type after cv/ref normalization. */
template <class Mat>
using matrix_scalar_t = typename mat_traits<remove_cvref_t<Mat>>::value_type;

/** @brief Registered compile-time row count after cv/ref normalization. */
template <class Mat>
inline constexpr std::size_t mat_rows_v = mat_traits<remove_cvref_t<Mat>>::ROWS;

/** @brief Registered compile-time column count after cv/ref normalization. */
template <class Mat>
inline constexpr std::size_t mat_cols_v = mat_traits<remove_cvref_t<Mat>>::COLS;

/** @name Common scalar and matrix aliases
 *
 * These aliases shorten frequently used real and complex fixed-size types.
 * `ColVector*` and `RowVector*` remain matrices; `Vector*` names the dedicated
 * vector container.
 * @{
 */
template <std::size_t Dim>
using Vectord = Vector<double, Dim>;

template <std::size_t Dim>
using Vectorf = Vector<float, Dim>;

template <std::size_t Dim>
using Vectorcd = Vector<std::complex<double>, Dim>;

template <std::size_t Dim>
using Vectorcf = Vector<std::complex<float>, Dim>;

template <std::size_t Rows, std::size_t Cols>
using Matrixd = Matrix<double, Rows, Cols>;

template <std::size_t Dim>
using ColVectord = ColVector<double, Dim>;

template <std::size_t Dim>
using RowVectord = RowVector<double, Dim>;

template <std::size_t Rows, std::size_t Cols>
using Matrixf = Matrix<float, Rows, Cols>;

template <std::size_t Dim>
using ColVectorf = ColVector<float, Dim>;

template <std::size_t Dim>
using RowVectorf = RowVector<float, Dim>;

template <std::size_t Rows, std::size_t Cols>
using Matrixcd = Matrix<std::complex<double>, Rows, Cols>;

template <std::size_t Dim>
using ColVectorcd = ColVector<std::complex<double>, Dim>;

template <std::size_t Dim>
using RowVectorcd = RowVector<std::complex<double>, Dim>;

template <std::size_t Rows, std::size_t Cols>
using Matrixcf = Matrix<std::complex<float>, Rows, Cols>;

template <std::size_t Dim>
using ColVectorcf = ColVector<std::complex<float>, Dim>;

template <std::size_t Dim>
using RowVectorcf = RowVector<std::complex<float>, Dim>;
/** @} */

} // namespace linalg
