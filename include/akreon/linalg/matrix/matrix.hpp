/*
 * Copyright © 2025-2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "akreon/linalg/assert.hpp"

#include <array>
#include <cstddef>
#include <span>
#include <type_traits>

#include "akreon/linalg/scalar/scalar_concepts.hpp"
#include "akreon/linalg/vector/vector.hpp"

namespace linalg {

/**
 * @brief Owning, fixed-size, row-major matrix container over a scalar field.
 *
 * `Matrix` stores exactly `Rows * Cols` elements inline. Its dimensions are
 * part of the type, so construction, indexing, iteration, and conversion can
 * all be used during constant evaluation. Default construction
 * value-initializes the backing storage; arithmetic field elements therefore
 * begin at zero.
 *
 * This type supplies storage and shape, not a coordinate-space policy or a
 * specialized matrix category. Algebraic operations are defined separately,
 * while types such as symmetric matrices and their coordinate maps layer
 * additional meaning over a matrix representation.
 *
 * Elements use two-dimensional indexing: `matrix(row, col)`. Both indices are
 * zero-based and checked with `AK_ASSERT`. Iteration and `view()` expose the
 * flattened row-major order
 * `[A(0,0), ..., A(0,Cols-1), A(1,0), ...]`.
 *
 * @tparam T Real or complex field element type.
 * @tparam Rows Compile-time row count.
 * @tparam Cols Compile-time column count.
 */
template <Field T, std::size_t Rows, std::size_t Cols>
struct Matrix {
  /** @brief Construct a value-initialized matrix. */
  constexpr Matrix() = default;

  /**
   * @brief Construct by copying one fixed-size vector per matrix row.
   *
   * The number of arguments must equal `Rows`, and every argument must be a
   * `Vector<T, Cols>` after cv/ref normalization. This constructor also
   * supports class template argument deduction through the guide below.
   *
   * @param rows Row vectors in increasing row-index order.
   */
  template <typename... RowVectors>
    requires(sizeof...(RowVectors) == Rows && Rows > 0 &&
             (std::same_as<remove_cvref_t<RowVectors>, Vector<T, Cols>> && ...))
  constexpr explicit Matrix(RowVectors&&... rows) {
    std::size_t row = 0;
    const auto copy_row = [this, &row](const Vector<T, Cols>& values) {
      for (std::size_t col = 0; col < Cols; ++col)
        (*this)(row, col) = values(col);
      ++row;
    };
    (copy_row(rows), ...);
  }

  /**
   * @brief Read an element at a zero-based row and column.
   * @return Const reference to the stored element.
   */
  constexpr const T& operator()(std::size_t row, std::size_t col) const {
    AK_ASSERT(row < Rows && col < Cols &&
              "Matrix const access index out of bounds");
    return elements[row * Cols + col];
  }

  /**
   * @brief Read or modify an element at a zero-based row and column.
   * @return Mutable reference to the stored element.
   */
  constexpr T& operator()(std::size_t row, std::size_t col) {
    AK_ASSERT(row < Rows && col < Cols &&
              "Matrix mutable access index out of bounds");
    return elements[row * Cols + col];
  }

  /**
   * @brief Copy one matrix row into a fixed-size vector.
   *
   * Returning a value deliberately keeps `Matrix` storage encapsulated. Use
   * `view()` when direct access to the contiguous flattened storage is needed.
   */
  constexpr auto row(std::size_t index) const {
    AK_ASSERT(index < Rows && "Matrix row index out of bounds");
    Vector<T, Cols> result;
    for (std::size_t col = 0; col < Cols; ++col)
      result(col) = (*this)(index, col);
    return result;
  }

  /** @brief Iterator to the first element in flattened row-major order. */
  constexpr auto begin() { return elements.begin(); }
  /** @brief Const iterator to the first element in row-major order. */
  constexpr auto begin() const { return elements.begin(); }
  /** @brief Iterator one past the final flattened element. */
  constexpr auto end() { return elements.end(); }
  /** @brief Const iterator one past the final flattened element. */
  constexpr auto end() const { return elements.end(); }
  /** @brief Mutable, fixed-extent view of contiguous row-major storage. */
  constexpr auto view() { return std::span<T, Rows * Cols>{elements}; }
  /** @brief Const, fixed-extent view of contiguous row-major storage. */
  constexpr auto view() const {
    return std::span<const T, Rows * Cols>{elements};
  }
  /** @brief Number of stored scalar elements (`Rows * Cols`). */
  constexpr auto size() const { return Rows * Cols; }

  /** @brief Compare shape-compatible matrices element by element. */
  constexpr bool operator==(const Matrix&) const = default;

private:
  std::array<T, Rows * Cols> elements{};
};

/**
 * @brief Deduce matrix scalar type and shape from one or more row vectors.
 *
 * For example, two `Vector<double, 3>` arguments deduce `Matrix<double, 2, 3>`.
 */
template <Field T, std::size_t Cols, typename... Rows>
Matrix(Vector<T, Cols>, Rows...) -> Matrix<T, 1 + sizeof...(Rows), Cols>;

/** @brief Represent a vector as a `1 x Dim` matrix. */
template <Field T, std::size_t Dim>
constexpr auto as_row_matrix(const Vector<T, Dim>& vector) {
  return Matrix<T, 1, Dim>{vector};
}

/** @brief Represent a vector as a `Dim x 1` matrix. */
template <Field T, std::size_t Dim>
constexpr auto as_col_matrix(const Vector<T, Dim>& vector) {
  Matrix<T, Dim, 1> result;
  for (std::size_t i = 0; i < Dim; ++i) result(i, 0) = vector(i);
  return result;
}

/**
 * @brief Convert a one-row or one-column matrix to a vector.
 *
 * The element order follows the non-unit dimension. General rectangular
 * matrices do not implicitly flatten to vectors.
 */
template <Field T, std::size_t Rows, std::size_t Cols>
  requires(Rows == 1 || Cols == 1)
constexpr auto as_vector(const Matrix<T, Rows, Cols>& matrix) {
  constexpr auto Dim = Rows == 1 ? Cols : Rows;
  Vector<T, Dim> result;
  for (std::size_t i = 0; i < Dim; ++i)
    result(i) = Rows == 1 ? matrix(0, i) : matrix(i, 0);
  return result;
}

} // namespace linalg
