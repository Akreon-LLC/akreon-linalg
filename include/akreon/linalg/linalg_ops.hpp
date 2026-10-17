/*
 * Copyright © 2025-2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "operator/inner_product.hpp"
#include "operator/operator_traits.hpp"
#include "vector/vector.hpp"

namespace linalg {

/**
 * @brief add two vectors together to get a third
 *
 * A + B = C
 *
 * @param vector A of type T with N elements
 * @param vector B of type T with N elements
 *
 * @return vector
 */
template <VectorContainer Vec>
constexpr auto add(const Vec& A, const Vec& B) {
  // (dakre) below is safeguard against non-square matrix acting as vector with
  // duck type T as Vector<Vector<...>>
  using T = vector_element_t<Vec>;
  constexpr auto N = vector_dim_v<Vec>;
  Vector<T, N> C;
  for (std::size_t i = 0; i < N; ++i) C(i) = A(i) + B(i);
  return C;
}

/**
 * @brief operator overload for add
 *
 * @see add
 */
template <VectorContainer Vec>
constexpr auto operator+(const Vec& A, const Vec& B) {
  return add(A, B);
}

/**
 * @brief Add two shape-compatible matrices element by element.
 * @param A Left matrix.
 * @param B Right matrix of the same registered type and shape.
 * @return Owning `Matrix` containing `A(row,col) + B(row,col)`.
 */
template <MatrixContainer Mat>
constexpr auto add(const Mat& A, const Mat& B) {
  using T = matrix_scalar_t<Mat>;
  constexpr auto Rows = mat_rows_v<Mat>;
  constexpr auto Cols = mat_cols_v<Mat>;
  Matrix<T, Rows, Cols> result;
  for (std::size_t row = 0; row < Rows; ++row)
    for (std::size_t col = 0; col < Cols; ++col)
      result(row, col) = A(row, col) + B(row, col);
  return result;
}

/** @brief Matrix addition operator delegating to `add`. */
template <MatrixContainer Mat>
constexpr auto operator+(const Mat& A, const Mat& B) {
  return add(A, B);
}

/**
 * @brief multiply a vector A with a scalar alpha
 *
 * alpha * A = alpha A
 *
 * @param alpha of scalar type T
 * @param vector x of type T with N elements
 *
 * @return vector
 */
template <VectorContainer Vec, Field Alpha>
constexpr auto scalar_multiplication(const Alpha& alpha, const Vec& A) {
  using T = vector_element_t<Vec>;
  constexpr auto N = vector_dim_v<Vec>;
  Vector<T, N> C;
  for (std::size_t i = 0; i < N; ++i) C(i) = alpha * A(i);
  return C;
}

/**
 * @brief operator overload for scalar multiplication
 *
 * @see scalar_multiplication
 */
template <VectorContainer Vec, Field Alpha>
constexpr auto operator*(const Alpha& alpha, const Vec& A) {
  return scalar_multiplication(alpha, A);
}

/**
 * @brief Multiply every matrix entry by a field scalar.
 * @param alpha Scalar multiplier.
 * @param A Matrix operand.
 * @return Owning matrix with the same shape and registered matrix scalar type.
 */
template <MatrixContainer Mat, Field Alpha>
constexpr auto scalar_multiplication(const Alpha& alpha, const Mat& A) {
  using T = matrix_scalar_t<Mat>;
  constexpr auto Rows = mat_rows_v<Mat>;
  constexpr auto Cols = mat_cols_v<Mat>;
  Matrix<T, Rows, Cols> result;
  for (std::size_t row = 0; row < Rows; ++row)
    for (std::size_t col = 0; col < Cols; ++col)
      result(row, col) = alpha * A(row, col);
  return result;
}

/** @brief Left scalar-matrix multiplication operator. */
template <MatrixContainer Mat, Field Alpha>
constexpr auto operator*(const Alpha& alpha, const Mat& A) {
  return scalar_multiplication(alpha, A);
}

/**
 * @brief operator overload for subtraction
 *
 * @see add
 * @see scalar_multiplication
 */
template <VectorContainer Vec>
constexpr auto operator-(const Vec& A, const Vec& B) {
  return A + vector_scalar_t<Vec>(-1) * B;
}

/** @brief Subtract two matrices as `A + (-1) * B`. */
template <MatrixContainer Mat>
constexpr auto operator-(const Mat& A, const Mat& B) {
  return A + matrix_scalar_t<Mat>(-1) * B;
}

/**
 * @brief tranpose of a vector
 *
 * Vector is naturally row vector in how it's stored so the transpose flips it
 * from a row to column based matrix
 *
 * @param vector of type T with N elements
 *
 * @return (N x 1) matrix
 */
template <FieldVector Vec>
constexpr auto transpose(const Vec& x) {
  using T = vector_element_t<Vec>;
  constexpr auto N = vector_dim_v<Vec>;
  Matrix<T, N, 1> x_t;
  for (std::size_t i = 0; i < N; ++i) x_t(i, 0) = x(i);
  return x_t;
}

/**
 * @brief conjugate a vector
 *
 * @param vector of type T with N elements
 *
 * @return conjugated vector
 */
template <ComplexVector Vec>
constexpr auto conjugate(const Vec& x) {
  using T = vector_element_t<Vec>;
  constexpr auto N = vector_dim_v<Vec>;
  Vector<T, N> result;
  for (std::size_t i = 0; i < N; ++i) result(i) = std::conj(x(i));
  return result;
}

/**
 * @brief cross product between x x y = Z
 *
 * @param vector of type T with 3 elements
 * @param vector of type T with 3 elements
 *
 * @return (3x1) vector
 */
template <FieldVector Vec>
  requires(vector_dim_v<Vec> == 3)
constexpr auto cross(const Vec& x, const Vec& y) {
  return Vec{x(1) * y(2) - x(2) * y(1), -(x(0) * y(2) - x(2) * y(0)),
             x(0) * y(1) - x(1) * y(0)};
}

/**
 * @brief simple function that checks if a vector contains only zeros (hence
 * spans nullspace)
 *
 * @param vector of type T with N elements
 *
 * @return T/F
 */
template <FieldVector Vec>
  requires(vector_dim_v<Vec> > 0)
constexpr bool all_zero(const Vec& x) {
  using T = vector_element_t<Vec>;
  constexpr auto N = vector_dim_v<Vec>;
  int cnt = 0;
  for (std::size_t i = 0; i < N; ++i) cnt += x(i) == T(0);
  return cnt == N;
}

/**
 * @brief splice operator
 *
 * @param vector type T with N elements
 *
 * @return tuple
 */
template <std::size_t idx, FieldVector Vec>
  requires(vector_dim_v<Vec> > 0 && idx < vector_dim_v<Vec>)
constexpr auto splice(const Vec& x) {
  using T = vector_element_t<Vec>;
  constexpr auto N = vector_dim_v<Vec>;
  Vector<T, idx> a;
  std::size_t i = 0;
  for (; i < idx; ++i) a(i) = x(i);
  Vector<T, N - idx> b;
  for (std::size_t j = 0; i < N; ++i, ++j) b(j) = x(i);
  return std::tuple(a, b);
}

/**
 * @brief Multiply matrix-shaped column and row operands as an outer product.
 *
 * A * B = C where C is a new matrix of Rows_of_A x Cols_of_B
 *
 * @param matrix A of type T Rows1 x Cols1 elements
 * @param matrix B of type T Cols1 x Cols2 elements
 *
 * @return Matrix with `mat_rows_v<MatA>` rows and `mat_cols_v<MatB>` columns.
 * The implementation uses `A(i, 0) * B(0, j)`; callers should supply a
 * one-column left operand and one-row right operand.
 */
template <MatrixContainer MatA, MatrixContainer MatB>
  requires(MatMulCompatible<MatA, MatB>)
constexpr auto outer_product(const MatA& A, const MatB& B) {
  using T = matrix_scalar_t<MatA>;
  constexpr auto Rows = mat_rows_v<MatA>;
  constexpr auto Cols = mat_cols_v<MatB>;
  Matrix<T, Rows, Cols> result;
  for (std::size_t i = 0; i < Rows; ++i)
    for (std::size_t j = 0; j < Cols; ++j) result(i, j) = A(i, 0) * B(0, j);
  return result;
}

/**
 * @brief transpose of a matrix
 *
 * A^T (rows and cols flipped from A)
 *
 * @param matrix A of type T Rows x Cols elements
 *
 * @return A^T
 */
template <MatrixContainer Mat>
constexpr auto transpose(const Mat& A) {
  using T = matrix_scalar_t<Mat>;
  constexpr auto Cols = mat_cols_v<Mat>;
  constexpr auto Rows = mat_rows_v<Mat>;
  Matrix<T, Cols, Rows> result;
  for (std::size_t i = 0; i < Rows; ++i)
    for (std::size_t j = 0; j < Cols; ++j) result(j, i) = A(i, j);
  return result;
}

/**
 * @brief trace of a matrix
 *
 * Sum of the diagonal elements of a matrix
 *
 * @param matrix A of type T Rows x Cols elements
 *
 * @return sum
 */
template <SquareMatrix Mat>
  requires(mat_rows_v<Mat> > 0)
constexpr auto trace(const Mat& A) {
  using T = matrix_scalar_t<Mat>;
  constexpr auto Rows = mat_rows_v<Mat>;
  T sum = 0;
  for (std::size_t i = 0; i < Rows; ++i) sum += A(i, i);
  return sum;
}

/**
 * @brief conjugate transpose of a matrix
 *
 * Like A^T, however the field is complex and the elements get conjugated
 *
 * @param matrix A of type T Rows x Cols elements
 *
 * @return A^*
 */
template <MatrixContainer Mat>
constexpr auto conjugate_transpose(const Mat& A)
  requires(Complex<matrix_scalar_t<Mat>>)
{
  constexpr auto Rows = mat_rows_v<Mat>;
  constexpr auto Cols = mat_cols_v<Mat>;
  auto A_t = transpose(A);
  for (std::size_t i = 0; i < Cols; ++i)
    for (std::size_t j = 0; j < Rows; ++j) A_t(i, j) = std::conj(A_t(i, j));
  return A_t;
}

/**
 * @brief Multiply compatible matrices with the standard row-column product.
 *
 * @param matrix A of type T Rows x Cols elements
 * @param matrix B of type T Cols x Rows elements
 *
 * @return `Matrix<T, rows(A), cols(B)>`.
 */
template <MatrixContainer MatA, MatrixContainer MatB>
  requires(MatMulCompatible<MatA, MatB>)
constexpr auto operator*(const MatA& A, const MatB& B) {
  using T = matrix_scalar_t<MatA>;
  constexpr auto Rows = mat_rows_v<MatA>;
  constexpr auto Cols = mat_cols_v<MatB>;
  Matrix<T, Rows, Cols> result;
  constexpr auto Inner = mat_cols_v<MatA>;
  for (std::size_t i = 0; i < Rows; ++i) {
    for (std::size_t j = 0; j < Cols; ++j) {
      T value{};
      for (std::size_t k = 0; k < Inner; ++k) value += A(i, k) * B(k, j);
      result(i, j) = value;
    }
  }
  return result;
}

/**
 * @brief Apply a compatible matrix to a vector.
 *
 * @param matrix A of type T Rows x Cols elements
 * @param vector X of type T Cols elements
 *
 * @return Vector over the same field with dimension `mat_rows_v<Mat>`.
 */
template <MatrixContainer Mat, VectorContainer Vec>
  requires(MatVecCompatible<Mat, Vec>)
constexpr auto operator*(const Mat& A, const Vec& X) {
  using T = matrix_scalar_t<Mat>;
  constexpr auto Rows = mat_rows_v<Mat>;
  Vector<T, Rows> result;
  constexpr auto Cols = mat_cols_v<Mat>;
  for (std::size_t i = 0; i < Rows; ++i) {
    T value{};
    for (std::size_t j = 0; j < Cols; ++j) value += A(i, j) * X(j);
    result(i) = value;
  }
  return result;
}

/**
 * @brief min element
 *
 * @param vector x
 *
 * @return min element
 */
template <VectorContainer Vec>
  requires(vector_dim_v<Vec> > 0)
constexpr auto min(const Vec& x) {
  using T = vector_element_t<Vec>;
  constexpr std::size_t N = vector_dim_v<Vec>;
  T mn = x(0);
  for (std::size_t i = 1; i < N; ++i) mn = x(i) < mn ? x(i) : mn;
  return mn;
}

/**
 * @brief max element
 *
 * @param vector x
 *
 * @return max element
 */
template <VectorContainer Vec>
  requires(vector_dim_v<Vec> > 0)
constexpr auto max(const Vec& x) {
  using T = vector_element_t<Vec>;
  constexpr std::size_t N = vector_dim_v<Vec>;
  T mx = x(0);
  for (std::size_t i = 1; i < N; ++i) mx = x(i) > mx ? x(i) : mx;
  return mx;
}

/**
 * @brief min element
 *
 * @param vector x
 *
 * @return min element
 */
template <VectorContainer Vec>
  requires(vector_dim_v<Vec> > 0)
constexpr std::size_t arg_min(const Vec& x) {
  using T = vector_element_t<Vec>;
  constexpr std::size_t N = vector_dim_v<Vec>;
  T mn = x(0);
  std::size_t idx{0};
  for (std::size_t i = 1; i < N; ++i)
    if (x(i) < mn) mn = x(i), idx = i;
  return idx;
}

/**
 * @brief max element
 *
 * @param vector x
 *
 * @return max element
 */
template <VectorContainer Vec>
  requires(vector_dim_v<Vec> > 0)
constexpr std::size_t arg_max(const Vec& x) {
  using T = vector_element_t<Vec>;
  constexpr std::size_t N = vector_dim_v<Vec>;
  T mx = x(0);
  std::size_t idx{0};
  for (std::size_t i = 1; i < N; ++i)
    if (x(i) > mx) mx = x(i), idx = i;
  return idx;
}

} // namespace linalg
