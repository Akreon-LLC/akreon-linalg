/*
 * Copyright © 2025-2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "akreon/linalg/assert.hpp"
#include "akreon/linalg/numerical_config.hpp"
#include "akreon/linalg/constants.hpp"
#include "norms/matrix_norms.hpp"
#include "norms/vector_norms.hpp"
#include "operator/operator_traits.hpp"

namespace linalg {

/**
 * @brief Copy the half-open vector slice `[Start, End)`.
 *
 * @param vector of type T with N elements
 *
 * @return Vector of type `T` with `End - Start` elements.
 */
template <std::size_t Start, std::size_t End, VectorContainer Vec>
  requires(End - Start < vector_dim_v<Vec> && Start < vector_dim_v<Vec>)
constexpr auto get_elements(const Vec& x) {
  using T = vector_element_t<Vec>;
  Vector<T, End - Start> result;
  for (std::size_t i = Start, idx = 0; i < End; ++i, ++idx) result(idx) = x(i);
  return result;
}

/**
 * @brief create an identity matrix of size N type T
 *
 * @return I
 */
template <Field T, std::size_t N>
constexpr Matrix<T, N, N> make_identity() {
  Matrix<T, N, N> I;
  for (std::size_t i = 0; i < N; ++i) I(i, i) = T(1);
  return I;
}

/**
 * @brief create a matrix filled with zeros of size Rows x Cols of type T
 *
 * @return Zeros
 */
template <Field T, std::size_t Rows, std::size_t Cols>
constexpr Matrix<T, Rows, Cols> make_zeros() {
  Matrix<T, Rows, Cols> Z;
  for (std::size_t i = 0; i < Rows; ++i)
    for (std::size_t j = 0; j < Cols; ++j) Z(i, j) = T(0);
  return Z;
}

/**
 * @brief express the vector X as a column vector (N x 1)
 *
 * @param vector X of type T and N elements
 *
 * @return (N x 1) matrix
 */
template <VectorContainer Vec>
constexpr auto as_col_vector(const Vec& X) {
  using T = vector_element_t<Vec>;
  constexpr auto N = vector_dim_v<Vec>;
  ColVector<T, N> A;
  for (std::size_t i = 0; i < N; ++i) A(i, 0) = X(i);
  return A;
}

/**
 * @brief express the vector X as a row vector (1 x N)
 *
 * @param vector X of type T and N elements
 *
 * @return (1 x N) matrix
 */
template <VectorContainer Vec>
constexpr auto as_row_vector(const Vec& X) {
  using T = vector_element_t<Vec>;
  constexpr auto N = vector_dim_v<Vec>;
  RowVector<T, N> A;
  for (std::size_t i = 0; i < N; ++i) A(0, i) = X(i);
  return A;
}

/**
 * @brief express matrix as complex one
 *
 * @param Matrix A type T with Rows x Cols elements
 *
 * @return A in C
 */
template <MatrixContainer Mat>
constexpr auto as_complex_matrix(const Mat& A) {
  using T = matrix_scalar_t<Mat>;
  constexpr auto Rows = mat_rows_v<Mat>;
  constexpr auto Cols = mat_cols_v<Mat>;
  Matrix<std::complex<real_type_t<T>>, Rows, Cols> result;
  if constexpr (Real<T>) {
    for (std::size_t i = 0; i < Rows; ++i)
      for (std::size_t j = 0; j < Cols; ++j)
        result(i, j) = std::complex<T>(A(i, j));
  } else {
    result = A;
  }
  return result;
}

/**
 * @brief express col vector as vector
 *
 * @param Matrix A type T with N x 1 elements
 *
 * @return vector
 */
template <Field T, std::size_t N>
constexpr auto as_vector(const ColVector<T, N>& A) {
  Vector<T, N> res;
  for (std::size_t i = 0; i < N; ++i) res(i) = A(i, 0);
  return res;
}

/**
 * @brief express row vector as vector
 *
 * @param Matrix A type T with 1 x N elements
 *
 * @return vector
 */
template <Field T, std::size_t N>
constexpr auto as_vector(const RowVector<T, N>& A) {
  Vector<T, N> res;
  for (std::size_t i = 0; i < N; ++i) res(i) = A(0, i);
  return res;
}

/**
 * @brief get the column of matrix A based on given index
 *
 * @param matrix A of size Rows x Cols of type T
 * @param index
 *
 * @return vector X
 */
template <MatrixContainer Mat>
constexpr auto get_col(const Mat& A, std::size_t idx) {
  using T = matrix_scalar_t<Mat>;
  constexpr auto Rows = mat_rows_v<Mat>;
  AK_ASSERT(idx < mat_cols_v<Mat> && "get_col index out of bounds");
  Vector<T, Rows> X;
  for (std::size_t i = 0; i < Rows; ++i) X(i) = A(i, idx);
  return X;
}

/**
 * @brief get the row of matrix A based on given index
 *
 * @param matrix A of size Rows x Cols of type T
 * @param index
 *
 * @return vector X
 */
template <MatrixContainer Mat>
constexpr auto get_row(const Mat& A, std::size_t idx) {
  AK_ASSERT(idx < mat_rows_v<Mat> && "get_row index out of bounds");
  return A.row(idx);
}

/**
 * @brief get the diagonal of a matrix A
 *
 * @param matrix A of size Rows x Cols of type T
 *
 * @return diagonal vector
 */
template <MatrixContainer Mat>
constexpr auto get_diag(const Mat& A) {
  using T = matrix_scalar_t<Mat>;
  constexpr auto Rows = mat_rows_v<Mat>;
  constexpr auto Cols = mat_cols_v<Mat>;
  constexpr std::size_t mn = std::min(Rows, Cols);
  Vector<T, mn> diag;
  for (std::size_t i = 0; i < mn; ++i) diag(i) = A(i, i);
  return diag;
}

/**
 * @brief make a diagonal matrix from a vector
 *
 * @param vector X of size N and type T
 *
 * @return diagonal matrix
 */
template <Field T, std::size_t N>
constexpr auto make_diag(const Vector<T, N>& X) {
  Matrix<T, N, N> diag;
  for (std::size_t i = 0; i < N; ++i) diag(i, i) = X(i);
  return diag;
}

/**
 * @brief Copy the half-open matrix row range `[Start, End)`.
 *
 * @param matrix A of type T Rows x Cols elements
 *
 * @return matrix
 */
template <std::size_t Start, std::size_t End, MatrixContainer Mat>
  requires(End - Start <= mat_rows_v<Mat> && Start < mat_rows_v<Mat>)
constexpr auto get_n_rows(const Mat& A) {
  using T = matrix_scalar_t<Mat>;
  constexpr auto Cols = mat_cols_v<Mat>;
  Matrix<T, End - Start, Cols> result;
  for (std::size_t i = Start, idx = 0; i < End; ++i, ++idx)
    for (std::size_t col = 0; col < Cols; ++col) result(idx, col) = A(i, col);
  return result;
}

/**
 * @brief Copy the half-open matrix column range `[Start, End)`.
 *
 * @param matrix A of type T Rows x Cols elements
 *
 * @return matrix
 */
template <std::size_t Start, std::size_t End, MatrixContainer Mat>
  requires(End - Start <= mat_cols_v<Mat> && Start < mat_cols_v<Mat>)
constexpr auto get_n_cols(const Mat& A) {
  using T = matrix_scalar_t<Mat>;
  constexpr auto Rows = mat_rows_v<Mat>;
  Matrix<T, Rows, End - Start> result;
  for (std::size_t col = Start, idx = 0; col < End; ++col, ++idx)
    for (std::size_t row = 0; row < Rows; ++row) result(row, idx) = A(row, col);
  return result;
}

/**
 * @brief Copy a compile-time rectangular matrix block.
 *
 * @param matrix A of type T Rows x Cols elements
 *
 * Row and column bounds are half-open:
 * `[StartRow, EndRow) x [StartCol, EndCol)`.
 *
 * @return Owning matrix with the selected block shape.
 */
template <std::size_t StartRow, std::size_t EndRow, std::size_t StartCol,
          std::size_t EndCol, MatrixContainer Mat>
  requires(EndRow - StartRow <= mat_rows_v<Mat> && StartRow < mat_rows_v<Mat> &&
           EndCol - StartCol <= mat_cols_v<Mat> && StartCol < mat_cols_v<Mat>)
constexpr auto get_block(const Mat& A) {
  using T = matrix_scalar_t<Mat>;
  Matrix<T, EndRow - StartRow, EndCol - StartCol> result;
  for (std::size_t i = StartRow, a = 0; i < EndRow; ++i, ++a)
    for (std::size_t j = StartCol, b = 0; j < EndCol; ++j, ++b)
      result(a, b) = A(i, j);
  return result;
}

/**
 * @brief Return a matrix copy with one compile-time block replaced.
 *
 * @param matrix A of type T Rows x Cols elements
 * @param matrix B of type T (EndRow - StartRow) x (EndCol - StartCol)
 *
 * Bounds are half-open. `A` is accepted by value, so this operation does not
 * mutate the caller's original matrix.
 *
 * @return Updated matrix copy.
 */
template <std::size_t StartRow, std::size_t EndRow, std::size_t StartCol,
          std::size_t EndCol, MatrixContainer Mat>
  requires(EndRow - StartRow <= mat_rows_v<Mat> && StartRow < mat_rows_v<Mat> &&
           EndCol - StartCol <= mat_cols_v<Mat> && StartCol < mat_cols_v<Mat>)
constexpr auto set_block(Mat A,
                         const Matrix<matrix_scalar_t<Mat>, EndRow - StartRow,
                                      EndCol - StartCol>& B) {
  for (std::size_t i = StartRow, a = 0; i < EndRow; ++i, ++a)
    for (std::size_t j = StartCol, b = 0; j < EndCol; ++j, ++b)
      A(i, j) = B(a, b);
  return A;
}

/**
 * @brief get upper right triangular matrix from A
 *
 * @param matrix A of type T Rows x Cols elements
 *
 * @return matrix
 */
template <MatrixContainer Mat>
constexpr auto get_upper_right_triangular_matrix(const Mat& A) {
  constexpr auto Rows = mat_rows_v<Mat>;
  constexpr auto Cols = mat_cols_v<Mat>;
  Mat R;
  for (std::size_t i = 0; i < Rows; ++i)
    for (std::size_t j = i; j < Cols; ++j) R(i, j) = A(i, j);
  return R;
}

/**
 * @brief get lower left triangular matrix from A
 *
 * @param matrix A of type T Rows x Cols elements
 *
 * @return matrix
 */
template <MatrixContainer Mat>
constexpr auto get_lower_left_triangular_matrix(const Mat& A) {
  constexpr auto Rows = mat_rows_v<Mat>;
  constexpr auto Cols = mat_cols_v<Mat>;
  Mat L;
  for (std::size_t i = 0; i < Cols; ++i)
    for (std::size_t j = i; j < Rows; ++j) L(j, i) = A(j, i);
  return L;
}

/**
 * @brief Compute the euclidean distance between two vectors (L2 norm of delta)
 *
 * @param vector X of type T with N elements
 * @param vector Y of type T with N elements
 *
 * @return distance between two vectors as a scalar
 */
template <VectorContainer Vec>
constexpr auto euclidean_distance(const Vec& X, const Vec& Y) {
  return L2<Vec>()(X - Y);
}

/**
 * @brief Compute the manhattan distance between two vectors (L1 norm of delta)
 *
 * @param vector X of type T with N elements
 * @param vector Y of type T with N elements
 *
 * @return distance between two vectors as a scalar
 */
template <VectorContainer Vec>
constexpr auto manhattan_distance(const Vec& X, const Vec& Y) {
  return L1<Vec>()(X - Y);
}

/**
 * @brief checks if two vectors are equivalent to a given tolerance (different
 * than == operator on vector which must be exactly equal
 *
 * @param vector X of type T N elements
 * @param vector Y of type T N elements
 *
 * @return T/F
 */
template <FieldVector Vec>
constexpr bool all_equal(const Vec& X, const Vec& Y,
                         real_type_t<vector_element_t<Vec>> tol = 1e-10) {
  return euclidean_distance(X, Y) <= tol;
}

/**
 * @brief checks if two given matrix are equivalent
 *
 * @param matrix A of type T with Rows x Cols elements
 * @param vector B of type T with Rows x Cols elements
 * @param tolerance set to 1e-10 by default
 *
 * @return T/F
 */
template <Field T, std::size_t Rows, std::size_t Cols>
constexpr bool all_equal(const Matrix<T, Rows, Cols>& A,
                         const Matrix<T, Rows, Cols>& B,
                         real_type_t<T> tol = 1e-10) {
  bool rv = true;
  for (std::size_t i = 0; i < Rows; ++i)
    rv &= all_equal(A.row(i), B.row(i), tol);
  return rv;
}

/**
 * @brief checks if the row space of one matrix is equivalent to the column
 * space of another matrix
 *
 * @param matrix A of type T with Rows x Cols elements
 * @param vector B of type T with Cols x Rows elements
 *
 * @return T/F
 */
template <MatrixContainer MatA, MatrixContainer MatB>
  requires(std::same_as<matrix_scalar_t<MatA>, matrix_scalar_t<MatB>> &&
           mat_rows_v<MatA> == mat_cols_v<MatB> &&
           mat_cols_v<MatA> == mat_rows_v<MatB>)
constexpr bool rows_equal_cols(const MatA& A, const MatB& B) {
  constexpr auto Rows = mat_rows_v<MatA>;
  bool rv = true;
  for (std::size_t i = 0; i < Rows; ++i)
    rv &= euclidean_distance(A.row(i), get_col(B, i)) <= 1e-10;
  return rv;
}

/**
 * @brief make symmetric matrix
 *
 * @param symmetric vector (flattened)
 *
 * @return matrix
 */
template <std::size_t N, FieldVector Vec>
  requires(vector_dim_v<Vec> == N * (N + 1) / 2)
constexpr auto make_smat(const Vec& L) {
  using T = vector_element_t<Vec>;
  Matrix<T, N, N> S{};
  for (std::size_t i = 0; i < N; ++i) {
    for (std::size_t j = 0; j <= i; ++j) {
      const auto idx = i * (i + 1) / 2 + j;
      S(i, j) = L(idx);
      if (i != j) S(i, j) /= linalg::sqrt2_v<T>;
    }
  }
  const auto S_T = transpose(S);
  Matrix<T, N, N> res{};
  for (std::size_t i = 0; i < N; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      if (i >= j) res(i, j) = S(i, j);
      else res(i, j) = S_T(i, j);
    }
  }
  return res;
}

/**
 * @brief get lower triangle with diag
 *
 * [ * 0 0 0 ]
 * [ * * 0 0 ]
 * [ * * * 0 ]
 * [ * * * * ]
 *
 * @param matrix A
 *
 * @return vec
 */
template <SquareMatrix Mat>
constexpr auto get_lower_triangle_with_diag(const Mat& A) {
  using T = matrix_scalar_t<Mat>;
  static constexpr std::size_t N = mat_rows_v<Mat>;
  static constexpr std::size_t K = N * (N + 1) / 2;
  Vector<T, K> vec{};
  for (std::size_t i = 0, k = 0; i < N; ++i)
    for (std::size_t j = 0; j <= i; ++j, ++k) vec(k) = A(i, j);
  return vec;
}

/**
 * @brief get svec encoding of a symmetric matrix
 *
 * Off-diagonal entries are scaled by sqrt(2) to preserve the Frobenius inner
 * product under vectorization.
 *
 * @param matrix A
 *
 * @return svec(A)
 */
template <SquareMatrix Mat>
constexpr auto get_svec(const Mat& A) {
  using T = matrix_scalar_t<Mat>;
  static constexpr std::size_t N = mat_rows_v<Mat>;
  static constexpr std::size_t K = N * (N + 1) / 2;
  Vector<T, K> vec{};
  for (std::size_t i = 0, k = 0; i < N; ++i) {
    for (std::size_t j = 0; j <= i; ++j, ++k) {
      vec(k) = A(i, j);
      if (i != j) vec(k) *= sqrt2_v<T>;
    }
  }
  return vec;
}

/**
 * @brief Compatibility spelling for `get_svec(A)`.
 * @param A Square matrix to encode in lower-triangular `svec` order.
 * @return Square-root-of-two-scaled symmetric coordinates.
 */
template <SquareMatrix Mat>
constexpr auto make_svec(const Mat& A) {
  return get_svec(A);
}

/**
 * @brief symmetric part of a square matrix
 *
 * sym(A) = 0.5 * (A + A^T)
 *
 * @param matrix A
 *
 * @return symmetric matrix
 */
template <SquareMatrix Mat>
constexpr auto symmetric_part(const Mat& A) {
  using T = matrix_scalar_t<Mat>;
  return T(0.5) * (A + transpose(A));
}

/**
 * @brief get lower triangle
 *
 * [ 0 0 0 0 ]
 * [ * 0 0 0 ]
 * [ * * 0 0 ]
 * [ * * * 0 ]
 *
 * @param matrix A
 *
 * @return vec
 */
template <SquareMatrix Mat>
constexpr auto get_lower_triangle(const Mat& A) {
  using T = matrix_scalar_t<Mat>;
  static constexpr std::size_t N = mat_rows_v<Mat>;
  static constexpr std::size_t K = N * (N + 1) / 2 - N;
  Vector<T, K> vec{};
  for (std::size_t i = 0, k = 0; i < N; ++i)
    for (std::size_t j = 0; j < i; ++j, ++k) vec(k) = A(i, j);
  return vec;
}

/**
 * @brief Construct a unit lower-triangular matrix from strict-lower entries.
 *
 * @param vector x
 *
 * Coordinates are consumed row by row below the diagonal. Diagonal entries are
 * set to one and upper-triangular entries remain value-initialized zero.
 *
 * @return Unit lower-triangular matrix `L`.
 */
template <SquareMatrix Mat, VectorContainer Vec>
  requires(std::same_as<matrix_scalar_t<Mat>, vector_element_t<Vec>>)
constexpr auto make_unit_lower_triangle(const Vec& x) {
  using T = matrix_scalar_t<Mat>;
  static constexpr std::size_t N = mat_rows_v<Mat>;
  static_assert(N * (N + 1) / 2 - N == vector_dim_v<Vec>,
                "Invalid vector size to make lower triangle of square matrix");
  Mat L{};
  for (std::size_t i = 0, k = 0; i < N; ++i) {
    std::size_t j = 0;
    for (; j < i; ++j, ++k) L(i, j) = x(k);
    if (i == j) L(i, j) = T(1);
  }

  return L;
}

/**
 * @brief Construct a lower-triangular matrix including its diagonal.
 *
 * @param x (vector)
 *
 * Coordinates are consumed row by row through the diagonal; entries above the
 * diagonal remain value-initialized zero.
 *
 * @return Lower-triangular matrix `L`.
 */
template <SquareMatrix Mat, VectorContainer Vec>
  requires(std::same_as<matrix_scalar_t<Mat>, vector_element_t<Vec>>)
constexpr auto make_lower_triangle(const Vec& x) {
  constexpr std::size_t N = mat_rows_v<Mat>;
  static_assert(N * (N + 1) / 2 == vector_dim_v<Vec>,
                "Invalid vector size to make lower triangle of square matrix");

  Mat L{};
  for (std::size_t i = 0, k = 0; i < N; ++i)
    for (std::size_t j = 0; j < i + 1; ++j, ++k) L(i, j) = x(k);

  return L;
}

} // namespace linalg
