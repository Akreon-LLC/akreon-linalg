/*
 * Copyright © 2025-2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <concepts>

#include "akreon/linalg/scalar/scalar_concepts.hpp"
#include "akreon/linalg/vector/vector_concepts.hpp"
#include "matrix_traits.hpp"

namespace linalg {

/**
 * @brief Trait-registered, fixed-shape matrix over a supported scalar field.
 *
 * Satisfaction requires `mat_traits<remove_cvref_t<Mat>>` to provide a field
 * scalar plus compile-time row and column counts. This is intentionally a
 * metadata/container classification; it does not by itself require indexing,
 * arithmetic, symmetry, or coordinate-space semantics. Algorithms that need
 * those operations express the additional requirements at their own boundary.
 */
template <class Mat>
concept MatrixContainer = requires {
  typename matrix_scalar_t<Mat>;
  { mat_rows_v<Mat> } -> std::convertible_to<std::size_t>;
  { mat_cols_v<Mat> } -> std::convertible_to<std::size_t>;
} && Field<matrix_scalar_t<Mat>>;

/** @brief Matrix container whose compile-time row and column counts are equal.
 */
template <class Mat>
concept SquareMatrix =
    MatrixContainer<Mat> && mat_rows_v<Mat> == mat_cols_v<Mat>;

/** @brief Matrix container with a real scalar field. */
template <class Mat>
concept RealMatrix = MatrixContainer<Mat> && Real<matrix_scalar_t<Mat>>;

/** @brief Matrix container with a complex scalar field. */
template <class Mat>
concept ComplexMatrix = MatrixContainer<Mat> && Complex<matrix_scalar_t<Mat>>;

/**
 * @brief Compile-time compatibility contract for matrix-vector multiplication.
 *
 * The scalar fields must match exactly, and the vector dimension must equal the
 * matrix column count. For `Mat * Vec`, the resulting vector dimension is the
 * matrix row count.
 */
template <class Mat, class Vec>
concept MatVecCompatible =
    MatrixContainer<Mat> && FieldVector<Vec> &&
    std::same_as<matrix_scalar_t<Mat>, vector_element_t<Vec>> &&
    mat_cols_v<Mat> == vector_dim_v<Vec>;

/**
 * @brief Compile-time compatibility contract for matrix multiplication.
 *
 * The scalar fields must match exactly, and the left column count must equal
 * the right row count. For `MatA * MatB`, the result shape is
 * `mat_rows_v<MatA> x mat_cols_v<MatB>`.
 */
template <class MatA, class MatB>
concept MatMulCompatible =
    MatrixContainer<MatA> && MatrixContainer<MatB> &&
    std::same_as<matrix_scalar_t<MatA>, matrix_scalar_t<MatB>> &&
    mat_cols_v<MatA> == mat_rows_v<MatB>;

} // namespace linalg
