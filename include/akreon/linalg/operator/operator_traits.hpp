/*
 * Copyright © 2025-2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "akreon/linalg/constants.hpp"
#include "akreon/linalg/space/space_concepts.hpp"
#include "akreon/linalg/space/coordinate_map.hpp"

namespace linalg {

/// @brief simple enum to define symmetric matrix types
/// INV = Invalid
/// IND = Indefinite
/// PSD = Positive Semidefinite
/// PD = Positive Definite
/// NSD = Negative Semidefinite
/// ND = Negative Definite
enum struct SymmetricOperatorType { INV = 0, IND, PSD, PD, NSD, ND };

/// @brief differential type
enum struct DifferentialType { PUSHFORWARD, PULLBACK };

/// @brief forward declaration for linear operator
template <SquareMatrix Mat, typename Derived>
struct Endomorphism;

/// @brief forward declaration for symmetric operator
template <SquareMatrix Mat, typename Derived>
struct SymmetricOperator;

/// @brief forward declaration for a matrix
template <SquareMatrix Mat>
struct SymmetricMatrix;

/// @brief forward declaration of hessenberg matrix
template <SquareMatrix Mat>
struct Hessenberg;

/// @brief forward declaration of hessian
template <SquareMatrix Mat, DifferentialType Type>
struct Hessian;

/// @brief forward declaration of LDL^T operator
template <SquareMatrix Mat>
struct LDL;

/// @brief forward declaration of P(x) / P(x^{-1})
template <SquareMatrix Mat, typename Circ>
struct QuadraticRep;

/// @brief forward declaration of cholesky
template <SquareMatrix Mat>
struct Cholesky;

/// @brief type traits definition for symmetric matrices (subspace of R^n)
template <SquareMatrix Mat>
struct space_traits<SymmetricMatrix<Mat>> {
  using value_type = matrix_scalar_t<Mat>;
  using Point = Mat;
  using scalar_type = value_type;
  using element_type = Point;

  static constexpr std::size_t N = mat_rows_v<Mat>;
  static constexpr std::size_t DIM = N * (N + 1) / 2;
  using coordinate_type = Vector<value_type, DIM>;
};

template <SquareMatrix Mat>
struct coordinate_map<SymmetricMatrix<Mat>> {
  using Space = SymmetricMatrix<Mat>;
  using T = space_scalar_t<Space>;
  static constexpr std::size_t N = mat_rows_v<Mat>;

  static constexpr auto to_coordinates(const Mat& element) {
    space_coordinate_t<Space> coordinate;
    for (std::size_t row = 0, index = 0; row < N; ++row) {
      for (std::size_t col = 0; col <= row; ++col, ++index) {
        coordinate(index) = element(row, col);
        if (row != col) coordinate(index) *= sqrt2_v<T>;
      }
    }
    return coordinate;
  }

  static constexpr auto from_coordinates(
      const space_coordinate_t<Space>& coordinate) {
    Mat element;
    for (std::size_t row = 0, index = 0; row < N; ++row) {
      for (std::size_t col = 0; col <= row; ++col, ++index) {
        auto value = coordinate(index);
        if (row != col) value /= sqrt2_v<T>;
        element(row, col) = element(col, row) = value;
      }
    }
    return element;
  }
};

/// @brief mat type traits definition for linear operator
template <SquareMatrix Mat, typename Derived>
struct mat_traits<Endomorphism<Mat, Derived>> {
  using value_type = matrix_scalar_t<Mat>;
  static constexpr std::size_t ROWS = mat_rows_v<Mat>;
  static constexpr std::size_t COLS = mat_cols_v<Mat>;
  using MatT = Mat;
};

/// @brief mat type traits definition for symmetric operator
template <SquareMatrix Mat, typename Derived>
struct mat_traits<SymmetricOperator<Mat, Derived>> {
  using value_type = matrix_scalar_t<Mat>;
  static constexpr std::size_t ROWS = mat_rows_v<Mat>;
  static constexpr std::size_t COLS = mat_cols_v<Mat>;
  using MatT = Mat;
};

/// @brief mat type traits definition for symmetric matrices
template <SquareMatrix Mat>
struct mat_traits<SymmetricMatrix<Mat>> {
  using value_type = matrix_scalar_t<Mat>;
  static constexpr std::size_t ROWS = mat_rows_v<Mat>;
  static constexpr std::size_t COLS = mat_cols_v<Mat>;
  using MatT = Mat;
};

/// @brief mat type traits definition for hessenberg matrices
template <SquareMatrix Mat>
struct mat_traits<Hessenberg<Mat>> {
  using value_type = matrix_scalar_t<Mat>;
  static constexpr std::size_t ROWS = mat_rows_v<Mat>;
  static constexpr std::size_t COLS = mat_cols_v<Mat>;
  using MatT = Mat;
};

/// @brief mat type traits definition for hessian matrices
template <SquareMatrix Mat, DifferentialType Type>
struct mat_traits<Hessian<Mat, Type>> {
  using value_type = matrix_scalar_t<Mat>;
  static constexpr std::size_t ROWS = mat_rows_v<Mat>;
  static constexpr std::size_t COLS = mat_cols_v<Mat>;
  using MatT = Mat;
};

/// @brief mat type traits definition for LDL
template <SquareMatrix Mat>
struct mat_traits<LDL<Mat>> {
  using value_type = matrix_scalar_t<Mat>;
  static constexpr std::size_t ROWS = mat_rows_v<Mat>;
  static constexpr std::size_t COLS = mat_cols_v<Mat>;
  using MatT = Mat;
};

/// @brief mat type traits definition for P(x)
template <SquareMatrix Mat, typename Circ>
struct mat_traits<QuadraticRep<Mat, Circ>> {
  using value_type = matrix_scalar_t<Mat>;
  static constexpr std::size_t ROWS = mat_rows_v<Mat>;
  static constexpr std::size_t COLS = mat_cols_v<Mat>;
  using MatT = Mat;
};

/// @brief mat type traits definition for Cholesky
template <SquareMatrix Mat>
struct mat_traits<Cholesky<Mat>> {
  using value_type = matrix_scalar_t<Mat>;
  static constexpr std::size_t ROWS = mat_rows_v<Mat>;
  static constexpr std::size_t COLS = mat_cols_v<Mat>;
  using MatT = Mat;
};

/// @brief vec type traits for linear operator
template <SquareMatrix Mat, typename Derived>
struct vector_traits<Endomorphism<Mat, Derived>> {
  using element_type = matrix_scalar_t<Mat>;
  static constexpr std::size_t N = mat_rows_v<Mat>;
  static constexpr std::size_t dim = N * N;
  using Vec = Vector<element_type, N>;
};

/// @brief vec type traits for symmetric operator
template <SquareMatrix Mat, typename Derived>
struct vector_traits<SymmetricOperator<Mat, Derived>> {
  using element_type = matrix_scalar_t<Mat>;
  static constexpr std::size_t N = mat_rows_v<Mat>;
  static constexpr std::size_t dim = N * (N + 1) / 2;
  using Vec = Vector<element_type, N>;
};

/// @brief vec type traits for symmetric matrices
template <SquareMatrix Mat>
struct vector_traits<SymmetricMatrix<Mat>> {
  using element_type = matrix_scalar_t<Mat>;
  static constexpr std::size_t N = mat_rows_v<Mat>;
  static constexpr std::size_t dim = N * (N + 1) / 2;
  using Vec = Vector<element_type, N>;
};

/// @brief vec type traits for hessenberg matrices
template <SquareMatrix Mat>
struct vector_traits<Hessenberg<Mat>> {
  using element_type = matrix_scalar_t<Mat>;
  static constexpr std::size_t N = mat_rows_v<Mat>;
  static constexpr std::size_t dim = N * N;
  using Vec = Vector<element_type, N>;
};

/// @brief vec type traits for hessian matrices
template <SquareMatrix Mat, DifferentialType Type>
struct vector_traits<Hessian<Mat, Type>> {
  using element_type = matrix_scalar_t<Mat>;
  static constexpr std::size_t N = mat_rows_v<Mat>;
  static constexpr std::size_t dim = N * N;
  using Vec = Vector<element_type, N>;
};

/// @brief vec type traits for LDL matrices
template <SquareMatrix Mat>
struct vector_traits<LDL<Mat>> {
  using element_type = matrix_scalar_t<Mat>;
  static constexpr std::size_t N = mat_rows_v<Mat>;
  static constexpr std::size_t dim = N * (N + 1) / 2;
  using Vec = Vector<element_type, N>;
};

/// @brief vec type traits for quadratic rep matrices
template <SquareMatrix Mat, typename Circ>
struct vector_traits<QuadraticRep<Mat, Circ>> {
  using element_type = matrix_scalar_t<Mat>;
  static constexpr std::size_t N = mat_rows_v<Mat>;
  static constexpr std::size_t dim = N * N;
  using Vec = Vector<element_type, N>;
};

/// @brief vec type traits for cholesky matrices
template <SquareMatrix Mat>
struct vector_traits<Cholesky<Mat>> {
  using element_type = matrix_scalar_t<Mat>;
  static constexpr std::size_t N = mat_rows_v<Mat>;
  static constexpr std::size_t dim = N * (N + 1) / 2;
  using Vec = Vector<element_type, N>;
};

/// @brief symmetric matrix space type trait
template <typename Space>
struct is_symmetric_matrix_space : std::false_type {};

/// @brief symmetric matrix space type trait
template <SquareMatrix Mat>
struct is_symmetric_matrix_space<SymmetricMatrix<Mat>> : std::true_type {};

/// @brief symmetric matrix space type trait var
template <typename Space>
inline constexpr bool is_symmetric_matrix_space_v =
    is_symmetric_matrix_space<remove_cvref_t<Space>>::value;

template <class Operator>
struct operator_traits;

template <typename T, std::size_t Rows, std::size_t Cols>
struct operator_traits<Matrix<T, Rows, Cols>> {
  using scalar_type = T;
  using domain_type = Vector<T, Cols>;
  using codomain_type = Vector<T, Rows>;
  using matrix_type = Matrix<T, Rows, Cols>;
};

template <class Operator>
  requires requires {
    typename mat_traits<remove_cvref_t<Operator>>::value_type;
    typename vector_traits<remove_cvref_t<Operator>>::Vec;
  }
struct operator_traits<Operator> {
  using scalar_type = typename mat_traits<remove_cvref_t<Operator>>::value_type;
  using domain_type = typename vector_traits<remove_cvref_t<Operator>>::Vec;
  using codomain_type = domain_type;
  using matrix_type = typename mat_traits<remove_cvref_t<Operator>>::MatT;
};

template <class Operator>
using operator_traits_t = operator_traits<remove_cvref_t<Operator>>;

template <class Operator>
using operator_scalar_t = typename operator_traits_t<Operator>::scalar_type;

template <class Operator>
using operator_domain_t = typename operator_traits_t<Operator>::domain_type;

template <class Operator>
using operator_codomain_t = typename operator_traits_t<Operator>::codomain_type;

template <class Operator>
using operator_matrix_t = typename operator_traits_t<Operator>::matrix_type;

template <MatrixContainer Mat>
constexpr auto apply(const Mat& matrix, const operator_domain_t<Mat>& vector)
    -> operator_codomain_t<Mat> {
  operator_codomain_t<Mat> result{};
  for (std::size_t row = 0; row < mat_rows_v<Mat>; ++row)
    for (std::size_t col = 0; col < mat_cols_v<Mat>; ++col)
      result(row) += matrix(row, col) * vector(col);
  return result;
}

template <class Operator>
concept LinearOperator =
    requires(const remove_cvref_t<Operator>& op,
             const operator_domain_t<Operator>& value) {
      { apply(op, value) } -> std::same_as<operator_codomain_t<Operator>>;
    } ||
    requires(const remove_cvref_t<Operator>& op,
             const operator_domain_t<Operator>& value) {
      { op.apply(value) } -> std::same_as<operator_codomain_t<Operator>>;
    };

template <class Operator>
concept EndomorphicOperator =
    LinearOperator<Operator> &&
    std::same_as<operator_domain_t<Operator>, operator_codomain_t<Operator>>;
} // namespace linalg
