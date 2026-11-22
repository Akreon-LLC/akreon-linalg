/*
 * Copyright © 2025-2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cmath>
#include <complex>
#include <limits>
#include <type_traits>

#include <catch2/catch_test_macros.hpp>

#include "akreon/linalg/linalg.hpp"
#include "akreon/linalg/norms/norm_traits.hpp"
#include "akreon/linalg/operator/operator_traits.hpp"

namespace {
struct IndexedVector {
  double value;
  constexpr double operator()(std::size_t) const { return value; }
};
} // namespace

template <>
struct linalg::vector_traits<IndexedVector> {
  using element_type = double;
  static constexpr std::size_t dim = 1;
};

namespace {

using RealVector = linalg::Vector<double, 3>;
using ComplexVector = linalg::Vector<std::complex<double>, 3>;
using RectangularMatrix = linalg::Matrix<double, 2, 3>;
using SquareMatrix = linalg::Matrix<double, 3, 3>;
using RealSpace = linalg::RowVector<double, 3>;
using ComplexSpace = linalg::RowVector<std::complex<double>, 3>;
using Coordinates3 = linalg::CoordinateSpace<double, 3>;
using MatrixCoordinates = linalg::MatrixSpace<double, 2, 3>;
using SymmetricCoordinates = linalg::SymmetricMatrixSpace<double, 2>;

struct UnsupportedScalar {};

TEST_CASE("linalg compile-time contracts", "[linalg][contracts]") {
  STATIC_REQUIRE(linalg::Scalar<double>);
  STATIC_REQUIRE(linalg::Scalar<std::complex<double>>);
  STATIC_REQUIRE(!linalg::Scalar<int>);
  STATIC_REQUIRE(!linalg::Scalar<UnsupportedScalar>);

  STATIC_REQUIRE(linalg::Field<double>);
  STATIC_REQUIRE(linalg::Field<const double>);
  STATIC_REQUIRE(linalg::Field<volatile double&>);
  STATIC_REQUIRE(linalg::Field<std::complex<double>>);
  STATIC_REQUIRE(linalg::Field<const std::complex<float>&>);
  STATIC_REQUIRE(!linalg::Field<int>);
  STATIC_REQUIRE(!linalg::Field<UnsupportedScalar>);

  STATIC_REQUIRE(linalg::Real<double>);
  STATIC_REQUIRE(linalg::Real<const float&>);
  STATIC_REQUIRE(!linalg::Real<std::complex<double>>);
  STATIC_REQUIRE(linalg::Complex<std::complex<double>>);
  STATIC_REQUIRE(linalg::Complex<const std::complex<float>&>);
  STATIC_REQUIRE(!linalg::Complex<double>);

  STATIC_REQUIRE(std::same_as<linalg::scalar_type_t<const double&>, double>);
  STATIC_REQUIRE(
      std::same_as<linalg::scalar_type_t<const std::complex<double>&>,
                   std::complex<double>>);
  STATIC_REQUIRE(
      std::same_as<linalg::real_type_t<const std::complex<double>&>, double>);
  STATIC_REQUIRE(linalg::is_complex_v<const std::complex<double>&>);
  STATIC_REQUIRE(!linalg::is_complex_v<const double&>);
  STATIC_REQUIRE(linalg::sqrt2_v<double> > 1.4);
  STATIC_REQUIRE(linalg::inv_sqrt2_v<double> > 0.7);
  STATIC_REQUIRE(linalg::ln2_v<double> > 0.0);

  STATIC_REQUIRE(linalg::VectorContainer<RealVector>);
  STATIC_REQUIRE(linalg::VectorContainer<const RealVector&>);
  STATIC_REQUIRE(linalg::VectorContainer<ComplexVector>);
  STATIC_REQUIRE(!linalg::VectorContainer<double>);
  STATIC_REQUIRE(linalg::FieldVector<RealVector>);
  STATIC_REQUIRE(linalg::FieldVector<const ComplexVector&>);
  STATIC_REQUIRE(!linalg::FieldVector<linalg::Vector<int, 3>>);
  STATIC_REQUIRE(linalg::VectorContainer<IndexedVector>);
  STATIC_REQUIRE(!linalg::FieldVector<IndexedVector>);
  STATIC_REQUIRE(linalg::RealVector<const RealVector&>);
  STATIC_REQUIRE(!linalg::ComplexVector<RealVector>);
  STATIC_REQUIRE(linalg::ComplexVector<const ComplexVector&>);
  STATIC_REQUIRE(!linalg::RealVector<ComplexVector>);
  STATIC_REQUIRE(linalg::vector_dim_v<const RealVector&> == 3);
  STATIC_REQUIRE(
      std::same_as<linalg::vector_element_t<const RealVector&>, double>);
  STATIC_REQUIRE(!linalg::VectorContainer<RectangularMatrix>);
  STATIC_REQUIRE(!linalg::FieldVector<RectangularMatrix>);
  STATIC_REQUIRE(
      std::same_as<linalg::matrix_scalar_t<RectangularMatrix>, double>);

  STATIC_REQUIRE(linalg::MatrixContainer<RectangularMatrix>);
  STATIC_REQUIRE(linalg::MatrixContainer<const RectangularMatrix&>);
  STATIC_REQUIRE(!linalg::SquareMatrix<RectangularMatrix>);
  STATIC_REQUIRE(linalg::SquareMatrix<SquareMatrix>);
  STATIC_REQUIRE(linalg::SquareMatrix<const SquareMatrix&>);
  STATIC_REQUIRE(linalg::RealMatrix<const RectangularMatrix&>);
  STATIC_REQUIRE(!linalg::ComplexMatrix<RectangularMatrix>);
  STATIC_REQUIRE(
      linalg::ComplexMatrix<linalg::Matrix<std::complex<double>, 2, 2>>);
  STATIC_REQUIRE(linalg::mat_rows_v<const RectangularMatrix&> == 2);
  STATIC_REQUIRE(linalg::mat_cols_v<const RectangularMatrix&> == 3);

  using DeducedMatrix = decltype(linalg::Matrix{linalg::Vector{1.0, 2.0},
                                                linalg::Vector{3.0, 4.0}});
  STATIC_REQUIRE(std::same_as<DeducedMatrix, linalg::Matrix<double, 2, 2>>);
  STATIC_REQUIRE(
      std::same_as<decltype(std::declval<RectangularMatrix&>()(0, 0)),
                   double&>);
  STATIC_REQUIRE(
      std::same_as<decltype(std::declval<const RectangularMatrix&>()(0, 0)),
                   const double&>);

  STATIC_REQUIRE(linalg::CompatibleVectors<RealVector, const RealVector&>);
  STATIC_REQUIRE(!linalg::CompatibleVectors<RealVector, ComplexVector>);
  STATIC_REQUIRE(linalg::MatVecCompatible<RectangularMatrix, RealVector>);
  STATIC_REQUIRE(
      !linalg::MatVecCompatible<RectangularMatrix, linalg::Vector<double, 2>>);
  STATIC_REQUIRE(linalg::MatMulCompatible<RectangularMatrix,
                                          linalg::Matrix<double, 3, 4>>);
  STATIC_REQUIRE(!linalg::MatMulCompatible<RectangularMatrix,
                                           linalg::Matrix<double, 2, 4>>);

  STATIC_REQUIRE(linalg::VectorSpace<RealSpace>);
  STATIC_REQUIRE(linalg::VectorSpace<const RealSpace&>);
  STATIC_REQUIRE(linalg::RnSpace<RealSpace>);
  STATIC_REQUIRE(!linalg::RnSpace<ComplexSpace>);
  STATIC_REQUIRE(linalg::space_dim_v<const RealSpace&> == 3);
  STATIC_REQUIRE(std::same_as<linalg::space_type_t<const RealSpace&>, double>);
  STATIC_REQUIRE(
      std::same_as<linalg::space_point_t<const RealSpace&>, RealVector>);

  STATIC_REQUIRE(linalg::CoordinatizedSpace<Coordinates3>);
  STATIC_REQUIRE(linalg::CoordinatizedSpace<MatrixCoordinates>);
  STATIC_REQUIRE(linalg::CoordinatizedSpace<SymmetricCoordinates>);
  STATIC_REQUIRE(linalg::space_dim_v<const Coordinates3&> == 3);
  STATIC_REQUIRE(linalg::space_dim_v<const MatrixCoordinates&> == 6);
  STATIC_REQUIRE(linalg::space_dim_v<const SymmetricCoordinates&> == 3);
  STATIC_REQUIRE(std::same_as<linalg::space_element_t<MatrixCoordinates>,
                              RectangularMatrix>);
  STATIC_REQUIRE(std::same_as<linalg::space_coordinate_t<MatrixCoordinates>,
                              linalg::Vector<double, 6>>);

  using EuclideanNorm = linalg::L2<RealVector>;
  using ComplexEuclideanNorm = linalg::L2<ComplexVector>;
  using RectangularFrobeniusNorm = linalg::Fro<RectangularMatrix>;
  STATIC_REQUIRE(linalg::NormFunction<EuclideanNorm>);
  STATIC_REQUIRE(linalg::NormFunction<const EuclideanNorm&>);
  STATIC_REQUIRE(linalg::NormFunction<ComplexEuclideanNorm>);
  STATIC_REQUIRE(linalg::NormFunction<RectangularFrobeniusNorm>);
  STATIC_REQUIRE(
      std::same_as<linalg::norm_domain_t<const EuclideanNorm&>, RealVector>);
  STATIC_REQUIRE(std::same_as<linalg::norm_scalar_t<ComplexEuclideanNorm>,
                              std::complex<double>>);
  STATIC_REQUIRE(
      std::same_as<linalg::norm_codomain_t<ComplexEuclideanNorm>, double>);
  STATIC_REQUIRE(linalg::norm_dim_v<const EuclideanNorm&> == 3);
  STATIC_REQUIRE(linalg::norm_dim_v<RectangularFrobeniusNorm> == 6);
  STATIC_REQUIRE(!linalg::NormFunction<UnsupportedScalar>);

  using RealInnerProduct = linalg::StandardInnerProduct<RealVector>;
  using ComplexInnerProduct = linalg::StandardInnerProduct<ComplexVector>;
  STATIC_REQUIRE(linalg::InnerProduct<RealInnerProduct>);
  STATIC_REQUIRE(linalg::InnerProduct<const ComplexInnerProduct&>);
  STATIC_REQUIRE(
      std::same_as<linalg::inner_product_domain_t<ComplexInnerProduct>,
                   ComplexVector>);
  STATIC_REQUIRE(
      std::same_as<linalg::inner_product_codomain_t<ComplexInnerProduct>,
                   std::complex<double>>);

  STATIC_REQUIRE(linalg::LinearOperator<SquareMatrix>);
  STATIC_REQUIRE(linalg::LinearOperator<const SquareMatrix&>);
  STATIC_REQUIRE(linalg::LinearOperator<RectangularMatrix>);
  STATIC_REQUIRE(linalg::EndomorphicOperator<SquareMatrix>);
  STATIC_REQUIRE(!linalg::EndomorphicOperator<RectangularMatrix>);
  STATIC_REQUIRE(std::same_as<linalg::operator_domain_t<RectangularMatrix>,
                              linalg::Vector<double, 3>>);
  STATIC_REQUIRE(std::same_as<linalg::operator_codomain_t<RectangularMatrix>,
                              linalg::Vector<double, 2>>);
}

} // namespace

TEST_CASE("coordinate maps round trip", "[linalg][contracts][space]") {
  constexpr RectangularMatrix matrix{linalg::Vector{1.0, 2.0, 3.0},
                                     linalg::Vector{4.0, 5.0, 6.0}};
  constexpr auto coordinate = linalg::coordinates<MatrixCoordinates>(matrix);
  STATIC_REQUIRE(coordinate == linalg::Vector{1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
  STATIC_REQUIRE(linalg::from_coordinates<MatrixCoordinates>(coordinate) ==
                 matrix);

  constexpr linalg::Matrix<double, 2, 2> symmetric{linalg::Vector{2.0, 1.0},
                                                   linalg::Vector{1.0, 2.0}};
  constexpr linalg::space_element_t<SymmetricCoordinates> element{symmetric};
  constexpr auto svec = linalg::coordinates<SymmetricCoordinates>(element);
  STATIC_REQUIRE(svec(0) == 2.0);
  STATIC_REQUIRE(svec(2) == 2.0);
  constexpr auto reconstructed =
      linalg::from_coordinates<SymmetricCoordinates>(svec);
  STATIC_REQUIRE(linalg::coordinates<SymmetricCoordinates>(reconstructed) ==
                 svec);
}

TEST_CASE("inner products follow the field convention",
          "[linalg][contracts][inner-product]") {
  constexpr RealVector real_lhs{1.0, 2.0, 3.0};
  constexpr RealVector real_rhs{4.0, 5.0, 6.0};
  STATIC_REQUIRE(linalg::inner_product(real_lhs, real_rhs) == 32.0);

  constexpr ComplexVector complex_lhs{std::complex<double>{1.0, 1.0},
                                      std::complex<double>{2.0, -1.0},
                                      std::complex<double>{0.0, 3.0}};
  constexpr auto squared_norm = linalg::inner_product(complex_lhs, complex_lhs);
  STATIC_REQUIRE(squared_norm == std::complex<double>{16.0, 0.0});
  STATIC_REQUIRE(linalg::constexpr_abs(
                     linalg::L2<ComplexVector>{}(complex_lhs)-4.0) < 1e-14);
}

TEST_CASE("L2 preserves representable norms at extreme magnitudes",
          "[linalg][contracts][norm]") {
  constexpr double large = 0x1p700;
  constexpr double small = 0x1p-700;
  constexpr RealVector zero{0.0, 0.0, 0.0};
  constexpr RealVector large_vector{0.0, -large, small};
  constexpr RealVector small_vector{small, 0.0, 0.0};
  constexpr auto norm = linalg::L2<RealVector>{};
  STATIC_REQUIRE(norm(zero) == 0.0);
  STATIC_REQUIRE(norm(large_vector) == large);
  STATIC_REQUIRE(norm(small_vector) == small);
  STATIC_REQUIRE(norm(RealVector{1e-100, 0.0, 0.0}) == 1e-100);
  STATIC_REQUIRE(norm(RealVector{1e-6, 0.0, 0.0}) == 1e-6);
  STATIC_REQUIRE(norm(RealVector{1e72, 0.0, 0.0}) == 1e72);
  STATIC_REQUIRE(norm(RealVector{1e100, 0.0, 0.0}) == 1e100);
  STATIC_REQUIRE(norm(RealVector{1e150, 0.0, 0.0}) == 1e150);
  STATIC_REQUIRE(linalg::euclidean_distance(small_vector, zero) == small);
  STATIC_REQUIRE_FALSE(linalg::all_equal(small_vector, zero, small / 2));

  constexpr auto complex_norm = linalg::L2<ComplexVector>{};
  constexpr ComplexVector large_complex{std::complex{3 * large, 4 * large},
                                        std::complex<double>{},
                                        std::complex<double>{}};
  constexpr ComplexVector small_complex{std::complex{3 * small, 4 * small},
                                        std::complex<double>{},
                                        std::complex<double>{}};
  STATIC_REQUIRE(complex_norm(large_complex) == 5 * large);
  STATIC_REQUIRE(complex_norm(small_complex) == 5 * small);

  const auto infinity = std::numeric_limits<double>::infinity();
  const auto nan = std::numeric_limits<double>::quiet_NaN();
  REQUIRE(std::isinf(norm(RealVector{infinity, -infinity, 1.0})));
  REQUIRE(std::isnan(norm(RealVector{nan, 1.0, infinity})));
  REQUIRE(std::isnan(norm(RealVector{infinity, 1.0, nan})));
}

TEST_CASE("matrix operator traits preserve domain and codomain",
          "[linalg][contracts][operator]") {
  constexpr RectangularMatrix matrix{linalg::Vector{1.0, 2.0, 3.0},
                                     linalg::Vector{4.0, 5.0, 6.0}};
  constexpr linalg::Vector<double, 3> value{1.0, 0.0, -1.0};
  STATIC_REQUIRE(linalg::apply(matrix, value) ==
                 linalg::Vector<double, 2>{-2.0, -2.0});
}
