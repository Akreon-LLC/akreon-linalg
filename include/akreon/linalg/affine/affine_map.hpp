/*
 * Copyright © 2025-2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "affine_traits.hpp"
#include "akreon/linalg/assert.hpp"
#include "akreon/linalg/operator/matrix_inverse.hpp"

namespace affine {

/// @brief general affine map thats 2D
template <linalg::Field T>
using AffineMap2d = AffineMap<linalg::Matrix<T, 2, 2>>;

/// @brief general affine map thats 3D
template <linalg::Field T>
using AffineMap3d = AffineMap<linalg::Matrix<T, 3, 3>>;

/**
 * @brief make map ID
 *
 * @return [I; 0]
 */
template <linalg::SquareMatrix Mat>
constexpr AffineMap<Mat> make_map_id() {
  using T = linalg::matrix_scalar_t<Mat>;
  static constexpr std::size_t N = linalg::mat_rows_v<Mat>;
  return AffineMap<Mat>(linalg::make_identity<T, N>(), linalg::Vector<T, N>{});
}

/**
 * @brief make map selector
 *
 * @param idx selected ind
 *
 * @return [ 0... 1.. ; 0 ]
 */
template <linalg::MatrixContainer Mat>
  requires(linalg::mat_rows_v<Mat> == 1)
constexpr AffineMap<Mat> make_selector_map(std::size_t idx) {
  AK_ASSERT(idx < linalg::mat_cols_v<Mat> &&
            "make_selector_map index must be within [0, mat_cols)");

  using T = linalg::matrix_scalar_t<Mat>;
  Mat A{};
  A(0, idx) = T(1);
  return AffineMap<Mat>(A, linalg::Vector<T, 1>{});
}

/**
 * @brief AffineMap class definition
 *
 * Valid mapping which transforms one affine set to another.
 *
 * f(x) = U x + c
 *
 * Applies linear transform via U and then translation by vector c
 */
template <linalg::MatrixContainer Mat>
class AffineMap {
public:
  /// @brief alias to extract underlying type of affine set
  using T = linalg::matrix_scalar_t<Mat>;

  /// @brief helpers to extracting M x N matrix
  static constexpr std::size_t M = linalg::mat_rows_v<Mat>;
  static constexpr std::size_t N = linalg::mat_cols_v<Mat>;

  /// @brief alias for underlying vector spanning R^n (domain)
  using DomVec = linalg::Vector<T, N>;

  /// @brief alias for underlying vector spanning R^m (codomain)
  using AmbientVec = linalg::Vector<T, M>;

  /**
   * @brief CTOR
   *
   * empty map
   */
  constexpr explicit AffineMap() {}

  /**
   * @brief CTOR
   *
   * @param matrix M x N
   */
  constexpr explicit AffineMap(const Mat& U_in) : U(U_in) {}

  /**
   * @brief CTOR
   *
   * @param matrix of type T M x N elements
   * @param vector of type T M elements
   */
  constexpr explicit AffineMap(const Mat& U_in, const AmbientVec& c_in)
      : U(U_in), c(c_in) {}

  /**
   * @brief CTOR
   *
   * @param matrix of type T (M + 1) x (N + 1) elements
   */
  constexpr explicit AffineMap(const linalg::Matrix<T, M + 1, N + 1>& H)
      : U(get_block<0, M, 0, N>(H)),
        c(as_vector(get_block<0, M, N, N + 1>(H))) {}

  /**
   * @brief compose function
   *
   * Compose a sequence of affine maps as such:
   * f circ g = f(g(x))
   *
   * g(x) = A_g x + b_g
   * f(x) = A_f x + b_f
   * f(g(x)) = A_f (A_g x + b_g) + b_f = (A_f A_g) x + (A_f b_g + b_f)
   *
   * @param valid reference to an affine map object
   *
   * @return new affine map object with the new applied mapping
   */
  template <linalg::MatrixContainer OtherMat>
    requires(std::same_as<T, linalg::matrix_scalar_t<OtherMat>> &&
             linalg::mat_rows_v<OtherMat> == N)
  constexpr auto compose(const AffineMap<OtherMat>& other) const {
    constexpr std::size_t P = linalg::mat_cols_v<OtherMat>;
    const auto U_comp = U * other.linear_part();
    const auto c_comp = U * other.translation() + c;
    return AffineMap<linalg::Matrix<T, M, P>>(U_comp, c_comp);
  }

  /**
   * @brief inverse mapping
   *
   * Apply the inverse of the linear transformation matrix U and back out the
   * translation by the inverse.
   *
   * f(x) = U x + c
   * f^{-1}(f(x)) = id or x
   *
   * f^{-1}(x) = U^{-1} x - U^{-1} c
   * f^{-1}(f(x)) = U^{-1}(U x + c) - U^{-1} c == x
   *
   * @return new affine map object that is the inverse of the forward mapping
   */
  constexpr auto inverse() const {
    static_assert(M >= N, "short-fat TBD");
    const auto [U_inv, event] = linalg::MatrixInverse<decltype(U)>()(U);
    static_cast<void>(event);
    return AffineMap(U_inv, T(-1) * U_inv * c);
  }

  /**
   * @brief linear part
   *
   * @return U
   */
  constexpr auto linear_part() const { return U; }

  /**
   * @brief translation
   *
   * @return c
   */
  constexpr auto translation() const { return c; }

  /**
   * @brief equality comparison overload
   *
   * @param affine set
   *
   * @return T/F
   */
  constexpr bool operator==(const AffineMap<Mat>& other) const {
    return is_equal(other);
  }

  /**
   * @brief inequality comparison overload
   *
   * @param affine set
   *
   * @return T/F
   */
  constexpr bool operator!=(const AffineMap<Mat>& other) const {
    return !is_equal(other);
  }

  /**
   * @brief transform point to coords
   *
   * @param point in R^n
   *
   * @return transformed point
   */
  constexpr auto operator()(const DomVec& x) const { return U * x + c; }

private:
  /** @brief member variables for affine map */
  Mat U;        //<  valid linear transformation mapping
  AmbientVec c; //< c bias vector that translates the linear part to
                // a valid affine subspace

  /**
   * @brief equality comparison
   *
   * @param affine set
   *
   * @return T/F
   */
  constexpr bool is_equal(const AffineMap<Mat>& other) const {
    const auto linear_equal = all_equal(other.linear_part(), linear_part());
    const auto translation_equal =
        all_equal(other.translation(), translation());
    return linear_equal && translation_equal;
  }
};

} // namespace affine
