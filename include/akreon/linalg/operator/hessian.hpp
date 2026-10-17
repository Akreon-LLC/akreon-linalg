/*
 * Copyright © 2025-2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "akreon/linalg/linalg_util.hpp"

namespace linalg {

/**
 * @brief Hessian as a linear operator
 *
 * pushforward
 * H: A(X) -> B, acts as a push-forward operator mapping primal / tangent
 * vectors onto a change of gradient (i.e., F''(w)x -> s)
 *
 * pullback
 * H^{-1}: B(x) -> A, (i.e., F''(w)^{-1} s -> x) acts as a pull-back operator
 * mapping gradient space vectors back to tangent / primal direction
 */
template <SquareMatrix Mat,
          DifferentialType Type = DifferentialType::PUSHFORWARD>
struct Hessian : Endomorphism<Mat, Hessian<Mat, Type>> {
  /// @brief alias from base class
  using OperatorBase = Endomorphism<Mat, Hessian<Mat, Type>>;
  using typename OperatorBase::T;

  /// @brief helper from base to get num rows
  using OperatorBase::N;

  /// @brief alias to bring in vec
  using Vec = linalg::Vector<T, N>;

  /**
   * @brief CTOR
   *
   * @param A in R^{n x n}
   */
  constexpr explicit Hessian(const Mat& A) : H(A) {}

  /**
   * @brief get element at (i,j)
   *
   * @return element
   */
  constexpr auto operator()(std::size_t i, std::size_t j) const {
    return H(i, j);
  }

  /**
   * @brief apply
   *
   * H : A(X) -> B
   * H^{-1} : B(X) -> A
   *
   * @param Mat X
   *
   * @return B or A pending on diffeomorphism
   */
  template <MatrixContainer MatX>
    requires(N == mat_rows_v<MatX>)
  constexpr auto apply(const MatX& X) const {
    return H * X;
  }

  /**
   * @brief apply
   *
   * H : A(x) -> b
   * H^{-1} : A^{-1}(b) = x
   *
   * @param Vec x
   *
   * @return b or x pending on diffeomorphism
   */
  constexpr auto apply(const Vec& x) const { return H * x; }

  /**
   * @brief apply
   *
   * H : H(a) = aH (same for inv)
   *
   * @param a scalar
   *
   * @return scaled operator
   */
  constexpr auto apply(const T& a) const { return a * H; }

  /**
   * @brief is pullback
   *
   * @return T/F
   */
  constexpr bool is_pullback() const {
    return Type == DifferentialType::PULLBACK;
  }

  /** @brief member variables */
  Mat H; ///< hessian mat
};

} // namespace linalg
