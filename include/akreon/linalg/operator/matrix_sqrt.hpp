/*
 * Copyright © 2025-2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "akreon/linalg/numerical_config.hpp"
#include "akreon/linalg/linalg_util.hpp"
#include "akreon/linalg/operator/matrix_inverse.hpp"

namespace linalg {

/**
 * @brief Matrix sqrt (X^{1/2})
 *
 * @tparam MatIn Real square matrix type.
 */
template <SquareMatrix MatIn>
struct MatrixSqrt {
  /// @brief alias & types
  using Mat = linalg::remove_cvref_t<MatIn>;
  using T = matrix_scalar_t<Mat>;
  static constexpr std::size_t N = mat_rows_v<Mat>;

  static_assert(Real<T>, "Sqrt assumes reals domain");
  static_assert(N >= 1, "Sqrt requires matrix dim of at least 1");

  /**
   * @brief Compute the matrix square root and inverse square root.
   *
   * @param[in] A Real square input matrix for Denman-Beavers iteration.
   * @return Tuple (Mat root, Mat inverse_root, NumericalEvent event).
   */
  constexpr auto operator()(const Mat& A) const { return apply(A); }

  /**
   * @brief Compute the matrix square root and inverse square root.
   *
   * @param[in] A Real square input matrix for Denman-Beavers iteration.
   * @return Tuple (Mat root, Mat inverse_root, NumericalEvent event).
   */
  constexpr auto apply(const Mat& A) const {
    // Initialize the coupled Denman-Beavers iterates for A^(1/2) and A^(-1/2).
    Mat Y{A};
    Mat Z{make_identity<T, N>()};

    for (std::size_t i = 0; i < config.steps; ++i) {
      const auto [Y_inv, Y_event] = MatrixInverse<Mat>()(Y);
      if (Y_event != linalg::NumericalEvent::NONE)
        return std::tuple(Mat{}, Mat{}, Y_event);

      const auto [Z_inv, Z_event] = MatrixInverse<Mat>()(Z);
      if (Z_event != linalg::NumericalEvent::NONE)
        return std::tuple(Mat{}, Mat{}, Z_event);

      // Y_{k+1} = (Y_k + Z_k^-1)/2; Z_{k+1} = (Z_k + Y_k^-1)/2.
      const auto Y_next = T(0.5) * (Y + Z_inv);
      const auto Z_next = T(0.5) * (Z + Y_inv);

      // Accept when the square-root residual Y^2 - A meets tolerance.
      if (all_equal(Y_next * Y_next, A, config.tol))
        return std::tuple(Y_next, Z_next, linalg::NumericalEvent::NONE);

      Y = Y_next;
      Z = Z_next;
    }

    return std::tuple(Y, Z, linalg::NumericalEvent::DB_MAX_ITERS_REACHED);
  }

  /** @brief member vars **/
  config::FactorConfig<T> config{
      .steps = 128, .tol = T(1e-12), .floor_eps = T(1e-14)}; ///< factor config
};

} // namespace linalg
