/*
 * Copyright © 2025-2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>

#include "akreon/linalg/assert.hpp"
#include "akreon/linalg/linalg_ops.hpp"
#include "akreon/linalg/linalg_util.hpp"
#include "akreon/linalg/norms/vector_norms.hpp"

namespace linalg {
/**
 * @brief Compact representation of a real or complex Householder reflector.
 *
 * The stored `v` and real coefficient `beta` represent
 * `P = I - beta * v * v^T` over a real field and
 * `P = I - beta * v * v^*` over a complex field. Construction follows the
 * Golub and Van Loan algorithm and selects a numerically stable leading
 * coordinate. Applying the reflector maps the input toward a multiple of the
 * first coordinate vector. For a non-axis real input, that multiple has
 * the opposite sign to the leading input coordinate to avoid cancellation.
 *
 * For `N == 1`, the representation is the identity reflector (`v = [1]`,
 * `beta = 0`). A zero complex input similarly requires no reflector.
 *
 * @tparam T Real or complex field scalar.
 * @tparam N Positive compile-time vector dimension.
 */
template <typename T, std::size_t N>
  requires(Field<T> && N > 0)
struct HouseholderVector {
  /**
   * @brief Construct a compact Householder reflector for x.
   *
   * @param[in] x Vector to reduce toward the first coordinate axis.
   */
  constexpr HouseholderVector(const Vector<T, N>& x) : beta(real_type_t<T>(0)) {
    make_householder(x, std::integral_constant<std::size_t, N>{});
  }

  Vector<T, N> v;      ///< Reflector direction vector.
  real_type_t<T> beta; ///< Real coefficient in `I - beta * v * v^*`.

private:
  /**
   * @brief Compute the reflector for N > 1.
   *
   * The unnamed dimension tag selects the general-size overload.
   *
   * @param[in] x Input vector used to populate v and beta.
   */
  constexpr void make_householder(const Vector<T, N>& x, auto) {
    // Scale x before computing its norm and reflector direction.
    real_type_t<T> scale = 0;
    for (const auto& value : x) scale = std::max(scale, constexpr_abs(value));
    v(0) = T(1);
    if (scale == real_type_t<T>(0)) return;
    for (std::size_t i = 0; i < N; ++i) v(i) = x(i) / scale;
    const auto norm = L2<Vector<T, N>>()(v);
    if constexpr (Complex<T>) {
      // Choose the leading phase so x + phase*||x||*e_0 avoids cancellation.
      // Normalize the phase at unit scale. The constexpr magnitude's
      // absolute square-root tolerance otherwise distorts tiny leading
      // entries and leaves a residual that packed QR silently discards.
      const auto phase_scale =
          std::max(constexpr_abs(v(0).real()), constexpr_abs(v(0).imag()));
      const auto leading =
          phase_scale == real_type_t<T>(0) ? T(1) : v(0) / phase_scale;
      const auto phase = leading / constexpr_abs(leading);
      v(0) += phase * norm;
      // QR and Hessenberg packing reconstruct the leading entry as one.
      v = (T(1) / v(0)) * v;
      const auto vv = inner_product(v, v);
      beta = real_type_t<T>(2) / vv.real();
    } else {
      const auto tail = get_elements<1, N>(v);
      if (inner_product(tail, tail) == T(0)) {
        beta = x(0) < T(0) ? T(2) : T(0);
        v(0) = T(1);
        return;
      }
      // Use the opposite-sign reflection and normalize its leading entry to
      // one.
      const auto leading = v(0) + (v(0) >= T(0) ? norm : -norm);
      v = (T(1) / leading) * v;
      v(0) = T(1);
      beta = T(2) / inner_product(v, v);
    }
  }

  /**
   * @brief Construct the identity reflector for N == 1.
   *
   * The unnamed input vector is unused; the dimension tag selects this
   * overload.
   */
  constexpr void make_householder(const Vector<T, N>&,
                                  std::integral_constant<std::size_t, 1>) {
    beta = real_type_t<T>(0);
    v(0) = T(1);
  }
};

} // namespace linalg
