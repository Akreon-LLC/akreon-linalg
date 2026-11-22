/*
 * Copyright © 2025-2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cmath>
#include <limits>

#include "akreon/linalg/assert.hpp"
#include "akreon/linalg/linalg_ops.hpp"
#include "norm_traits.hpp"

namespace linalg {

/**
 * @brief Compute the Euclidean norm from scaled real components.
 *
 * @tparam Vec Field vector domain.
 */
template <FieldVector Vec>
struct L2 {
  using domain_type = remove_cvref_t<Vec>;
  using scalar_type = vector_element_t<domain_type>;
  using codomain_type = real_type_t<scalar_type>;

  /**
   * @brief Evaluate the Euclidean norm of a real or complex vector.
   *
   * Let $c_i$ be the real components of \p x, taking both the real and
   * imaginary parts of each complex entry. For $s = \max_i |c_i| > 0$,
   * $$
   * \lVert x \rVert_2 = \sqrt{\sum_i c_i^2}
   *         = s \sqrt{\sum_i (c_i/s)^2}.
   * $$
   * Squaring only the ratios avoids overflow in the intermediate squares.
   * In one pass, keep the largest magnitude $s$ seen so far and the
   * normalized sum $q$. If the next magnitude $m$ exceeds $s$,
   * set $q \leftarrow 1 + q(s/m)^2$ before setting $s \leftarrow m$.
   * Otherwise, a nonzero component adds $(m/s)^2$ to $q$.
   *
   * @param x Input vector.
   * @return The Euclidean norm, or zero for an all-zero vector.
   */
  constexpr codomain_type operator()(const domain_type& x) const {
    codomain_type scale = 0;
    codomain_type sum = 0;
    // Accumulate component squares relative to the largest magnitude so far.
    const auto accumulate = [&](codomain_type component) {
      const auto magnitude = constexpr_abs(component);
      if (magnitude == 0) return;
      if (std::isnan(magnitude)) {
        sum = magnitude;
        return;
      }

      if (magnitude > scale) {
        const auto ratio = scale / magnitude;
        sum = 1 + sum * ratio * ratio;
        scale = magnitude;
      } else if (magnitude == scale) {
        // Equal magnitudes contribute one, including repeated infinities.
        sum += 1;
      } else {
        const auto ratio = magnitude / scale;
        sum += ratio * ratio;
      }
    };

    for (const auto& value : x) {
      if constexpr (Real<scalar_type>) {
        accumulate(value);
      } else {
        accumulate(value.real());
        accumulate(value.imag());
      }
    }

    if (std::isnan(sum)) return sum;
    if (scale == 0 || scale == std::numeric_limits<codomain_type>::infinity())
      return scale;

    if (sum <= 1) return scale;
    const auto tolerance = std::numeric_limits<codomain_type>::epsilon() * sum;
    return scale * constexpr_sqrt_safe(sum, codomain_type(0), codomain_type(1),
                                       tolerance);
  }
};

/**
 * @brief compute the L1 norm of a vector
 *
 * || X ||_1 or sum_i^N(|x_i|) for all i
 *
 * @param vector X with N elements of type T
 *
 * @return scalar
 */
template <FieldVector Vec>
struct L1 {
  using domain_type = remove_cvref_t<Vec>;
  using scalar_type = vector_element_t<domain_type>;
  using codomain_type = real_type_t<scalar_type>;

  constexpr codomain_type operator()(const domain_type& x) const {
    codomain_type sum = 0;
    for (const auto& _x : x) sum += constexpr_abs(_x);
    return sum;
  }
};

/**
 * @brief compute the max norm of a vector
 *
 * ||X||_infinity or max value of X (abs)
 *
 * @param vector X with N elements of type T
 *
 * @return scalar
 */
template <FieldVector Vec>
struct Linf {
  using domain_type = remove_cvref_t<Vec>;
  using scalar_type = vector_element_t<domain_type>;
  using codomain_type = real_type_t<scalar_type>;

  constexpr codomain_type operator()(const domain_type& x) const {
    codomain_type mx = 0;
    for (const auto& _x : x) mx = std::max(mx, constexpr_abs(_x));
    return mx;
  }
};

} // namespace linalg
