/*
 * Copyright © 2025-2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "akreon/linalg/assert.hpp"
#include "akreon/linalg/linalg_ops.hpp"
#include "norm_traits.hpp"

namespace linalg {

/**
 * @brief compute the L2 norm of a vector
 *
 * || X ||_2 or sqrt(X^T X)
 *
 * @param vector X with N elements of type T
 *
 * @return scalar
 */
template <FieldVector Vec>
struct L2 {
  using domain_type = remove_cvref_t<Vec>;
  using scalar_type = vector_element_t<domain_type>;
  using codomain_type = real_type_t<scalar_type>;

  constexpr codomain_type operator()(const domain_type& x) const {
    codomain_type result = 0;
    const auto squared_norm = inner_product(x, x);
    if constexpr (Real<scalar_type>) result = squared_norm;
    else {
      AK_ASSERT(is_real(squared_norm) &&
                "L2 complex norm expects real inner product");
      result = squared_norm.real();
    }
    return constexpr_sqrt_safe(result);
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
