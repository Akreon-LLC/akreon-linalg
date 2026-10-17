/*
 * Copyright © 2025-2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <concepts>

#include "akreon/linalg/scalar/scalar_concepts.hpp"
#include "vector_traits.hpp"

namespace linalg {

/**
 * @brief Trait-registered fixed-size vector container.
 *
 * This metadata concept requires an element type and compile-time dimension.
 * It does not require field arithmetic, indexing, iteration, or vector-space
 * operations; algorithms state those additional requirements as needed.
 */
template <class Vec>
concept VectorContainer = requires {
  typename vector_traits_t<Vec>::element_type;
  { vector_dim_v<Vec> } -> std::convertible_to<std::size_t>;
};

/** @brief Registered vector whose element type satisfies `Field`. */
template <class Vec>
concept FieldVector = VectorContainer<Vec> && Field<vector_element_t<Vec>>;

/** @brief Field vector over a scalar classified as real. */
template <class Vec>
concept RealVector = FieldVector<Vec> && Real<vector_element_t<Vec>>;

/** @brief Field vector over a scalar classified as complex. */
template <class Vec>
concept ComplexVector = FieldVector<Vec> && Complex<vector_element_t<Vec>>;

/**
 * @brief Pair of vectors with exactly matching element types and dimensions.
 *
 * Compatibility is structural and does not imply that a particular binary
 * operation has been defined for the two representations.
 */
template <class VecA, class VecB>
concept CompatibleVectors =
    VectorContainer<VecA> && VectorContainer<VecB> &&
    std::same_as<vector_element_t<VecA>, vector_element_t<VecB>> &&
    vector_dim_v<VecA> == vector_dim_v<VecB>;

} // namespace linalg
