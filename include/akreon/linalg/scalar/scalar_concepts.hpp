/*
 * Copyright © 2025-2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <concepts>

#include "scalar_traits.hpp"

namespace linalg {

/**
 * @brief Type registered with the SDK scalar trait system.
 *
 * Satisfaction requires `scalar_traits<remove_cvref_t<T>>` to expose both the
 * normalized scalar representation and its associated real component type.
 * `Scalar` is a metadata classification only; it does not itself require
 * arithmetic operations.
 */
template <typename T>
concept Scalar = requires {
  typename scalar_traits_t<T>::scalar_type;
  typename scalar_traits_t<T>::real_type;
};

/**
 * @brief Registered scalar supporting the SDK's field arithmetic interface.
 *
 * Addition, subtraction, multiplication, and division between two normalized
 * values must each produce a result convertible to that same scalar type. The
 * concept checks the C++ expression contract, not mathematical field axioms;
 * in particular, it cannot prove associativity, distributivity, or that a
 * runtime divisor is nonzero.
 */
template <typename T>
concept Field = Scalar<T> && requires(const remove_cvref_t<T>& a,
                                      const remove_cvref_t<T>& b) {
  { a + b } -> std::convertible_to<remove_cvref_t<T>>;
  { a - b } -> std::convertible_to<remove_cvref_t<T>>;
  { a * b } -> std::convertible_to<remove_cvref_t<T>>;
  { a / b } -> std::convertible_to<remove_cvref_t<T>>;
};

/** @brief Field classified by `scalar_traits` as real. */
template <typename T>
concept Real = Field<T> && is_real_v<T>;

/** @brief Field classified by `scalar_traits` as complex. */
template <typename T>
concept Complex = Field<T> && is_complex_v<T>;

} // namespace linalg
