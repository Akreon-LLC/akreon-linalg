/*
 * Copyright © 2025-2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <utility>

#include "akreon/linalg/scalar/scalar_traits.hpp"

namespace linalg {

/** @brief Forward declaration of the owning fixed-size vector container. */
template <typename T, std::size_t Dim>
struct Vector;

/**
 * @brief Customization point for fixed-size vector metadata.
 *
 * A specialization registers `element_type` and compile-time `dim`. The
 * primary template is intentionally undefined, so arbitrary indexable types do
 * not accidentally satisfy vector concepts. Public aliases normalize cv/ref
 * qualification before lookup.
 */
template <class Vec>
struct vector_traits;

/**
 * @brief Traits and constexpr construction helpers for `Vector<T, Dim>`.
 */
template <typename T, std::size_t Dim>
struct vector_traits<Vector<T, Dim>> {
  /** @brief Stored coordinate type. */
  using element_type = T;
  /** @brief Compile-time coordinate count. */
  static constexpr std::size_t dim = Dim;

  /** @brief Construct from exactly `Dim` coordinates in index order. */
  template <typename... Ts>
  static constexpr auto operator()(Ts&&... xs) {
    static_assert(sizeof...(xs) == Dim,
                  "vector_traits::operator() requires exactly Dim coordinates");
    Vector<T, Dim> vector;
    std::size_t i = 0;
    auto set_element = [&](const T& x) { vector(i++) = x; };
    ((void)set_element(std::forward<Ts>(xs)), ...);
    return vector;
  }

  /** @brief Construct a vector whose coordinates are all equal to `x`. */
  static constexpr auto fill(const T& x) {
    Vector<T, Dim> vector;
    for (std::size_t i = 0; i < Dim; ++i) vector(i) = x;
    return vector;
  }
};

/** @brief Vector traits selected after cv/ref normalization. */
template <class Vec>
using vector_traits_t = vector_traits<remove_cvref_t<Vec>>;

/** @brief Registered vector coordinate type. */
template <class Vec>
using vector_element_t = typename vector_traits_t<Vec>::element_type;

/** @brief Registered compile-time vector dimension. */
template <class Vec>
inline constexpr std::size_t vector_dim_v = vector_traits_t<Vec>::dim;

/**
 * @brief Recursively obtain the non-vector scalar underlying a type.
 *
 * The primary form normalizes `T`; the `Vector` specialization recursively
 * unwraps nested vectors. This differs from `vector_element_t`, which removes
 * exactly one registered vector layer.
 */
template <class T>
struct vector_scalar {
  using type = remove_cvref_t<T>;
};

/** @brief Recursively unwrap one `Vector` layer. */
template <typename T, std::size_t Dim>
struct vector_scalar<Vector<T, Dim>> : vector_scalar<T> {};

/** @brief Fully unwrapped scalar type underlying a possibly nested vector. */
template <class Vec>
using vector_scalar_t = typename vector_scalar<remove_cvref_t<Vec>>::type;

} // namespace linalg
