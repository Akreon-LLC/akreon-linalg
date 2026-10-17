/*
 * Copyright © 2025-2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <complex>
#include <type_traits>

namespace linalg {

/**
 * @brief Remove top-level const, volatile, and reference qualification.
 *
 * Public linalg traits normalize their lookup keys through this alias so a
 * scalar and cv/ref-qualified views of that scalar share one specialization.
 */
template <typename T>
using remove_cvref_t = std::remove_cvref_t<T>;

/**
 * @brief Customization point for scalar metadata.
 *
 * A specialization registers a scalar representation by exposing
 * `scalar_type`, `real_type`, `is_real`, and `is_complex`. The primary template
 * is intentionally undefined, preventing arbitrary arithmetic types from
 * silently entering the SDK's mathematical type system.
 *
 * `scalar_type` is the normalized scalar representation. `real_type` is its
 * underlying real component type, which is the scalar itself for a real type
 * and `T` for `std::complex<T>`.
 *
 * @tparam T Unqualified scalar representation being described.
 */
template <typename T>
struct scalar_traits;

/** @brief Register a standard floating-point type as a real scalar. */
template <std::floating_point T>
struct scalar_traits<T> {
  /** @brief Normalized scalar representation. */
  using scalar_type = T;
  /** @brief Associated real component type. */
  using real_type = T;

  /** @brief Whether the registered scalar belongs to the real category. */
  static constexpr bool is_real = true;
  /** @brief Whether the registered scalar belongs to the complex category. */
  static constexpr bool is_complex = false;
};

/** @brief Register `std::complex<T>` over a floating-point real component. */
template <std::floating_point T>
struct scalar_traits<std::complex<T>> {
  /** @brief Normalized complex scalar representation. */
  using scalar_type = std::complex<T>;
  /** @brief Floating-point component type underlying the complex scalar. */
  using real_type = T;

  /** @brief Whether the registered scalar belongs to the real category. */
  static constexpr bool is_real = false;
  /** @brief Whether the registered scalar belongs to the complex category. */
  static constexpr bool is_complex = true;
};

/** @brief Scalar traits selected after cv/ref normalization. */
template <typename T>
using scalar_traits_t = scalar_traits<remove_cvref_t<T>>;

/** @brief Normalized scalar representation registered for `T`. */
template <typename T>
using scalar_type_t = typename scalar_traits_t<T>::scalar_type;

/**
 * @brief Associated real component type registered for `T`.
 *
 * For example, both `double` and `std::complex<double>` have `double` as their
 * `real_type_t`.
 */
template <typename T>
using real_type_t = typename scalar_traits_t<T>::real_type;

/** @brief Whether normalized `T` is registered as a real scalar. */
template <typename T>
inline constexpr bool is_real_v = scalar_traits_t<T>::is_real;

/** @brief Whether normalized `T` is registered as a complex scalar. */
template <typename T>
inline constexpr bool is_complex_v = scalar_traits_t<T>::is_complex;

} // namespace linalg
