/*
 * Copyright © 2025-2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <complex>
#include <concepts>
#include <type_traits>

#include "akreon/linalg/vector/vector_concepts.hpp"

namespace linalg {

/**
 * @brief Canonical inner product on a fixed-size vector over a scalar field.
 *
 * For a real vector, evaluation computes
 * `sum_i(lhs(i) * rhs(i))`. For a complex vector, it computes the Hermitian
 * inner product `sum_i(conj(lhs(i)) * rhs(i))`: the first argument is
 * conjugate-linear and the second is linear. This convention makes
 * `<x, x>` real and nonnegative mathematically, although its C++ return type
 * remains the vector's scalar type (including `std::complex<T>`).
 *
 * The vector dimension is known at compile time, the accumulation begins from
 * the scalar's value-initialized zero, and evaluation is `constexpr` whenever
 * the scalar operations are constant-evaluable.
 *
 * @tparam Vec Field-vector domain. cv/ref qualification is normalized in the
 * associated types.
 */
template <FieldVector Vec>
struct StandardInnerProduct {
  /** @brief Unqualified vector type accepted by both arguments. */
  using domain_type = remove_cvref_t<Vec>;
  /** @brief Real or complex scalar field of the vector domain. */
  using scalar_type = vector_element_t<domain_type>;
  /** @brief Scalar result type; equal to `scalar_type`. */
  using codomain_type = scalar_type;

  /**
   * @brief Pair two vectors from the same domain.
   * @param lhs First argument; conjugated element-wise for a complex field.
   * @param rhs Second argument; never conjugated.
   * @return Scalar inner-product value.
   */
  constexpr codomain_type operator()(const domain_type& lhs,
                                     const domain_type& rhs) const {
    codomain_type result{};
    for (std::size_t i = 0; i < vector_dim_v<domain_type>; ++i) {
      if constexpr (Complex<scalar_type>) result += std::conj(lhs(i)) * rhs(i);
      else result += lhs(i) * rhs(i);
    }
    return result;
  }
};

/**
 * @brief Customization point for inner-product associated types.
 *
 * A specialization must expose `domain_type`, `scalar_type`, and
 * `codomain_type`. The primary template is intentionally undefined so an
 * unregistered callable cannot accidentally satisfy `InnerProduct`.
 */
template <class Product>
struct inner_product_traits;

/** @brief Associated-type registration for `StandardInnerProduct`. */
template <FieldVector Vec>
struct inner_product_traits<StandardInnerProduct<Vec>> {
  using domain_type = remove_cvref_t<Vec>;
  using scalar_type = vector_element_t<domain_type>;
  using codomain_type = scalar_type;
};

/** @brief Normalized trait specialization for an inner-product type. */
template <class Product>
using inner_product_traits_t = inner_product_traits<remove_cvref_t<Product>>;

/** @brief Registered vector domain after product cv/ref normalization. */
template <class Product>
using inner_product_domain_t =
    typename inner_product_traits_t<Product>::domain_type;

/** @brief Registered scalar field associated with the product domain. */
template <class Product>
using inner_product_scalar_t =
    typename inner_product_traits_t<Product>::scalar_type;

/** @brief Registered result type of the inner product. */
template <class Product>
using inner_product_codomain_t =
    typename inner_product_traits_t<Product>::codomain_type;

/**
 * @brief Callable contract for a trait-registered inner-product operation.
 *
 * The product must accept two values from its registered domain and return its
 * registered codomain exactly. The concept checks the C++ interface, not the
 * mathematical inner-product axioms; custom products remain responsible for
 * conjugate symmetry, linearity, and positive definiteness.
 */
template <class Product>
concept InnerProduct = requires(const remove_cvref_t<Product>& product,
                                const inner_product_domain_t<Product>& lhs,
                                const inner_product_domain_t<Product>& rhs) {
  { product(lhs, rhs) } -> std::same_as<inner_product_codomain_t<Product>>;
};

/**
 * @brief Evaluate the canonical inner product for two field vectors.
 *
 * This is the standard call-site API and delegates to
 * `StandardInnerProduct<remove_cvref_t<Vec>>`. There is deliberately no
 * separate `dot` operation with different complex-number semantics.
 *
 * @param lhs First vector; conjugated for a complex field.
 * @param rhs Second vector.
 * @return Real or complex field scalar associated with `Vec`.
 */
template <FieldVector Vec>
constexpr auto inner_product(const Vec& lhs, const Vec& rhs) {
  return StandardInnerProduct<remove_cvref_t<Vec>>{}(lhs, rhs);
}

} // namespace linalg
