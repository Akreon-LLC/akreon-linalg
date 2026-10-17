/*
 * Copyright © 2025-2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace linalg {

/// @brief forward declaration for L2 norm
template <FieldVector Vec>
struct L2;

/// @brief forward declaration for L1 norm
template <FieldVector Vec>
struct L1;

/// @brief forward declaration for Linf norm
template <FieldVector Vec>
struct Linf;

/// @brief declaring norm traits
template <class Norm>
struct norm_traits : std::false_type {};

/// @brief declaring norm type traits - L2
template <FieldVector Vec>
struct norm_traits<L2<Vec>> : std::true_type {
  using domain_type = remove_cvref_t<Vec>;
  using scalar_type = vector_element_t<domain_type>;
  using codomain_type = real_type_t<scalar_type>;
  static constexpr std::size_t dim = vector_dim_v<domain_type>;
};

/// @brief declaring norm type traits - L1
template <FieldVector Vec>
struct norm_traits<L1<Vec>> : std::true_type {
  using domain_type = remove_cvref_t<Vec>;
  using scalar_type = vector_element_t<domain_type>;
  using codomain_type = real_type_t<scalar_type>;
  static constexpr std::size_t dim = vector_dim_v<domain_type>;
};

/// @brief declaring norm type traits - Linf
template <FieldVector Vec>
struct norm_traits<Linf<Vec>> : std::true_type {
  using domain_type = remove_cvref_t<Vec>;
  using scalar_type = vector_element_t<domain_type>;
  using codomain_type = real_type_t<scalar_type>;
  static constexpr std::size_t dim = vector_dim_v<domain_type>;
};

/// @brief normalized norm traits
template <class Norm>
using norm_traits_t = norm_traits<remove_cvref_t<Norm>>;

template <class Norm>
using norm_domain_t = typename norm_traits_t<Norm>::domain_type;

template <class Norm>
using norm_scalar_t = typename norm_traits_t<Norm>::scalar_type;

template <class Norm>
using norm_codomain_t = typename norm_traits_t<Norm>::codomain_type;

template <class Norm>
inline constexpr std::size_t norm_dim_v = norm_traits_t<Norm>::dim;

/// @brief helper for is norm type
template <class Norm>
inline constexpr bool is_norm_v = norm_traits<remove_cvref_t<Norm>>{};

/// @brief concept for norm like
template <class Candidate>
concept NormFunction =
    is_norm_v<Candidate> && requires(const remove_cvref_t<Candidate>& norm,
                                     const norm_domain_t<Candidate>& value) {
      { norm(value) } -> std::same_as<norm_codomain_t<Candidate>>;
    };

} // namespace linalg
