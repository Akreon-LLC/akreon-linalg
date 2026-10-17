/*
 * Copyright © 2025-2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <concepts>

#include "coordinate_map.hpp"

namespace linalg {

/**
 * @brief Trait-registered finite-dimensional mathematical space.
 *
 * A candidate space must expose a scalar field, element representation, and
 * compile-time dimension through `space_traits`. This concept classifies space
 * metadata only; it does not require vector-space arithmetic or a coordinate
 * conversion.
 */
template <class Candidate>
concept Space = requires {
  typename space_scalar_t<Candidate>;
  typename space_element_t<Candidate>;
  { space_dim_v<Candidate> } -> std::convertible_to<std::size_t>;
};

/**
 * @brief Finite-dimensional space with explicit element-coordinate conversion.
 *
 * `coordinates<Space>(element)` must produce the registered coordinate type,
 * and `from_coordinates<Space>(coordinate)` must reconstruct the registered
 * element type. The concept checks the exact C++ interface but cannot prove
 * that the two operations are mutual inverses.
 */
template <class Candidate>
concept CoordinatizedSpace =
    Space<Candidate> &&
    requires(const space_element_t<Candidate>& element,
             const space_coordinate_t<Candidate>& coordinate) {
      {
        coordinates<Candidate>(element)
      } -> std::same_as<space_coordinate_t<Candidate>>;
      {
        from_coordinates<Candidate>(coordinate)
      } -> std::same_as<space_element_t<Candidate>>;
    };

} // namespace linalg
