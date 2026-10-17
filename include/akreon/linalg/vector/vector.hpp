/*
 * Copyright © 2025-2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "akreon/linalg/assert.hpp"

#include <array>
#include <ranges>
#include <span>
#include <type_traits>

#include "algebra_util.hpp"

namespace linalg {

/**
 * @brief Owning, fixed-size vector container.
 *
 * `Vector` stores exactly `N` values inline in a `std::array`; its dimension is
 * part of the type and no dynamic allocation occurs. The container is
 * intentionally generic over `T`. Field-based algorithms opt in separately
 * through `FieldVector`, while `std::array` remains available for arbitrary
 * non-mathematical fixed-size data.
 *
 * Coordinates use zero-based `vector(index)` access guarded by `AK_ASSERT`.
 * Default construction value-initializes every coordinate. Iteration is const,
 * while mutable contiguous access is available through `view()`.
 *
 * @tparam T Stored coordinate type.
 * @tparam N Compile-time coordinate count.
 */
template <typename T, std::size_t N>
struct Vector {
  /**
   * @brief Construct from exactly `N` coordinate values.
   *
   * @param coords Coordinates in increasing index order.
   */
  template <typename... Coords>
  constexpr Vector(Coords... coords)
    requires(sizeof...(coords) == N)
      : coordinates{coords...} {}

  /** @brief Construct a value-initialized vector. */
  constexpr Vector() : coordinates{} {}

  /** @brief Const iterator to the first coordinate. */
  constexpr auto begin() const { return coordinates.begin(); }

  /** @brief Const iterator one past the final coordinate. */
  constexpr auto end() const { return coordinates.end(); }

  /**
   * @brief Read a coordinate at a zero-based index.
   * @param idx Coordinate index in `[0, N)`.
   * @return Const reference to the stored coordinate.
   */
  constexpr const T& operator()(std::size_t idx) const {
    AK_ASSERT(idx < N && "Vector const access index out of bounds");
    return coordinates[idx];
  }

  /**
   * @brief Read or modify a coordinate at a zero-based index.
   * @param idx Coordinate index in `[0, N)`.
   * @return Mutable reference to the stored coordinate.
   */
  constexpr T& operator()(std::size_t idx) {
    AK_ASSERT(idx < N && "Vector mutable access index out of bounds");
    return coordinates[idx];
  }

  /**
   * @brief Compare coordinates for exact element-wise equality.
   * @param other Shape-compatible vector to compare.
   */
  constexpr bool operator==(const Vector& other) const {
    return is_equal(other);
  }

  /**
   * @brief Compare coordinates for element-wise inequality.
   * @param other Shape-compatible vector to compare.
   */
  constexpr bool operator!=(const Vector& other) const {
    return !is_equal(other);
  }

  /**
   * @brief Obtain a mutable fixed-extent view of contiguous coordinate storage.
   */
  constexpr auto view() { return std::span<T, N>{coordinates}; }

  /** @brief Return the compile-time coordinate count `N`. */
  constexpr auto size() const { return N; }

private:
  /** @brief Implement exact coordinate-wise equality. */
  constexpr bool is_equal(const Vector& other) const {
    for (std::size_t i = 0; i < N; ++i)
      if (other.coordinates[i] != coordinates[i]) return false;
    return true;
  }

  std::array<T, N> coordinates; ///< Inline contiguous coordinate storage.
};

/**
 * @brief Deduce coordinate type and dimension from constructor arguments.
 *
 * The coordinate type is their `std::common_type_t`; the dimension is the
 * number of arguments. For example, `Vector{1.0, 2.0}` deduces
 * `Vector<double, 2>`.
 */
template <typename T, typename... Ts>
Vector(T, Ts...) -> Vector<std::common_type_t<T, Ts...>, 1 + sizeof...(Ts)>;

} // namespace linalg
