/*
 * Copyright © 2025-2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "akreon/linalg/space/space_concepts.hpp"

namespace affine {

/// @brief forward declaration of affine map
template <linalg::MatrixContainer Mat>
class AffineMap;

/// @brief meta type traits for affine map
template <class Mat>
struct affine_traits;

/// @brief derived type traits for affine map
template <linalg::MatrixContainer Mat>
struct affine_traits<AffineMap<Mat>> {
  using AffineMat = Mat;
};

/// @brief alias to extract matrix type for affine map
template <class AffineMap>
using affine_mat_t =
    typename affine_traits<linalg::remove_cvref_t<AffineMap>>::AffineMat;

/// @brief concept for affine map
template <class Map>
concept AffineMapLike = requires(const Map& map) {
  { map.linear_part() };
  { map.translation() };
};

} // namespace affine
