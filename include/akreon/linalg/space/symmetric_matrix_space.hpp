/*
 * Copyright © 2025-2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "coordinate_map.hpp"
#include "akreon/linalg/operator/symmetric_matrix.hpp"

namespace linalg {

/**
 * @brief Space of real symmetric `Dim x Dim` matrices.
 *
 * The element representation enforces symmetry through `SymmetricMatrix`; the
 * coordinate dimension is `Dim * (Dim + 1) / 2`, not `Dim * Dim`.
 */
template <Real T, std::size_t Dim>
struct SymmetricMatrixSpace {};

/** @brief Space metadata for real symmetric matrices. */
template <Real T, std::size_t Dim>
struct space_traits<SymmetricMatrixSpace<T, Dim>> {
  /** @brief Real scalar field. */
  using scalar_type = T;
  /** @brief Canonical symmetry-aware matrix representation. */
  using element_type = SymmetricMatrix<Matrix<T, Dim, Dim>>;
  /** @brief Half-vectorized symmetric coordinate representation. */
  using coordinate_type = Vector<T, Dim*(Dim + 1) / 2>;
  /** @brief Number of independent entries in a symmetric matrix. */
  static constexpr std::size_t dim = Dim * (Dim + 1) / 2;
  /** @brief Compatibility spelling of the coordinate dimension. */
  static constexpr std::size_t DIM = dim;
};

/**
 * @brief Symmetric half-vectorization coordinate isomorphism.
 *
 * `to_coordinates` returns the element's `svec` representation. Off-diagonal
 * entries use the SDK's square-root-of-two scaling, preserving the Frobenius
 * inner product in coordinates. `from_coordinates` applies the inverse `smat`
 * transform and reconstructs a `SymmetricMatrix`.
 */
template <Real T, std::size_t Dim>
struct coordinate_map<SymmetricMatrixSpace<T, Dim>> {
  using Space = SymmetricMatrixSpace<T, Dim>;

  static constexpr auto to_coordinates(const space_element_t<Space>& element) {
    return element.svec;
  }

  static constexpr auto from_coordinates(
      const space_coordinate_t<Space>& coordinate) {
    return space_element_t<Space>{make_smat<Dim>(coordinate)};
  }
};

} // namespace linalg
