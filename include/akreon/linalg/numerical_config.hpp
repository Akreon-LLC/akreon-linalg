/*
 * Copyright © 2025-2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "akreon/linalg/space/space_concepts.hpp"
#include "akreon/linalg/vector/algebra_util.hpp"
namespace config {
/**
 * @brief Configuration for generic numerical factorizations.
 */
template <linalg::Real T>
struct FactorConfig {
  std::size_t steps{64}; ///< total number of iter
  T tol{1e-8};           ///< tolerance of of solution
  T floor_eps{1e-10};    ///< machine precision (lower bounds)
};

} // namespace config
namespace linalg {

/**
 * @brief NumericalEvent
 */
enum struct NumericalEvent : uint8_t {
  NONE = 0,

  // detection / observation
  RANK_DEFICIENCY,
  INDEFINITE_FACTORIZATION,
  NEAR_SINGULAR_TRIANGULAR_SOLVE,
  NUMERICAL_SAFEGUARD,
  DB_MAX_ITERS_REACHED,

  // failure
  FACTORIZATION_FAILED,
  EVD_FAILED
};

/**
 * @brief is_fatal_event
 *
 * @param event (NumericalEvent)
 *
 * @return T/F
 */
inline constexpr bool is_fatal_event(NumericalEvent event) {
  return event == NumericalEvent::FACTORIZATION_FAILED ||
         event == NumericalEvent::EVD_FAILED;
}

} // namespace linalg
