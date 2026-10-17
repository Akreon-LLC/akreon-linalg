/*
 * Copyright © 2025-2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "akreon/linalg/scalar/scalar_concepts.hpp"

namespace linalg {

template <Field T>
inline constexpr T sqrt2_v{T(1.414213562373095048801688724209698079L)};

template <Field T>
inline constexpr T inv_sqrt2_v{T(1) / sqrt2_v<T>};

template <Field T>
inline constexpr T ln2_v{T(0.693147180559945309417232121458176568L)};

} // namespace linalg
