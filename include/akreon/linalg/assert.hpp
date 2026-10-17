/*
 * Copyright © 2025-2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cassert>

#ifdef AKREON_REPORT_OUTPUT
#define AK_ASSERT(x) assert(x)
#else
#define AK_ASSERT(x) ((void)0)
#endif
