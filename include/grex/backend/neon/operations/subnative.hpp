// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_NEON_OPERATIONS_SUBNATIVE_HPP
#define INCLUDE_GREX_BACKEND_NEON_OPERATIONS_SUBNATIVE_HPP

#include <arm_neon.h>

#include "grex/backend/defs.hpp"
#include "grex/backend/macros/base.hpp"
#include "grex/backend/macros/for-each.hpp"
#include "grex/backend/macros/math.hpp"
#include "grex/backend/neon/operations/reinterpret.hpp"
#include "grex/backend/neon/types.hpp"

namespace grex::backend {
#define GREX_CUTOFF_SUB(KIND, BITS, PART, SIZE) \
  inline Vector<KIND##BITS, SIZE> full_cutoff(SubVector<KIND##BITS, PART, SIZE> v) { \
    const auto reinterpreted = as<GREX_CAT(u, GREX_MULTIPLY(BITS, PART))>(v.registr()); \
    const auto low = GREX_CAT(vget_low_u, GREX_MULTIPLY(BITS, PART))(reinterpreted); \
    auto out = GREX_CAT(vdupq_n_u, GREX_MULTIPLY(BITS, PART))(0); \
    out = GREX_CAT(vcopyq_lane_u, GREX_MULTIPLY(BITS, PART))(out, 0, low, 0); \
    return {.r = as<KIND##BITS>(out)}; \
  }
GREX_FOREACH_SUB(GREX_CUTOFF_SUB)
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_SUBNATIVE_HPP
