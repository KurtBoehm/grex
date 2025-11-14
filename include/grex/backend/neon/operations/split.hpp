// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_NEON_OPERATIONS_SPLIT_HPP
#define INCLUDE_GREX_BACKEND_NEON_OPERATIONS_SPLIT_HPP

#include <cstddef>

#include <arm_neon.h>

#include "grex/backend/choosers.hpp"
#include "grex/backend/macros/for-each.hpp"
#include "grex/backend/macros/math.hpp"
#include "grex/backend/neon/types.hpp" // IWYU pragma: keep

namespace grex::backend {
#define GREX_SPLIT(KIND, BITS, PART, SIZE) \
  inline VectorFor<KIND##BITS, PART> get_low(VectorFor<KIND##BITS, GREX_MULTIPLY(PART, 2)> v) { \
    return VectorFor<KIND##BITS, PART>{v.registr()}; \
  } \
  inline VectorFor<KIND##BITS, PART> get_high(VectorFor<KIND##BITS, GREX_MULTIPLY(PART, 2)> v) { \
    const auto r = v.registr(); \
    const auto out = GREX_ISUFFIXED(vextq, KIND, BITS)(r, r, PART); \
    return VectorFor<KIND##BITS, PART>{out}; \
  }
GREX_FOREACH_SUB(GREX_SPLIT)
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_SPLIT_HPP
