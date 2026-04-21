// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_NEON_OPERATIONS_SHIFT_HPP
#define INCLUDE_GREX_BACKEND_NEON_OPERATIONS_SHIFT_HPP

#include <arm_neon.h>

#include "grex/backend/base.hpp"
#include "grex/backend/defs.hpp" // IWYU pragma: keep
#include "grex/backend/macros/for-each.hpp"
#include "grex/backend/neon/macros/types.hpp"
#include "grex/backend/neon/types.hpp"
#include "grex/backend/shared/operations/shift.hpp" // IWYU pragma: keep
#include "grex/base.hpp"

namespace grex::backend {
#define GREX_LSHIFT(KIND, BITS, SIZE) \
  inline NativeVector<KIND##BITS, SIZE> shift_left(NativeVector<KIND##BITS, SIZE> v, \
                                                   AnyIndexTag auto offset) { \
    return {.r = GREX_ISUFFIXED(vshlq_n, KIND, BITS)(v.r, offset.value)}; \
  }
GREX_FOREACH_INT_TYPE(GREX_LSHIFT, 128)

#define GREX_RSHIFT(KIND, BITS, SIZE) \
  inline NativeVector<KIND##BITS, SIZE> shift_right(NativeVector<KIND##BITS, SIZE> v, \
                                                    AnyIndexTag auto offset) { \
    if constexpr (offset == 0) { \
      return v; \
    } else { \
      return {.r = GREX_ISUFFIXED(vshrq_n, KIND, BITS)(v.r, offset.value)}; \
    } \
  }
GREX_FOREACH_INT_TYPE(GREX_RSHIFT, 128)
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_SHIFT_HPP
