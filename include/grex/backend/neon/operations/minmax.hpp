// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_NEON_OPERATIONS_MINMAX_HPP
#define INCLUDE_GREX_BACKEND_NEON_OPERATIONS_MINMAX_HPP

#include <arm_neon.h>

#include "grex/backend/macros/base.hpp"
#include "grex/backend/neon/macros/types.hpp"
#include "grex/backend/neon/types.hpp"

namespace grex::backend {
#define GREX_MINMAX_IMPL_BASE(NAME, KIND, BITS) \
  return {.r = GREX_CAT(v##NAME##q_, GREX_ISUFFIX(KIND, BITS))(a.r, b.r)};
#define GREX_MINMAX_IMPL_INT8 GREX_MINMAX_IMPL_BASE
#define GREX_MINMAX_IMPL_INT16 GREX_MINMAX_IMPL_BASE
#define GREX_MINMAX_IMPL_INT32 GREX_MINMAX_IMPL_BASE
#define GREX_MINMAX_CMP_min vcgtq
#define GREX_MINMAX_CMP_max vcltq
#define GREX_MINMAX_IMPL_INT64(NAME, KIND, BITS) \
  const auto mask = GREX_CAT(GREX_MINMAX_CMP_##NAME, _, GREX_ISUFFIX(KIND, BITS))(a.r, b.r); \
  return {.r = vbslq_u64(mask, a.r, b.r)};

#define GREX_MINMAX_IMPL_f(NAME, KIND, BITS) return {.r = v##NAME##nmq_f##BITS(a.r, b.r)};
#define GREX_MINMAX_IMPL_i(NAME, KIND, BITS) GREX_MINMAX_IMPL_INT##BITS(NAME, KIND, BITS)
#define GREX_MINMAX_IMPL_u(NAME, KIND, BITS) GREX_MINMAX_IMPL_INT##BITS(NAME, KIND, BITS)

#define GREX_MINMAX(KIND, BITS, SIZE) \
  inline Vector<KIND##BITS, SIZE> min(Vector<KIND##BITS, SIZE> a, Vector<KIND##BITS, SIZE> b) { \
    GREX_MINMAX_IMPL_##KIND(min, KIND, BITS) \
  } \
  inline Vector<KIND##BITS, SIZE> max(Vector<KIND##BITS, SIZE> a, Vector<KIND##BITS, SIZE> b) { \
    GREX_MINMAX_IMPL_##KIND(max, KIND, BITS) \
  }

GREX_FOREACH_TYPE(GREX_MINMAX, 128)
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_MINMAX_HPP
