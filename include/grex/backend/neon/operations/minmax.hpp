// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_NEON_OPERATIONS_MINMAX_HPP
#define INCLUDE_GREX_BACKEND_NEON_OPERATIONS_MINMAX_HPP

#include <arm_neon.h>

#include "grex/backend/defs.hpp"
#include "grex/backend/macros/for-each.hpp"
#include "grex/backend/macros/types.hpp"
#include "grex/backend/neon/macros/types.hpp"
#include "grex/backend/neon/types.hpp"

namespace grex::backend {
#define GREX_MINMAX_IMPL_BASE(NAME, KIND, BITS) \
  return {.r = GREX_ISUFFIXED(v##NAME##q, KIND, BITS)(a.r, b.r)};
#define GREX_MINMAX_IMPL_INT8 GREX_MINMAX_IMPL_BASE
#define GREX_MINMAX_IMPL_INT16 GREX_MINMAX_IMPL_BASE
#define GREX_MINMAX_IMPL_INT32 GREX_MINMAX_IMPL_BASE
#define GREX_MINMAX_CMP_min vcltq
#define GREX_MINMAX_CMP_max vcgtq
#define GREX_MINMAX_IMPL_INT64(NAME, KIND, BITS) \
  const auto mask = GREX_ISUFFIXED(GREX_MINMAX_CMP_##NAME, KIND, BITS)(a.r, b.r); \
  return {.r = GREX_ISUFFIXED(vbslq, KIND, BITS)(mask, a.r, b.r)};

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

GREX_SUBVECTOR_BINARY(min)
GREX_SUBVECTOR_BINARY(max)

GREX_SUPERVECTOR_BINARY(min)
GREX_SUPERVECTOR_BINARY(max)
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_MINMAX_HPP
