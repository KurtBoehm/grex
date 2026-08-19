// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_NEON_OPERATIONS_MINMAX_HPP
#define INCLUDE_GREX_BACKEND_NEON_OPERATIONS_MINMAX_HPP

#include <arm_neon.h>

#include "grex/backend/base.hpp"
#include "grex/backend/defs.hpp"
#include "grex/backend/macros/for-each.hpp"
#include "grex/backend/macros/types.hpp"
#include "grex/backend/neon/macros/types.hpp"
#include "grex/backend/neon/operations/f16.hpp"
#include "grex/backend/neon/types.hpp"

// With FP16, native binary16 operations are defined together with the other versions, otherwise the
// round-trip through binary32 is used.

namespace grex::backend {
#define GREX_MINMAX_IMPL_BASE(NAME, KIND, BITS) \
  return {.r = GREX_ISUFFIXED(v##NAME##q, KIND, BITS)(ar, br)};
#define GREX_MINMAX_IMPL_INT8 GREX_MINMAX_IMPL_BASE
#define GREX_MINMAX_IMPL_INT16 GREX_MINMAX_IMPL_BASE
#define GREX_MINMAX_IMPL_INT32 GREX_MINMAX_IMPL_BASE
#define GREX_MINMAX_CMP_min vcltq
#define GREX_MINMAX_CMP_max vcgtq
#define GREX_MINMAX_IMPL_INT64(NAME, KIND, BITS) \
  const auto mask = GREX_ISUFFIXED(GREX_MINMAX_CMP_##NAME, KIND, BITS)(ar, br); \
  return {.r = GREX_ISUFFIXED(vbslq, KIND, BITS)(mask, ar, br)};

#define GREX_MINMAX_IMPL_f(NAME, KIND, BITS) \
  return {.r = to_stored<KIND##BITS>(v##NAME##nmq_f##BITS(ar, br))};
#define GREX_MINMAX_IMPL_i(NAME, KIND, BITS) GREX_MINMAX_IMPL_INT##BITS(NAME, KIND, BITS)
#define GREX_MINMAX_IMPL_u(NAME, KIND, BITS) GREX_MINMAX_IMPL_INT##BITS(NAME, KIND, BITS)

#define GREX_MINMAX(KIND, BITS, SIZE) \
  inline NativeVector<KIND##BITS, SIZE> min(NativeVector<KIND##BITS, SIZE> a, \
                                            NativeVector<KIND##BITS, SIZE> b) { \
    const auto ar = from_stored<KIND##BITS>(a.r); \
    const auto br = from_stored<KIND##BITS>(b.r); \
    GREX_MINMAX_IMPL_##KIND(min, KIND, BITS) \
  } \
  inline NativeVector<KIND##BITS, SIZE> max(NativeVector<KIND##BITS, SIZE> a, \
                                            NativeVector<KIND##BITS, SIZE> b) { \
    const auto ar = from_stored<KIND##BITS>(a.r); \
    const auto br = from_stored<KIND##BITS>(b.r); \
    GREX_MINMAX_IMPL_##KIND(max, KIND, BITS) \
  }

GREX_FOREACH_TYPE_OPT_EXT(GREX_MINMAX, 128)

GREX_NNVECTOR_BINARY(min)
GREX_NNVECTOR_BINARY(max)

#if !GREX_F16_NATIVE_ARITHMETIC
#define GREX_F16_MINMAX(NAME) \
  inline f16x8 NAME(f16x8 a, f16x8 b) { \
    return f32_to_f16(NAME(f16_to_f32(a), f16_to_f32(b))); \
  }
GREX_F16_MINMAX(min)
GREX_F16_MINMAX(max)
#undef GREX_F16_MINMAX
#endif
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_MINMAX_HPP
