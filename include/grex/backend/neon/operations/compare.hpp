// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_NEON_OPERATIONS_COMPARE_HPP
#define INCLUDE_GREX_BACKEND_NEON_OPERATIONS_COMPARE_HPP

#include <cstddef>

#include <arm_neon.h>

#include "grex/backend/base.hpp"
#include "grex/backend/defs.hpp" // IWYU pragma: keep
#include "grex/backend/macros/for-each.hpp"
#include "grex/backend/macros/types.hpp"
#include "grex/backend/neon/macros/types.hpp"
#include "grex/backend/neon/operations/bitwise.hpp"
#include "grex/backend/neon/operations/f16.hpp"
#include "grex/backend/neon/operations/reinterpret.hpp"
#include "grex/backend/neon/types.hpp"
#include "grex/backend/shared/operations/compare.hpp" // IWYU pragma: export
#include "grex/base.hpp"

namespace grex::backend {
#define GREX_CMP_VEC(KIND, BITS, SIZE) \
  inline NativeMask<KIND##BITS, SIZE> compare_eq(NativeVector<KIND##BITS, SIZE> a, \
                                                 NativeVector<KIND##BITS, SIZE> b) { \
    return {.r = GREX_ISUFFIXED(vceqq, KIND, BITS)(a.r, b.r)}; \
  } \
  inline NativeMask<KIND##BITS, SIZE> compare_lt(NativeVector<KIND##BITS, SIZE> a, \
                                                 NativeVector<KIND##BITS, SIZE> b) { \
    return {.r = GREX_ISUFFIXED(vcltq, KIND, BITS)(a.r, b.r)}; \
  } \
  inline NativeMask<KIND##BITS, SIZE> compare_ge(NativeVector<KIND##BITS, SIZE> a, \
                                                 NativeVector<KIND##BITS, SIZE> b) { \
    return {.r = GREX_ISUFFIXED(vcgeq, KIND, BITS)(a.r, b.r)}; \
  }
#define GREX_CMP_MSK(KIND, BITS, SIZE) \
  inline NativeMask<KIND##BITS, SIZE> compare_eq(NativeMask<KIND##BITS, SIZE> a, \
                                                 NativeMask<KIND##BITS, SIZE> b) { \
    return {.r = vceqq_u##BITS(a.r, b.r)}; \
  }

// Binary16 requires special handling.
GREX_FOREACH_TYPE(GREX_CMP_VEC, 128)
// Binary16 is included, since mask comparisons are bitwise anyway.
GREX_FOREACH_TYPE_EXT(GREX_CMP_MSK, 128)

template<Vectorizable T, std::size_t tSize>
inline NativeMask<T, tSize> compare_neq(NativeVector<T, tSize> a, NativeVector<T, tSize> b) {
  return logical_not(compare_eq(a, b));
}

GREX_NNMASK_BINARY(compare_eq)

// Binary16: Comparing two vectors uses dedicated instructions if the FP16 extension is available
// and otherwise compares in binary32 via `f16_to_f32`, which is exact and therefore preserves the
// result; the generic `compare_neq` above already covers binary16 once `compare_eq`/`logical_not`
// do.
#if GREX_F16_NATIVE_ARITHMETIC
#define GREX_F16_CMP(NAME, INTRINSIC) \
  inline bf16x8 NAME(f16x8 a, f16x8 b) { \
    return {.r = INTRINSIC(as_f16(a.r), as_f16(b.r))}; \
  }
#else
#define GREX_F16_CMP(NAME, INTRINSIC) \
  inline bf16x8 NAME(f16x8 a, f16x8 b) { \
    const auto cmp32 = NAME(f16_to_f32(a), f16_to_f32(b)); \
    return {.r = vuzp1q_u16(as<u16>(cmp32.lower.r), as<u16>(cmp32.upper.r))}; \
  }
#endif

GREX_F16_CMP(compare_eq, vceqq_f16)
GREX_F16_CMP(compare_lt, vcltq_f16)
GREX_F16_CMP(compare_ge, vcgeq_f16)

#undef GREX_F16_CMP
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_COMPARE_HPP
