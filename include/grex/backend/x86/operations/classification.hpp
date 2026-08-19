// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_CLASSIFICATION_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_CLASSIFICATION_HPP

#include <immintrin.h>

#include "grex/backend/base.hpp"
#include "grex/backend/defs.hpp"
#include "grex/backend/macros/base.hpp"
#include "grex/backend/macros/for-each.hpp"
#include "grex/backend/macros/math.hpp"
#include "grex/backend/macros/types.hpp"
#include "grex/backend/x86/instruction-sets.hpp"
#include "grex/backend/x86/macros/for-each.hpp"
#include "grex/backend/x86/macros/intrinsics.hpp"
#include "grex/backend/x86/types.hpp"
#include "grex/base.hpp"

#if !GREX_F16_NATIVE_ARITHMETIC
#include "grex/backend/x86/operations/bitwise.hpp"
#include "grex/backend/x86/operations/compare.hpp"
#include "grex/backend/x86/operations/set.hpp"
#endif

namespace grex::backend {
// The `fpclass` category set 0x99 consists of QNaN, SNaN, -∞, and +∞, i.e. of exactly the
// non-finite values, which negating the resulting mask turns into the finite ones; based on VCL.
#define GREX_ISFIN_AVX512(KIND, BITS, SIZE, REGISTERBITS, RKIND) \
  return {.r = GREX_CAT(_knot_mask, GREX_MAX(SIZE, 8))( \
            GREX_CAT(GREX_BITPREFIX(REGISTERBITS), _fpclass_, GREX_FP_SUFFIX(BITS), \
                     _mask)(GREX_KINDCAST(RKIND, f, BITS, REGISTERBITS, v.r), 0x99))};

// The bit patterns of positive infinity, which are exactly the exponent masks, plus the upper half
// of the binary64 one
#define GREX_ISFIN_INFTY_16 0x7C00
#define GREX_ISFIN_INFTY_32 0x7F800000
#define GREX_ISFIN_INFTY_64 0x7FF0000000000000
#define GREX_ISFIN_INFTY_HIGH_64 0x7FF00000
// A value is non-finite iff all of its exponent bits are set, i.e. iff masking out sign and
// mantissa yields the bit pattern of infinity. Since that mask-out can only ever produce a subset
// of the exponent bits, the result is a non-negative integer which is at most the bit pattern of
// infinity, so a single signed comparison against the very same constant suffices.
#define GREX_ISFIN_FALLBACK_BASE(KIND, BITS, SIZE, REGISTERBITS, RKIND) \
  using Bits = NativeVector<i##BITS, SIZE>; \
  const auto infty = broadcast(i##BITS{GREX_ISFIN_INFTY_##BITS}, type_tag<Bits>); \
  const Bits expo = bitwise_and(Bits{GREX_KINDCAST(RKIND, i, BITS, REGISTERBITS, v.r)}, infty); \
  return {.r = compare_lt(expo, infty).r};
// Binary64 on level 1, which lacks `pcmpgtq`: masking out sign and mantissa clears the lower half
// of every value, so comparing the upper halves as `i32` and broadcasting the outcome down is far
// cheaper than emulating the 64-bit comparison.
#define GREX_ISFIN_FALLBACK_SSE64 \
  const __m128i infty = _mm_set1_epi32(GREX_ISFIN_INFTY_HIGH_64); \
  const __m128i expo = _mm_and_si128(_mm_castpd_si128(v.r), infty); \
  return {.r = _mm_shuffle_epi32(_mm_cmpgt_epi32(infty, expo), 0b11'11'01'01)};

#define GREX_ISFIN_FALLBACK_16 GREX_ISFIN_FALLBACK_BASE
#define GREX_ISFIN_FALLBACK_32 GREX_ISFIN_FALLBACK_BASE
#if GREX_X86_64_LEVEL >= 2
#define GREX_ISFIN_FALLBACK_64 GREX_ISFIN_FALLBACK_BASE
#else
#define GREX_ISFIN_FALLBACK_64(...) GREX_ISFIN_FALLBACK_SSE64
#endif
#define GREX_ISFIN_FALLBACK(KIND, BITS, ...) GREX_ISFIN_FALLBACK_##BITS(KIND, BITS, __VA_ARGS__)

#if GREX_X86_64_LEVEL >= 4
// `fpclass` is only available for binary16 through AVX512-FP16.
#if GREX_F16_NATIVE_ARITHMETIC
#define GREX_ISFIN_IMPL_16 GREX_ISFIN_AVX512
#else
#define GREX_ISFIN_IMPL_16 GREX_ISFIN_FALLBACK
#endif
#define GREX_ISFIN_IMPL_32 GREX_ISFIN_AVX512
#define GREX_ISFIN_IMPL_64 GREX_ISFIN_AVX512
#define GREX_ISFIN_IMPL(KIND, BITS, ...) GREX_ISFIN_IMPL_##BITS(KIND, BITS, __VA_ARGS__)
#else
#define GREX_ISFIN_IMPL GREX_ISFIN_FALLBACK
#endif

#define GREX_ISFIN(KIND, BITS, SIZE, REGISTERBITS) \
  inline NativeMask<KIND##BITS, SIZE> is_finite(NativeVector<KIND##BITS, SIZE> v) { \
    GREX_ISFIN_IMPL(KIND, BITS, SIZE, REGISTERBITS, GREX_REGKIND(KIND, BITS)) \
  }

#define GREX_ISFIN_ALL(REGISTERBITS, BITPREFIX) \
  GREX_FOREACH_FP_TYPE_EXT(GREX_ISFIN, REGISTERBITS, REGISTERBITS)
GREX_FOREACH_X86_64_LEVEL(GREX_ISFIN_ALL)
} // namespace grex::backend

#include "grex/backend/shared/operations/classification.hpp" // IWYU pragma: export

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_CLASSIFICATION_HPP
