// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_ABS_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_ABS_HPP

#include <cstddef> // IWYU pragma: keep

#include <immintrin.h>

#include "grex/backend/defs.hpp"
#include "grex/backend/macros/base.hpp"
#include "grex/backend/macros/for-each.hpp"
#include "grex/backend/x86/instruction-sets.hpp"
#include "grex/backend/x86/macros/for-each.hpp"
#include "grex/backend/x86/macros/intrinsics.hpp"
#include "grex/backend/x86/macros/types.hpp"
#include "grex/backend/x86/types.hpp"
#include "grex/base/defs.hpp" // IWYU pragma: keep

#if GREX_X86_64_LEVEL >= 4
#include "grex/backend/x86/operations/intrinsics.hpp" // IWYU pragma: keep
#endif

namespace grex::backend {
// Floating-point
#if GREX_X86_64_LEVEL >= 4
#define GREX_ABS_FP(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  return { \
    .r = GREX_BITNS(REGISTERBITS)::GREX_CAT(range_, GREX_FP_SUFFIX(BITS))(v.r, v.r, imm_tag<8>)};
#else
#define GREX_ABS_FP32(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  BITPREFIX##_castsi##REGISTERBITS##_ps(BITPREFIX##_set1_epi32(0x7FFFFFFF))
#define GREX_ABS_FP64(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  BITPREFIX##_castsi##REGISTERBITS##_pd(BITPREFIX##_set1_epi64x(0x7FFFFFFFFFFFFFFF))
#define GREX_ABS_FP(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  auto mask = GREX_ABS_FP##BITS(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS); \
  return {.r = GREX_CAT(BITPREFIX##_and_, GREX_FP_SUFFIX(BITS))(v.r, mask)};
#endif

// Integer
// Base case: Intrinsics
#define GREX_ABS_INT_INTRINSIC(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  return {.r = GREX_CAT(BITPREFIX##_abs_, GREX_EPI_SUFFIX(KIND, BITS))(v.r)};
// 64 bit above level 1
#define GREX_ABS_INT64_CMP(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  /* mask that is true if the component is negative */ \
  auto sign = BITPREFIX##_cmpgt_epi64(BITPREFIX##_setzero_si##REGISTERBITS(), v.r); \
  /* invert bits if negative */ \
  auto inv = BITPREFIX##_xor_si##REGISTERBITS(v.r, sign); \
  /* subtract the sign mask, i.e. add 1 */ \
  return {.r = BITPREFIX##_sub_epi64(inv, sign)};
// Specialized implementations at level 1
#define GREX_ABS_INT64_SHIFT \
  /* expand the sign to the upper 32 bits */ \
  __m128i upsign = _mm_srai_epi32(v.r, 31); \
  /* copy sign to the lower 32 bits */ \
  __m128i sign = _mm_shuffle_epi32(upsign, 0xF5); \
  /* invert bits if negative */ \
  __m128i inv = _mm_xor_si128(v.r, sign); \
  /* subtract the sign mask, i.e. add 1 */ \
  return {.r = _mm_sub_epi64(inv, sign)};
#define GREX_ABS_INT32_SHIFT \
  /* expand the sign to the entire 32 bits */ \
  __m128i sign = _mm_srai_epi32(v.r, 31); \
  /* invert bits if negative */ \
  __m128i inv = _mm_xor_si128(v.r, sign); \
  /* subtract the sign mask, i.e. add 1 */ \
  return {.r = _mm_sub_epi32(inv, sign)};
#define GREX_ABS_INT16_MAX \
  /* compute -c for each component c */ \
  __m128i negative = _mm_sub_epi16(_mm_setzero_si128(), v.r); \
  /* take max(c, -c) for each component c */ \
  return {.r = _mm_max_epi16(v.r, negative)};
#define GREX_ABS_INT8_MIN \
  /* compute -c for each component c */ \
  __m128i negative = _mm_sub_epi8(_mm_setzero_si128(), v.r); \
  /* take unsigned min(c, -c) for each component c */ \
  /* because the bigger signed value is the smaller unsigned value due to the sign bit */ \
  return {.r = _mm_min_epu8(v.r, negative)};
// Case distinction
#if GREX_X86_64_LEVEL >= 2
#define GREX_ABS_INT64 GREX_ABS_INT64_CMP
#define GREX_ABS_INT32 GREX_ABS_INT_INTRINSIC
#define GREX_ABS_INT16 GREX_ABS_INT_INTRINSIC
#define GREX_ABS_INT8 GREX_ABS_INT_INTRINSIC
#else
#define GREX_ABS_INT64(...) GREX_ABS_INT64_SHIFT
#define GREX_ABS_INT32(...) GREX_ABS_INT32_SHIFT
#define GREX_ABS_INT16(...) GREX_ABS_INT16_MAX
#define GREX_ABS_INT8(...) GREX_ABS_INT8_MIN
#endif
#if GREX_X86_64_LEVEL >= 4
#define GREX_ABS_INT GREX_ABS_INT_INTRINSIC
#else
#define GREX_ABS_INT(KIND, BITS, ...) GREX_ABS_INT##BITS(KIND, BITS, __VA_ARGS__)
#endif

// Base and case distinction
#define GREX_ABS_IMPL_f GREX_ABS_FP
#define GREX_ABS_IMPL_i GREX_ABS_INT
#define GREX_ABS_BASE(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  inline Vector<KIND##BITS, SIZE> abs(Vector<KIND##BITS, SIZE> v) { \
    GREX_ABS_IMPL_##KIND(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  }

#define GREX_ABS_ALL(REGISTERBITS, BITPREFIX) \
  GREX_FOREACH_FP_TYPE(GREX_ABS_BASE, REGISTERBITS, BITPREFIX, REGISTERBITS) \
  GREX_FOREACH_SINT_TYPE(GREX_ABS_BASE, REGISTERBITS, BITPREFIX, REGISTERBITS)
GREX_FOREACH_X86_64_LEVEL(GREX_ABS_ALL)

GREX_SUBVECTOR_UNARY(abs)
GREX_SUPERVECTOR_UNARY(abs)
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_ABS_HPP
