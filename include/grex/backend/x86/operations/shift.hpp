// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_SHIFT_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_SHIFT_HPP

#include <cstddef>

#include <immintrin.h>

#include "grex/backend/base.hpp"
#include "grex/backend/macros/base.hpp"
#include "grex/backend/macros/for-each.hpp"
#include "grex/backend/x86/macros/for-each.hpp"
#include "grex/backend/x86/operations/set.hpp"
#include "grex/backend/x86/types.hpp"
#include "grex/base.hpp"

namespace grex::backend {
#define GREX_LSHIFT_INTRINSIC(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  return {.r = BITPREFIX##_slli_epi##BITS(v.r, offset.value)};
#define GREX_LSHIFT_8(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  const auto ret16 = GREX_CAT(BITPREFIX##_slli_epi16)(v.r, offset.value); \
  const auto mask = broadcast(u8(u8(-1) << offset), type_tag<u8x##SIZE>).r; \
  return {.r = GREX_CAT(BITPREFIX##_and_si##REGISTERBITS)(ret16, mask)};
#define GREX_LSHIFT_16 GREX_LSHIFT_INTRINSIC
#define GREX_LSHIFT_32 GREX_LSHIFT_INTRINSIC
#define GREX_LSHIFT_64 GREX_LSHIFT_INTRINSIC

#define GREX_LSHIFT(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  inline Vector<KIND##BITS, SIZE> shift_left(Vector<KIND##BITS, SIZE> v, \
                                             AnyIndexTag auto offset) { \
    GREX_LSHIFT_##BITS(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  }
#define GREX_LSHIFT_ALL(REGISTERBITS, BITPREFIX) \
  GREX_FOREACH_INT_TYPE(GREX_LSHIFT, REGISTERBITS, BITPREFIX, REGISTERBITS)
GREX_FOREACH_X86_64_LEVEL(GREX_LSHIFT_ALL)

#define GREX_SRLI_INTRINSIC(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  return {.r = BITPREFIX##_srli_epi##BITS(v.r, offset.value)};
#define GREX_RSHIFT_u8(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  const auto ret16 = GREX_CAT(BITPREFIX##_srli_epi16)(v.r, offset.value); \
  const auto mask = broadcast(u8(u8(-1) >> offset), type_tag<u8x##SIZE>).r; \
  return {.r = GREX_CAT(BITPREFIX##_and_si##REGISTERBITS)(ret16, mask)};
#define GREX_RSHIFT_u16 GREX_SRLI_INTRINSIC
#define GREX_RSHIFT_u32 GREX_SRLI_INTRINSIC
#define GREX_RSHIFT_u64 GREX_SRLI_INTRINSIC

#define GREX_SRAI_INTRINSIC(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  return {.r = BITPREFIX##_srai_epi##BITS(v.r, offset.value)};
#define GREX_RSHIFT_i8(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  /* logical shift right of the 16-bit lanes by 5 */ \
  const auto shifted = BITPREFIX##_srli_epi16(v.r, offset.value); \
  /* keep only the low 8-offset bits of each 8-bit word */ \
  const auto mask = broadcast(u8(u8(-1) >> offset), type_tag<u8x##SIZE>).r; \
  const auto masked = BITPREFIX##_and_si##REGISTERBITS(shifted, mask); \
  /* after shifting, the sign bit is in the `offset`th bit from the left, i.e. 128 >> offset */ \
  const auto sign = BITPREFIX##_set1_epi8(i8(128 >> offset)); \
  /* flip the sign bit */ \
  const auto flipped = BITPREFIX##_xor_si##REGISTERBITS(masked, sign); \
  /* subtracting the sign bit from the number with the flipped sign bit extends the sign */ \
  return {.r = BITPREFIX##_sub_epi8(flipped, sign)};
#define GREX_RSHIFT_i16 GREX_SRAI_INTRINSIC
#define GREX_RSHIFT_i32 GREX_SRAI_INTRINSIC
#if GREX_X86_64_LEVEL >= 4
#define GREX_RSHIFT_i64 GREX_SRAI_INTRINSIC
#else
#define GREX_RSHIFT_i64(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  /* [v[0, 32:], v[1, 32:], -, -] as u32x4/i32x4 */ \
  const auto shuf = BITPREFIX##_shuffle_epi32(v.r, 0b1101); \
  if constexpr (offset.value >= 32) { \
    /* [sign[0], sign[1], -, -] as i32x4 */ \
    const auto hi32 = BITPREFIX##_srai_epi32(shuf, 31); \
    /* [v[0] >> offset, v[1] >> offset, -, -] as i32x4 */ \
    const auto lo32 = BITPREFIX##_srai_epi32(shuf, offset.value - 32); \
    return {.r = BITPREFIX##_unpacklo_epi32(lo32, hi32)}; \
  } else { \
    /* [(v >> offset)[0, :32], -, (v >> offset)[1, :32], -] as u32x4 */ \
    const auto lo32 = BITPREFIX##_srli_epi64(v.r, offset.value); \
    /* [(v >> offset)[0, 32:], -, (v >> offset)[1, 32:], -] as i32x4 */ \
    const auto hi32 = BITPREFIX##_srai_epi32(shuf, offset.value); \
    /* [(v >> offset)[0, :32], (v >> offset)[1, :32], -, -] as u32x4 */ \
    const auto slo32 = BITPREFIX##_shuffle_epi32(lo32, 0b1000); \
    return {.r = BITPREFIX##_unpacklo_epi32(slo32, hi32)}; \
  }
#endif

#define GREX_RSHIFT(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  inline Vector<KIND##BITS, SIZE> shift_right(Vector<KIND##BITS, SIZE> v, \
                                              AnyIndexTag auto offset) { \
    GREX_RSHIFT_##KIND##BITS(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  }
#define GREX_RSHIFT_ALL(REGISTERBITS, BITPREFIX) \
  GREX_FOREACH_INT_TYPE(GREX_RSHIFT, REGISTERBITS, BITPREFIX, REGISTERBITS)
GREX_FOREACH_X86_64_LEVEL(GREX_RSHIFT_ALL)

#define GREX_SUBSUPER(NAME) \
  template<typename THalf> \
  inline SuperVector<THalf> NAME(SuperVector<THalf> v, AnyIndexTag auto offset) { \
    return {.lower = NAME(v.lower, offset), .upper = NAME(v.upper, offset)}; \
  } \
  template<IntVectorizable T, std::size_t tPart, std::size_t tSize> \
  inline SubVector<T, tPart, tSize> NAME(SubVector<T, tPart, tSize> v, AnyIndexTag auto offset) { \
    return SubVector<T, tPart, tSize>{NAME(v.full, offset)}; \
  }

GREX_SUBSUPER(shift_left)
GREX_SUBSUPER(shift_right)
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_SHIFT_HPP
