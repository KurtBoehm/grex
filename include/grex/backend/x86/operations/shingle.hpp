// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_SHINGLE_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_SHINGLE_HPP

#define GREX_X86_64_LEVEL 4
#include "grex/backend/defs.hpp"
#include "grex/backend/x86/helpers.hpp"
#include "grex/backend/x86/instruction-sets.hpp"
#include "grex/backend/x86/operations/expand-scalar.hpp"
#include "grex/backend/x86/types.hpp"
#include "grex/base/defs.hpp"

namespace grex::backend {
// shingle up with a zero
#define GREX_ZUSHINGLE_SHIFT128(KIND, BITS, ...) \
  const __m128i ivec = GREX_KINDCAST(KIND, i, BITS, 128, v.r); \
  return {.r = GREX_KINDCAST(i, KIND, BITS, 128, _mm_bslli_si128(ivec, GREX_BIT2BYTE(BITS)))};
#define GREX_ZUSHINGLE_ALIGNR_AVX(KIND, BITS, ...) \
  const __m256i ivec = GREX_KINDCAST(KIND, i, BITS, 256, v.r); \
  /* zero in the lower, v[n/2:] in the upper half */ \
  const __m256i perm = \
    _mm256_inserti128_si256(_mm256_setzero_si256(), _mm256_castsi256_si128(ivec), 1); \
  /* shift v up by one element in each half, shifting in 0 in the lower \
   * and v[n/2-1] in the upper half */ \
  const __m256i alignr = _mm256_alignr_epi8(ivec, perm, 16 - GREX_BIT2BYTE(BITS)); \
  return {.r = GREX_KINDCAST(i, KIND, BITS, 256, alignr)};
#define GREX_ZUSHINGLE_ALIGNR_AVX512(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  const auto ivec = GREX_KINDCAST(KIND, i, BITS, REGISTERBITS, v.r); \
  const auto alignr = \
    BITPREFIX##_alignr_epi##BITS(ivec, BITPREFIX##_setzero_si##REGISTERBITS(), SIZE - 1); \
  return {.r = GREX_KINDCAST(i, KIND, BITS, REGISTERBITS, alignr)};
// This is analogous to the AVX implementation with _mm512_alignr_epi64
// instead of _mm256_inserti128_si256
#define GREX_ZUSHINGLE_DBLALIGN(KIND, BITS, ...) \
  /* the comments assume 8-bit integers, but the steps are completely analogous for 16 bits */ \
  /* [0]*16 + v[:48] → alr[i] = 0 if i < 16 else v[i-16] */ \
  const __m512i alr = _mm512_alignr_epi64(v.r, _mm512_setzero_si512(), 6); \
  /* [alr[15], *v[:15], alr[31], *v[16:31], alr[47], v[32:47], alr[63], v[48:63] = [0, *v[1:]] */ \
  return {.r = _mm512_alignr_epi8(v.r, alr, 16 - GREX_BIT2BYTE(BITS))};

// 128 bit
#define GREX_ZUSHINGLE_64_2 GREX_ZUSHINGLE_SHIFT128
#define GREX_ZUSHINGLE_32_4 GREX_ZUSHINGLE_SHIFT128
#define GREX_ZUSHINGLE_16_8 GREX_ZUSHINGLE_SHIFT128
#define GREX_ZUSHINGLE_8_16 GREX_ZUSHINGLE_SHIFT128
// 256 bit
#if GREX_X86_64_LEVEL >= 4
// TODO On Zen 4, the AVX version is faster, whereas Tigerlake prefers this version → benchmark?
#define GREX_ZUSHINGLE_64_4 GREX_ZUSHINGLE_ALIGNR_AVX512
#define GREX_ZUSHINGLE_32_8 GREX_ZUSHINGLE_ALIGNR_AVX512
#else
#define GREX_ZUSHINGLE_64_4 GREX_ZUSHINGLE_ALIGNR_AVX
#define GREX_ZUSHINGLE_32_8 GREX_ZUSHINGLE_ALIGNR_AVX
#endif
#define GREX_ZUSHINGLE_16_16 GREX_ZUSHINGLE_ALIGNR_AVX
#define GREX_ZUSHINGLE_8_32 GREX_ZUSHINGLE_ALIGNR_AVX
// 512 bit
#define GREX_ZUSHINGLE_64_8 GREX_ZUSHINGLE_ALIGNR_AVX512
#define GREX_ZUSHINGLE_32_16 GREX_ZUSHINGLE_ALIGNR_AVX512
#define GREX_ZUSHINGLE_16_32 GREX_ZUSHINGLE_DBLALIGN
#define GREX_ZUSHINGLE_8_64 GREX_ZUSHINGLE_DBLALIGN

#define GREX_ZUSHINGLE(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  inline Vector<KIND##BITS, SIZE> shingle_up(Vector<KIND##BITS, SIZE> v) { \
    GREX_CAT(GREX_ZUSHINGLE_, BITS, _, SIZE)(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  }
#define GREX_ZUSHINGLE_ALL(REGISTERBITS, BITPREFIX) \
  GREX_FOREACH_TYPE(GREX_ZUSHINGLE, REGISTERBITS, BITPREFIX, REGISTERBITS)
GREX_FOREACH_X86_64_LEVEL(GREX_ZUSHINGLE_ALL)
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_SHINGLE_HPP
