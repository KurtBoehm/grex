// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_SHINGLE_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_SHINGLE_HPP

#include <cstddef>

#include <immintrin.h>

#include "grex/backend/defs.hpp"
#include "grex/backend/x86/instruction-sets.hpp"
#include "grex/backend/x86/macros/base.hpp"
#include "grex/backend/x86/macros/for-each.hpp"
#include "grex/backend/x86/macros/intrinsics.hpp"
#include "grex/backend/x86/macros/math.hpp"
#include "grex/backend/x86/operations/expand-scalar.hpp"
#include "grex/backend/x86/operations/extract.hpp"
#include "grex/backend/x86/operations/set.hpp"
#include "grex/backend/x86/types.hpp"
#include "grex/base/defs.hpp"

// “Z” means a “zero” is put into the empty spot, “V” means a value is put there
// “U” means “upwards” shingling, “D” “downwards” shingling

namespace grex::backend {
// 128 bits: just perform a shift
// the value is added through zero extension and a bit-wise OR
#define GREX_ZUSHINGLE_SHIFT128(KIND, BITS, ...) \
  const __m128i ivec = GREX_KINDCAST(KIND, i, BITS, 128, v.r); \
  return {.r = GREX_KINDCAST(i, KIND, BITS, 128, _mm_bslli_si128(ivec, GREX_DIVIDE(BITS, 8)))};
#define GREX_VUSHINGLE_OR(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  const auto xval = expand_zero(front, index_tag<SIZE>).r; \
  const auto zval = GREX_KINDCAST(KIND, i, BITS, REGISTERBITS, xval); \
  const auto zush = GREX_KINDCAST(KIND, i, BITS, REGISTERBITS, shingle_up(v).r); \
  const auto merged = BITPREFIX##_or_si##REGISTERBITS(zval, zush); \
  return {.r = GREX_KINDCAST(i, KIND, BITS, REGISTERBITS, merged)};
#define GREX_ZDSHINGLE_SHIFT128(KIND, BITS, ...) \
  const __m128i ivec = GREX_KINDCAST(KIND, i, BITS, 128, v.r); \
  return {.r = GREX_KINDCAST(i, KIND, BITS, 128, _mm_bsrli_si128(ivec, GREX_DIVIDE(BITS, 8)))};

// shingling downwards with value, 128 bits at level 2: using shift and insert
#define GREX_VDSHINGLE_INSERT(KIND, BITS, SIZE, ...) \
  const __m128i ivec = GREX_KINDCAST(KIND, i, BITS, 128, v.r); \
  const __m128i sh = _mm_bsrli_si128(ivec, GREX_DIVIDE(BITS, 8)); \
  return {.r = GREX_KINDCAST(i, KIND, BITS, 128, \
                             _mm_insert_epi##BITS(sh, back.value, GREX_DECR(SIZE)))};

// 256-bit with AVX: shuffle the lower into the upper 128 bits and use alignr
#define GREX_ZUSHINGLE_ALIGNR_AVX(KIND, BITS, ...) \
  const __m256i ivec = GREX_KINDCAST(KIND, i, BITS, 256, v.r); \
  /* zero in the lower, v[n/2:] in the upper half */ \
  const __m128i lo = _mm256_castsi256_si128(ivec); \
  const __m256i hlo = _mm256_inserti128_si256(_mm256_setzero_si256(), lo, 1); \
  /* shift v up by one element in each half, shifting in 0 in the lower \
   * and v[n/2-1] in the upper half */ \
  const __m256i alignr = _mm256_alignr_epi8(ivec, hlo, 16 - GREX_DIVIDE(BITS, 8)); \
  return {.r = GREX_KINDCAST(i, KIND, BITS, 256, alignr)};
#define GREX_VUSHINGLE_ALIGNR_AVX(KIND, BITS, SIZE, ...) \
  const __m256i ivec = GREX_KINDCAST(KIND, i, BITS, 256, v.r); \
  const auto xval128 = broadcast(front.value, type_tag<Vector<KIND##BITS, SIZE / 2>>).r; \
  const __m256i xval = _mm256_zextsi128_si256(GREX_KINDCAST(KIND, i, BITS, 128, xval128)); \
  /* the broadcast value in the lower, v[n/2:] in the upper half */ \
  const __m256i mix = _mm256_inserti128_si256(xval, _mm256_castsi256_si128(ivec), 1); \
  const __m256i alignr = _mm256_alignr_epi8(ivec, mix, 16 - GREX_DIVIDE(BITS, 8)); \
  return {.r = GREX_KINDCAST(i, KIND, BITS, 256, alignr)};
#define GREX_ZDSHINGLE_ALIGNR_AVX(KIND, BITS, ...) \
  const __m256i ivec = GREX_KINDCAST(KIND, i, BITS, 256, v.r); \
  /* v[:n/2] in the lower, zero in the upper half */ \
  const __m256i zhi = _mm256_zextsi128_si256(_mm256_extracti128_si256(ivec, 1)); \
  /* shift v down by one element in each half, shifting in 0 in the upper \
   * and v[n/2] in the lower half */ \
  const __m256i alignr = _mm256_alignr_epi8(zhi, ivec, GREX_DIVIDE(BITS, 8)); \
  return {.r = GREX_KINDCAST(i, KIND, BITS, 256, alignr)};
#define GREX_VDSHINGLE_ALIGNR_AVX(KIND, BITS, SIZE, ...) \
  const __m256i ivec = GREX_KINDCAST(KIND, i, BITS, 256, v.r); \
  const auto xval128 = broadcast(back.value, type_tag<Vector<KIND##BITS, SIZE / 2>>).r; \
  const auto xval256 = _mm256_castsi128_si256(GREX_KINDCAST(KIND, i, BITS, 128, xval128)); \
  /* v[n/2:] in the lower, the broadcast value in the upper half */ \
  const __m256i shin = _mm256_permute2x128_si256(ivec, xval256, 0x21); \
  const __m256i alignr = _mm256_alignr_epi8(shin, ivec, GREX_DIVIDE(BITS, 8)); \
  return {.r = GREX_KINDCAST(i, KIND, BITS, 256, alignr)};

// AVX-512 with 32- or 64-bit values: Use alignr intrinsics
#define GREX_ZUSHINGLE_ALIGNR_AVX512(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  const auto ivec = GREX_KINDCAST(KIND, i, BITS, REGISTERBITS, v.r); \
  const auto zero = BITPREFIX##_setzero_si##REGISTERBITS(); \
  const auto alignr = BITPREFIX##_alignr_epi##BITS(ivec, zero, GREX_DECR(SIZE)); \
  return {.r = GREX_KINDCAST(i, KIND, BITS, REGISTERBITS, alignr)};
#define GREX_VUSHINGLE_ALIGNR_AVX512(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  const auto ivec = GREX_KINDCAST(KIND, i, BITS, REGISTERBITS, v.r); \
  const auto xval = GREX_KINDCAST(KIND, i, BITS, REGISTERBITS, \
                                  broadcast(front.value, type_tag<Vector<KIND##BITS, SIZE>>).r); \
  const auto alignr = BITPREFIX##_alignr_epi##BITS(ivec, xval, GREX_DECR(SIZE)); \
  return {.r = GREX_KINDCAST(i, KIND, BITS, REGISTERBITS, alignr)};
#define GREX_ZDSHINGLE_ALIGNR_AVX512(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  const auto ivec = GREX_KINDCAST(KIND, i, BITS, REGISTERBITS, v.r); \
  const auto zero = BITPREFIX##_setzero_si##REGISTERBITS(); \
  const auto alignr = BITPREFIX##_alignr_epi##BITS(zero, ivec, 1); \
  return {.r = GREX_KINDCAST(i, KIND, BITS, REGISTERBITS, alignr)};
#define GREX_VDSHINGLE_ALIGNR_AVX512(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  const auto ivec = GREX_KINDCAST(KIND, i, BITS, REGISTERBITS, v.r); \
  const auto xval = GREX_KINDCAST(KIND, i, BITS, REGISTERBITS, \
                                  broadcast(back.value, type_tag<Vector<KIND##BITS, SIZE>>).r); \
  const auto alignr = BITPREFIX##_alignr_epi##BITS(xval, ivec, 1); \
  return {.r = GREX_KINDCAST(i, KIND, BITS, REGISTERBITS, alignr)};

// AVX-512 with 8- or 16-bit values: Analogous to the AVX implementation with _mm512_alignr_epi64
// instead of _mm256_inserti128_si256 to shuffle 128-bit lanes up/down by one
#define GREX_ZUSHINGLE_DBLALIGN(KIND, BITS, ...) \
  /* the comments assume 8-bit integers, but the steps are completely analogous for 16 bits */ \
  /* [0]*16 + v[:48] → alr[i] = 0 if i < 16 else v[i-16] */ \
  const __m512i alr = _mm512_alignr_epi64(v.r, _mm512_setzero_si512(), 6); \
  /* [alr[15], *v[:15], alr[31], *v[16:31], alr[47], v[32:47], alr[63], v[48:63] = [0, *v[1:]] */ \
  return {.r = _mm512_alignr_epi8(v.r, alr, 16 - GREX_DIVIDE(BITS, 8))};
#define GREX_VUSHINGLE_DBLALIGN(KIND, BITS, SIZE, ...) \
  const __m512i xval = broadcast(front.value, type_tag<Vector<KIND##BITS, SIZE>>).r; \
  const __m512i alr = _mm512_alignr_epi64(v.r, xval, 6); \
  return {.r = _mm512_alignr_epi8(v.r, alr, 16 - GREX_DIVIDE(BITS, 8))};
#define GREX_ZDSHINGLE_DBLALIGN(KIND, BITS, ...) \
  /* the comments assume 8-bit integers, but the steps are completely analogous for 16 bits */ \
  /* v[16:] + [0]*16 → alr[i] = v[i+16] if i < 48 else 0 */ \
  const __m512i alr = _mm512_alignr_epi64(_mm512_setzero_si512(), v.r, 2); \
  /* [alr[15], *v[:15], alr[31], *v[16:31], alr[47], v[32:47], alr[63], v[48:63] = [0, *v[1:]] */ \
  return {.r = _mm512_alignr_epi8(alr, v.r, GREX_DIVIDE(BITS, 8))};
#define GREX_VDSHINGLE_DBLALIGN(KIND, BITS, SIZE, ...) \
  const __m512i xval = broadcast(back.value, type_tag<Vector<KIND##BITS, SIZE>>).r; \
  const __m512i alr = _mm512_alignr_epi64(xval, v.r, 2); \
  return {.r = _mm512_alignr_epi8(alr, v.r, GREX_DIVIDE(BITS, 8))};

// upwards shingle with a zero
// 128 bit
#define GREX_ZUSHINGLE_64_2 GREX_ZUSHINGLE_SHIFT128
#define GREX_ZUSHINGLE_32_4 GREX_ZUSHINGLE_SHIFT128
#define GREX_ZUSHINGLE_16_8 GREX_ZUSHINGLE_SHIFT128
#define GREX_ZUSHINGLE_8_16 GREX_ZUSHINGLE_SHIFT128
// 256 bit
#if GREX_X86_64_LEVEL >= 4
// TODO On Zen 4, the AVX version is faster, whereas Tigerlake prefers this version
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

// upwards shingle with a given value
// 128 bit
#define GREX_VUSHINGLE_64_2(KIND, ...) \
  const __m128i xval = GREX_KINDCAST(KIND, i, 64, 128, expand_any(front, index_tag<2>).r); \
  const __m128i ivec = GREX_KINDCAST(KIND, i, 64, 128, v.r); \
  return {.r = GREX_KINDCAST(i, KIND, 64, 128, _mm_unpacklo_epi64(xval, ivec))};
#define GREX_VUSHINGLE_32_4(KIND, ...) \
  const __m128 xval = GREX_KINDCAST(KIND, f, 32, 128, expand_any(front, index_tag<4>).r); \
  const __m128 fvec = GREX_KINDCAST(KIND, f, 32, 128, v.r); \
  const __m128 merged = _mm_shuffle_ps(_mm_movelh_ps(xval, fvec), fvec, 0b10'01'10'00); \
  return {.r = GREX_KINDCAST(f, KIND, 32, 128, merged)};
#define GREX_VUSHINGLE_16_8 GREX_VUSHINGLE_OR
#define GREX_VUSHINGLE_8_16 GREX_VUSHINGLE_OR
// 256 bit
// TODO On Zen 4, the AVX version is faster, and both are equally fast on Tigerlake
// #define GREX_VUSHINGLE_64_4 GREX_VUSHINGLE_ALIGNR_AVX512
// #define GREX_VUSHINGLE_32_8 GREX_VUSHINGLE_ALIGNR_AVX512
#define GREX_VUSHINGLE_64_4 GREX_VUSHINGLE_ALIGNR_AVX
#define GREX_VUSHINGLE_32_8 GREX_VUSHINGLE_ALIGNR_AVX
#define GREX_VUSHINGLE_16_16 GREX_VUSHINGLE_ALIGNR_AVX
#define GREX_VUSHINGLE_8_32 GREX_VUSHINGLE_ALIGNR_AVX
// 512 bit
#define GREX_VUSHINGLE_64_8 GREX_VUSHINGLE_ALIGNR_AVX512
#define GREX_VUSHINGLE_32_16 GREX_VUSHINGLE_ALIGNR_AVX512
#define GREX_VUSHINGLE_16_32 GREX_VUSHINGLE_DBLALIGN
#define GREX_VUSHINGLE_8_64 GREX_VUSHINGLE_DBLALIGN

// downwards shingling with a zero
// sub-native
#define GREX_ZDSHINGLE_32_2(KIND, ...) \
  const __m128i ivec = GREX_KINDCAST(KIND, i, 32, 128, v.registr()); \
  const __m128i shuf = _mm_shuffle_epi32(ivec, 0b01'01'01'01); \
  const __m128 mvss = _mm_move_ss(_mm_setzero_ps(), _mm_castsi128_ps(shuf)); \
  return SubVector<KIND##32, 2, 4>{GREX_KINDCAST(f, KIND, 32, 128, mvss)};
#define GREX_ZDSHINGLE_16_4(KIND, ...) \
  /* v[:4] * 2 */ \
  const __m128i shuf = _mm_shuffle_epi32(v.registr(), 0b01'00'01'00); \
  /* [v[1], v[2], v[3]] + [0] * 5 */ \
  return SubVector<KIND##16, 4, 8>{_mm_bsrli_si128(shuf, 10)};
#define GREX_ZDSHINGLE_16_2(KIND, ...) \
  /* [v[0], v[1]] * 4 */ \
  const __m128i shuf = _mm_shuffle_epi32(v.registr(), 0); \
  /* [v[1]] + [0] * 7 */ \
  return SubVector<KIND##16, 2, 8>{_mm_bsrli_si128(shuf, 14)};
#define GREX_ZDSHINGLE_8_2(KIND, ...) \
  /* [0]*15 + [v[1]] */ \
  const __m128i shuf = _mm_bslli_si128(v.registr(), 14); \
  /* [v[1]] + [0] * 15 */ \
  return SubVector<KIND##8, 2, 16>{_mm_bsrli_si128(shuf, 15)};
#define GREX_ZDSHINGLE_8_4(KIND, ...) \
  /* v[:4] * 4 */ \
  const __m128i shuf = _mm_shuffle_epi32(v.registr(), 0); \
  /* v[1:4] + [0] * 13 */ \
  return SubVector<KIND##8, 4, 16>{_mm_bsrli_si128(shuf, 13)};
#define GREX_ZDSHINGLE_8_8(KIND, ...) \
  /* v[:8] * 2 */ \
  const __m128i shuf = _mm_shuffle_epi32(v.registr(), 0b01'00'01'00); \
  /* v[1:8] + [0] * 9 */ \
  return SubVector<KIND##8, 8, 16>{_mm_bsrli_si128(shuf, 9)};
// 128 bit
#define GREX_ZDSHINGLE_64_2 GREX_ZDSHINGLE_SHIFT128
#define GREX_ZDSHINGLE_32_4 GREX_ZDSHINGLE_SHIFT128
#define GREX_ZDSHINGLE_16_8 GREX_ZDSHINGLE_SHIFT128
#define GREX_ZDSHINGLE_8_16 GREX_ZDSHINGLE_SHIFT128
// 256 bit
#if GREX_X86_64_LEVEL >= 4
// TODO On Zen 4, the AVX version is faster, whereas Tigerlake prefers this version
#define GREX_ZDSHINGLE_64_4 GREX_ZDSHINGLE_ALIGNR_AVX512
#define GREX_ZDSHINGLE_32_8 GREX_ZDSHINGLE_ALIGNR_AVX512
#else
#define GREX_ZDSHINGLE_64_4 GREX_ZDSHINGLE_ALIGNR_AVX
#define GREX_ZDSHINGLE_32_8 GREX_ZDSHINGLE_ALIGNR_AVX
#endif
#define GREX_ZDSHINGLE_16_16 GREX_ZDSHINGLE_ALIGNR_AVX
#define GREX_ZDSHINGLE_8_32 GREX_ZDSHINGLE_ALIGNR_AVX
// 512 bit
#define GREX_ZDSHINGLE_64_8 GREX_ZDSHINGLE_ALIGNR_AVX512
#define GREX_ZDSHINGLE_32_16 GREX_ZDSHINGLE_ALIGNR_AVX512
#define GREX_ZDSHINGLE_16_32 GREX_ZDSHINGLE_DBLALIGN
#define GREX_ZDSHINGLE_8_64 GREX_ZDSHINGLE_DBLALIGN

// downwards shingling with a value
// sub-native
#define GREX_VDSHINGLE_32_2(KIND, ...) \
  const __m128i ivec = GREX_KINDCAST(KIND, i, 32, 128, v.registr()); \
  const auto xval = expand_any(back, index_tag<4>).r; \
  const __m128i aval = GREX_KINDCAST(KIND, i, 32, 128, xval); \
  const __m128i unpk = _mm_unpacklo_epi64(ivec, aval); \
  const __m128i shif = _mm_shuffle_epi32(unpk, 0b11'11'10'01); \
  return SubVector<KIND##32, 2, 4>{GREX_KINDCAST(i, KIND, 32, 128, shif)};
#define GREX_VDSHINGLE_16_SUB(KIND, BITS, PART, ...) \
  const __m128i shif = _mm_bsrli_si128(v.registr(), 2); \
  return SubVector<KIND##16, PART, 8>{_mm_insert_epi16(shif, back.value, GREX_DECR(PART))};
#define GREX_VDSHINGLE_16_4 GREX_VDSHINGLE_16_SUB
#define GREX_VDSHINGLE_16_2 GREX_VDSHINGLE_16_SUB
#if GREX_X86_64_LEVEL >= 2
#define GREX_VDSHINGLE_8_SUB(KIND, BITS, PART, ...) \
  const __m128i shif = _mm_bsrli_si128(v.registr(), 1); \
  return SubVector<KIND##8, PART, 16>{_mm_insert_epi8(shif, back.value, GREX_DECR(PART))};
#else
#define GREX_VDSHINGLE_8_SUB(KIND, BITS, PART, ...) \
  const __m128i ins = _mm_insert_epi16(v.registr(), back.value, PART / 2); \
  return SubVector<KIND##8, PART, 16>{_mm_bsrli_si128(ins, 1)};
#endif
#define GREX_VDSHINGLE_8_2 GREX_VDSHINGLE_8_SUB
#define GREX_VDSHINGLE_8_4 GREX_VDSHINGLE_8_SUB
#define GREX_VDSHINGLE_8_8 GREX_VDSHINGLE_8_SUB
// 128 bit
#define GREX_VDSHINGLE_32_4_BASE(KIND, ...) \
  const __m128 xval = GREX_KINDCAST(KIND, f, 32, 128, expand_any(back, index_tag<4>).r); \
  const __m128 fvec = GREX_KINDCAST(KIND, f, 32, 128, v.r); \
  /* [fvec[2], fvec[3], xval[0], xval[1]] = [v[2], v[3], back, 0] */ \
  const __m128 shuf = _mm_shuffle_ps(fvec, xval, 0b01'00'11'10); \
  /* [fvec[1], fvec[2], shuf[1], shuf[2]] = [v[1], v[2], v[3], back] */ \
  const __m128 merged = _mm_shuffle_ps(fvec, shuf, 0b10'01'10'01); \
  return {.r = GREX_KINDCAST(f, KIND, 32, 128, merged)};
#if GREX_X86_64_LEVEL == 1
#define GREX_VDSHINGLE_64_2(KIND, ...) \
  const auto xval = broadcast(back.value, type_tag<Vector<KIND##64, 2>>).r; \
  const __m128i ival = GREX_KINDCAST(KIND, i, 64, 128, xval); \
  const __m128i ivec = GREX_KINDCAST(KIND, i, 64, 128, v.r); \
  return {.r = GREX_KINDCAST(i, KIND, 64, 128, _mm_unpackhi_epi64(ivec, ival))};
#define GREX_VDSHINGLE_32_4 GREX_VDSHINGLE_32_4_BASE
#define GREX_VDSHINGLE_16_8 GREX_VDSHINGLE_INSERT
#define GREX_VDSHINGLE_8_16(KIND, ...) \
  const __m128i xval = GREX_KINDCAST(KIND, i, 8, 128, expand_any(back, index_tag<16>).r); \
  const __m128i shval = _mm_bslli_si128(xval, 15); \
  const __m128i ivec = GREX_KINDCAST(KIND, i, 8, 128, v.r); \
  const __m128i shvec = _mm_bsrli_si128(ivec, 1); \
  return {.r = GREX_KINDCAST(i, KIND, BITS, 128, _mm_or_si128(shvec, shval))};
#else
#define GREX_VDSHINGLE_f64_2(...) \
  return {.r = _mm_shuffle_pd(v.r, expand_any(back, index_tag<2>).r, 0b0'1)};
#define GREX_VDSHINGLE_i64_2(KIND, ...) \
  const __m128i sh = _mm_shuffle_epi32(v.r, 0b11'10'11'10); \
  return {.r = _mm_insert_epi64(sh, GREX_SIGNED_CAST(KIND, 64, back.value), 1)};
#define GREX_VDSHINGLE_u64_2 GREX_VDSHINGLE_i64_2
#define GREX_VDSHINGLE_64_2(KIND, ...) GREX_VDSHINGLE_##KIND##64_2(KIND, __VA_ARGS__)
#define GREX_VDSHINGLE_i32_4(KIND, ...) \
  const __m128i ivec = GREX_KINDCAST(KIND, i, 32, 128, v.r); \
  const __m128i sh = _mm_shuffle_epi32(ivec, 0b11'11'10'01); \
  return {.r = GREX_KINDCAST(i, KIND, 32, 128, \
                             _mm_insert_epi32(sh, GREX_SIGNED_CAST(KIND, 32, back.value), 3))};
#define GREX_VDSHINGLE_u32_4 GREX_VDSHINGLE_i32_4
#define GREX_VDSHINGLE_f32_4 GREX_VDSHINGLE_32_4_BASE
#define GREX_VDSHINGLE_32_4(KIND, ...) GREX_VDSHINGLE_##KIND##32_4(KIND, __VA_ARGS__)
#define GREX_VDSHINGLE_16_8 GREX_VDSHINGLE_INSERT
#define GREX_VDSHINGLE_8_16 GREX_VDSHINGLE_INSERT
#endif
// 256 bit
// TODO Zen 4 likes the AVX-512 and the AVX variants the same, Tigerlake prefers the AVX variant
// #define GREX_VDSHINGLE_64_4 GREX_VDSHINGLE_ALIGNR_AVX512
// #define GREX_VDSHINGLE_32_8 GREX_VDSHINGLE_ALIGNR_AVX512
#define GREX_VDSHINGLE_64_4 GREX_VDSHINGLE_ALIGNR_AVX
#define GREX_VDSHINGLE_32_8 GREX_VDSHINGLE_ALIGNR_AVX
#define GREX_VDSHINGLE_16_16 GREX_VDSHINGLE_ALIGNR_AVX
#define GREX_VDSHINGLE_8_32 GREX_VDSHINGLE_ALIGNR_AVX
// 512 bit
#define GREX_VDSHINGLE_64_8 GREX_VDSHINGLE_ALIGNR_AVX512
#define GREX_VDSHINGLE_32_16 GREX_VDSHINGLE_ALIGNR_AVX512
#define GREX_VDSHINGLE_16_32 GREX_VDSHINGLE_DBLALIGN
#define GREX_VDSHINGLE_8_64 GREX_VDSHINGLE_DBLALIGN

#define GREX_SHINGLE(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  inline Vector<KIND##BITS, SIZE> shingle_up(Vector<KIND##BITS, SIZE> v) { \
    GREX_CAT(GREX_ZUSHINGLE_, BITS, _, SIZE)(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  } \
  inline Vector<KIND##BITS, SIZE> shingle_up(Scalar<KIND##BITS> front, \
                                             Vector<KIND##BITS, SIZE> v) { \
    GREX_CAT(GREX_VUSHINGLE_, BITS, _, SIZE)(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  } \
  inline Vector<KIND##BITS, SIZE> shingle_down(Vector<KIND##BITS, SIZE> v) { \
    GREX_CAT(GREX_ZDSHINGLE_, BITS, _, SIZE)(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  } \
  inline Vector<KIND##BITS, SIZE> shingle_down(Vector<KIND##BITS, SIZE> v, \
                                               Scalar<KIND##BITS> back) { \
    GREX_CAT(GREX_VDSHINGLE_, BITS, _, SIZE)(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  }
#define GREX_SHINGLE_ALL(REGISTERBITS, BITPREFIX) \
  GREX_FOREACH_TYPE(GREX_SHINGLE, REGISTERBITS, BITPREFIX, REGISTERBITS)
GREX_FOREACH_X86_64_LEVEL(GREX_SHINGLE_ALL)

#define GREX_SHINGLE_SUB(KIND, BITS, PART, SIZE, BITPREFIX, REGISTERBITS) \
  inline SubVector<KIND##BITS, PART, SIZE> shingle_down(SubVector<KIND##BITS, PART, SIZE> v) { \
    GREX_CAT(GREX_ZDSHINGLE_, BITS, _, PART)(KIND, BITS, PART, SIZE, BITPREFIX, REGISTERBITS) \
  } \
  inline SubVector<KIND##BITS, PART, SIZE> shingle_down(SubVector<KIND##BITS, PART, SIZE> v, \
                                                        Scalar<KIND##BITS> back) { \
    GREX_CAT(GREX_VDSHINGLE_, BITS, _, PART)(KIND, BITS, PART, SIZE, BITPREFIX, REGISTERBITS) \
  }
GREX_SHINGLE_SUB(f, 32, 2, 4, _mm, 128)
GREX_SHINGLE_SUB(i, 32, 2, 4, _mm, 128)
GREX_SHINGLE_SUB(u, 32, 2, 4, _mm, 128)
GREX_SHINGLE_SUB(i, 16, 4, 8, _mm, 128)
GREX_SHINGLE_SUB(u, 16, 4, 8, _mm, 128)
GREX_SHINGLE_SUB(i, 16, 2, 8, _mm, 128)
GREX_SHINGLE_SUB(u, 16, 2, 8, _mm, 128)
GREX_SHINGLE_SUB(i, 8, 8, 16, _mm, 128)
GREX_SHINGLE_SUB(u, 8, 8, 16, _mm, 128)
GREX_SHINGLE_SUB(i, 8, 4, 16, _mm, 128)
GREX_SHINGLE_SUB(u, 8, 4, 16, _mm, 128)
GREX_SHINGLE_SUB(i, 8, 2, 16, _mm, 128)
GREX_SHINGLE_SUB(u, 8, 2, 16, _mm, 128)

// sub-native vectors: just use the native version
template<typename T, std::size_t tPart, std::size_t tSize>
inline SubVector<T, tPart, tSize> shingle_up(SubVector<T, tPart, tSize> v) {
  return SubVector<T, tPart, tSize>{shingle_up(v.full)};
}
template<typename T, std::size_t tPart, std::size_t tSize>
inline SubVector<T, tPart, tSize> shingle_up(Scalar<T> front, SubVector<T, tPart, tSize> v) {
  return SubVector<T, tPart, tSize>{shingle_up(front, v.full)};
}

// super-native vectors: carry over the last element from the lower part to the upper part
template<typename THalf>
inline SuperVector<THalf> shingle_up(SuperVector<THalf> v) {
  return {
    .lower = shingle_up(v.lower),
    .upper = shingle_up(Scalar{extract(v.lower, THalf::size - 1)}, v.upper),
  };
}
template<typename THalf>
inline SuperVector<THalf> shingle_up(Scalar<typename THalf::Value> front, SuperVector<THalf> v) {
  return {
    .lower = shingle_up(front, v.lower),
    .upper = shingle_up(Scalar{extract(v.lower, THalf::size - 1)}, v.upper),
  };
}
template<typename THalf>
inline SuperVector<THalf> shingle_down(SuperVector<THalf> v) {
  return {
    .lower = shingle_down(v.lower, Scalar{extract(v.upper, 0)}),
    .upper = shingle_down(v.upper),
  };
}
template<typename THalf>
inline SuperVector<THalf> shingle_down(SuperVector<THalf> v, Scalar<typename THalf::Value> back) {
  return {
    .lower = shingle_down(v.lower, Scalar{extract(v.upper, 0)}),
    .upper = shingle_down(v.upper, back),
  };
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_SHINGLE_HPP
