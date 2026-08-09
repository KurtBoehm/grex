// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_SHINGLE_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_SHINGLE_HPP

#include <concepts>

#include <immintrin.h>

#include "grex/backend/base.hpp"
#include "grex/backend/choosers.hpp"
#include "grex/backend/macros/base.hpp"
#include "grex/backend/macros/for-each.hpp"
#include "grex/backend/macros/math.hpp"
#include "grex/backend/macros/types.hpp"
#include "grex/backend/x86/instruction-sets.hpp"
#include "grex/backend/x86/macros/for-each.hpp"
#include "grex/backend/x86/macros/intrinsics.hpp"
#include "grex/backend/x86/operations/expand.hpp"
#include "grex/backend/x86/operations/insert-static.hpp"
#include "grex/backend/x86/operations/intrinsics.hpp"
#include "grex/backend/x86/types.hpp"
#include "grex/base.hpp"

#if GREX_X86_64_LEVEL != 2
#include "grex/backend/x86/operations/set.hpp"
#endif

// Shingling Overview
//
// A “shingle” operation shifts all elements in a vector by one position (upwards or downwards) and
// fills the newly vacated lane either with 0 or with a scalar value:
//
//  * "Z" prefix → zero is inserted into the empty spot
//  * "V" prefix → a scalar value is inserted into the empty spot
//  * "U" suffix → upwards shingling (elements move towards higher indices)
//  * "D" suffix → downwards shingling (elements move towards lower indices)
//
// Macro naming convention:
//   GREX_{Z, V}{U, D}SHINGLE_<BITS>_<SIZE>
//   - BITS: element width (8, 16, 32, 64)
//   - SIZE: number of elements in the vector

namespace grex::backend {
//==================================================================================================
// Byte-wise shift right
//==================================================================================================

// `GREX_BSRLI` provide a unified interface for shifting the first `BITS` right by `N` bytes,
// implemented either via `_mm_bsrli_si128` or the corresponding `srli` intrinsic.

#define GREX_BSRLI_128(X, N) _mm_bsrli_si128(X, N)
#define GREX_BSRLI_64(X, N) _mm_srli_epi64(X, GREX_MULTIPLY(N, 8))
#define GREX_BSRLI_32(X, N) _mm_srli_epi32(X, GREX_MULTIPLY(N, 8))
#define GREX_BSRLI(BITS, X, N) GREX_CAT(GREX_BSRLI_, BITS)(X, N)

//--------------------------------------------------------------------------------------------------
// x86-64-v1/v2
//--------------------------------------------------------------------------------------------------

// ZU: logically shift elements up by one, which zeroes the first element.
#define GREX_ZUSHINGLE_SHIFT128(KIND, BITS, ...) \
  const __m128i ivec = GREX_KINDCAST(GREX_REGKIND(KIND, BITS), i, BITS, 128, v.r); \
  const __m128i r = _mm_bslli_si128(ivec, GREX_DIVIDE(BITS, 8)); \
  return {.r = GREX_KINDCAST(i, GREX_REGKIND(KIND, BITS), BITS, 128, r)};

// VU: perform a ZU shingle first, then insert `front` into the first lane.
#define GREX_VUSHINGLE_SHINSERT(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  return insert(shingle_up(v), index_tag<0>, front);

// ZD: logically shift elements down by one, which zeroes the last element.
#define GREX_ZDSHINGLE_SHIFT128(KIND, BITS, ...) \
  const __m128i ivec = GREX_KINDCAST(GREX_REGKIND(KIND, BITS), i, BITS, 128, v.r); \
  const __m128i r = _mm_bsrli_si128(ivec, GREX_DIVIDE(BITS, 8)); \
  return {.r = GREX_KINDCAST(i, GREX_REGKIND(KIND, BITS), BITS, 128, r)};

// VD: shift elements down by one and insert `back` into the last lane via the appropriate `insert`
// intrinsic.
#define GREX_VDSHINGLE_INSERT(KIND, BITS, SIZE, ...) \
  const __m128i ivec = GREX_KINDCAST(GREX_REGKIND(KIND, BITS), i, BITS, 128, v.registr()); \
  const NativeVector<KIND##BITS, SIZE> sh{_mm_bsrli_si128(ivec, GREX_DIVIDE(BITS, 8))}; \
  const __m128i r = insert(sh, index_tag<GREX_DECR(SIZE)>, back).r; \
  return VectorFor<KIND##BITS, SIZE>{GREX_KINDCAST(i, GREX_REGKIND(KIND, BITS), BITS, 128, r)};

//--------------------------------------------------------------------------------------------------
// x86-64-v3
//--------------------------------------------------------------------------------------------------

// 256-bit ZU: shuffle the lower 128 bits up and zero-fill the lower 128 bits, then use
// `_mm256_alignr_epi8` to compute the full shift.
#define GREX_ZUSHINGLE_ALIGNR_AVX(KIND, BITS, ...) \
  const __m256i ivec = GREX_KINDCAST(GREX_REGKIND(KIND, BITS), i, BITS, 256, v.r); \
  /* zero in the lower, v[n/2:] in the upper half */ \
  const __m128i lo = _mm256_castsi256_si128(ivec); \
  const __m256i hlo = _mm256_inserti128_si256(_mm256_setzero_si256(), lo, 1); \
  /* shift v up by one element within each half, shifting in 0 in the lower \
   * and v[n/2-1] in the upper half */ \
  const __m256i alignr = _mm256_alignr_epi8(ivec, hlo, 16 - GREX_DIVIDE(BITS, 8)); \
  return {.r = GREX_KINDCAST(i, GREX_REGKIND(KIND, BITS), BITS, 256, alignr)};

// 256-bit VU, 64-bit elements: same approach as below, but with `_mm256_shuffle_pd` instead of
// `_mm256_alignr_epi8`.
#define GREX_VUSHINGLE_SHUFFLEPD_AVX(KIND, BITS, SIZE, ...) \
  const __m256d dvec = GREX_KINDCAST(KIND, f, 64, 256, v.r); \
  const auto xval128 = broadcast(front, type_tag<NativeVector<KIND##BITS, SIZE / 2>>).r; \
  /* [front, front, 0, 0] */ \
  const __m256d xval = _mm256_zextpd128_pd256(GREX_KINDCAST(KIND, f, BITS, 128, xval128)); \
  /* [front, front, v[0], v[1]] */ \
  const __m256d ins = _mm256_insertf128_pd(xval, _mm256_castpd256_pd128(dvec), 1); \
  /* [ins[1], dvec[0], ins[3], dvec[2]] = [front, v[0], v[1], v[2]] */ \
  return {.r = GREX_KINDCAST(f, KIND, 64, 256, _mm256_shuffle_pd(ins, dvec, 0b0101))};

// 256-bit VU, smaller elements: construct helper with broadcast `front` in the lower and the lower
// half of `v` in the upper half, then use `_mm256_alignr_epi8` to compute the full shift.
#define GREX_VUSHINGLE_ALIGNR_AVX(KIND, BITS, SIZE, ...) \
  const __m256i ivec = GREX_KINDCAST(GREX_REGKIND(KIND, BITS), i, BITS, 256, v.r); \
  const auto xval128 = broadcast(front, type_tag<NativeVector<KIND##BITS, SIZE / 2>>).r; \
  const auto cxval = GREX_KINDCAST(GREX_REGKIND(KIND, BITS), i, BITS, 128, xval128); \
  const __m256i xval = _mm256_zextsi128_si256(cxval); \
  /* the broadcast value in the lower, v[n/2:] in the upper half */ \
  const __m256i mix = _mm256_inserti128_si256(xval, _mm256_castsi256_si128(ivec), 1); \
  const __m256i alignr = _mm256_alignr_epi8(ivec, mix, 16 - GREX_DIVIDE(BITS, 8)); \
  return {.r = GREX_KINDCAST(i, GREX_REGKIND(KIND, BITS), BITS, 256, alignr)};

// 256-bit ZD: shuffle the upper 128 bits down and zero-extend to 256 bits, then use
// `_mm256_alignr_epi8` to compute the full shift.
#define GREX_ZDSHINGLE_ALIGNR_AVX(KIND, BITS, ...) \
  const __m256i ivec = GREX_KINDCAST(GREX_REGKIND(KIND, BITS), i, BITS, 256, v.r); \
  /* v[:n/2] in the lower, zero in the upper half */ \
  const __m256i zhi = _mm256_zextsi128_si256(_mm256_extracti128_si256(ivec, 1)); \
  /* shift v down by one element within each 128-bit lane, shifting in 0 in the upper \
   * and v[n/2] in the lower half */ \
  const __m256i alignr = _mm256_alignr_epi8(zhi, ivec, GREX_DIVIDE(BITS, 8)); \
  return {.r = GREX_KINDCAST(i, GREX_REGKIND(KIND, BITS), BITS, 256, alignr)};

// 256-bit VD, 64-bit elements: same approach as below, but with `_mm256_shuffle_pd` instead of
// `_mm256_alignr_epi8`.
#define GREX_VDSHINGLE_SHUFFLEPD_AVX(KIND, BITS, SIZE, ...) \
  const __m256d dvec = GREX_KINDCAST(KIND, f, 64, 256, v.r); \
  const auto xval128 = broadcast(back, type_tag<NativeVector<KIND##BITS, SIZE / 2>>).r; \
  /* [back, back, ?, ?] */ \
  const __m256d xval = _mm256_castpd128_pd256(GREX_KINDCAST(KIND, f, BITS, 128, xval128)); \
  /* [v[2], v[3], back, back] */ \
  const __m256d perm = _mm256_permute2f128_pd(dvec, xval, 0x21); \
  /* [dvec[1], perm[0], dvec[3], perm[2]] = [v[1], v[2], v[3], back] */ \
  return {.r = GREX_KINDCAST(f, KIND, 64, 256, _mm256_shuffle_pd(dvec, perm, 0b0101))};

// 256-bit VD, smaller elements: construct helper with the upper half of `v` in the lower and the
// broadcast `back` in the upper half, then use `_mm256_alignr_epi8` to compute the full shift.
#define GREX_VDSHINGLE_ALIGNR_AVX(KIND, BITS, SIZE, ...) \
  const __m256i ivec = GREX_KINDCAST(GREX_REGKIND(KIND, BITS), i, BITS, 256, v.r); \
  const auto xval128 = broadcast(back, type_tag<NativeVector<KIND##BITS, SIZE / 2>>).r; \
  const auto cxval = GREX_KINDCAST(GREX_REGKIND(KIND, BITS), i, BITS, 128, xval128); \
  const __m256i xval256 = _mm256_castsi128_si256(cxval); \
  /* v[n/2:] in the lower, the broadcast value in the upper half */ \
  const __m256i perm = _mm256_permute2x128_si256(ivec, xval256, 0x21); \
  const __m256i alignr = _mm256_alignr_epi8(perm, ivec, GREX_DIVIDE(BITS, 8)); \
  return {.r = GREX_KINDCAST(i, GREX_REGKIND(KIND, BITS), BITS, 256, alignr)};

//--------------------------------------------------------------------------------------------------
// x86-64-v4
//--------------------------------------------------------------------------------------------------

// AVX-512 ZU, 32/64-bit elements: use cross-lane `alignr` with zeros.
#define GREX_ZUSHINGLE_ALIGNR_AVX512(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  const auto ivec = GREX_KINDCAST(KIND, i, BITS, REGISTERBITS, v.r); \
  const auto zero = BITPREFIX##_setzero_si##REGISTERBITS(); \
  const auto alignr = BITPREFIX##_alignr_epi##BITS(ivec, zero, GREX_DECR(SIZE)); \
  return {.r = GREX_KINDCAST(i, KIND, BITS, REGISTERBITS, alignr)};

// AVX-512 VU, 32/64-bit elements: cross-lane `alignr` with broadcast `front`.
#define GREX_VUSHINGLE_ALIGNR_AVX512(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  const auto ivec = GREX_KINDCAST(KIND, i, BITS, REGISTERBITS, v.r); \
  const auto xval = GREX_KINDCAST(KIND, i, BITS, REGISTERBITS, \
                                  broadcast(front, type_tag<NativeVector<KIND##BITS, SIZE>>).r); \
  const auto alignr = BITPREFIX##_alignr_epi##BITS(ivec, xval, GREX_DECR(SIZE)); \
  return {.r = GREX_KINDCAST(i, KIND, BITS, REGISTERBITS, alignr)};

// AVX-512 ZD, 32/64-bit elements: use cross-lane `alignr` with zeros.
#define GREX_ZDSHINGLE_ALIGNR_AVX512(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  const auto ivec = GREX_KINDCAST(KIND, i, BITS, REGISTERBITS, v.r); \
  const auto zero = BITPREFIX##_setzero_si##REGISTERBITS(); \
  const auto alignr = BITPREFIX##_alignr_epi##BITS(zero, ivec, 1); \
  return {.r = GREX_KINDCAST(i, KIND, BITS, REGISTERBITS, alignr)};

// AVX-512 VD, 32/64-bit elements: cross-lane `alignr` with broadcast `front`.
#define GREX_VDSHINGLE_ALIGNR_AVX512(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  const auto ivec = GREX_KINDCAST(KIND, i, BITS, REGISTERBITS, v.r); \
  const auto xval = GREX_KINDCAST(KIND, i, BITS, REGISTERBITS, \
                                  broadcast(back, type_tag<NativeVector<KIND##BITS, SIZE>>).r); \
  const auto alignr = BITPREFIX##_alignr_epi##BITS(xval, ivec, 1); \
  return {.r = GREX_KINDCAST(i, KIND, BITS, REGISTERBITS, alignr)};

// AVX-512 with 8- or 16-bit values: analogous to the AVX implementation with _mm512_alignr_epi64
// instead of _mm256_inserti128_si256 to shuffle 128-bit lanes up/down by one element.
//
// The comments below assume 8-bit integers; the steps for 16 bits are analogous.

#define GREX_ZUSHINGLE_DBLALIGN(KIND, BITS, ...) \
  /* [0]*16 + v[:48] → alr[i] = 0 if i < 16 else v[i-16] */ \
  const __m512i alr = _mm512_alignr_epi64(v.r, _mm512_setzero_si512(), 6); \
  /* [alr[15], *v[:15], alr[31], *v[16:31], alr[47], v[32:47], alr[63], v[48:63]] = [0, *v[1:]] */ \
  return {.r = _mm512_alignr_epi8(v.r, alr, 16 - GREX_DIVIDE(BITS, 8))};

#define GREX_VUSHINGLE_DBLALIGN(KIND, BITS, SIZE, ...) \
  const __m512i xval = broadcast(front, type_tag<NativeVector<KIND##BITS, SIZE>>).r; \
  const __m512i alr = _mm512_alignr_epi64(v.r, xval, 6); \
  return {.r = _mm512_alignr_epi8(v.r, alr, 16 - GREX_DIVIDE(BITS, 8))};

#define GREX_ZDSHINGLE_DBLALIGN(KIND, BITS, ...) \
  /* v[16:] + [0]*16 → alr[i] = v[i+16] if i < 48 else 0 */ \
  const __m512i alr = _mm512_alignr_epi64(_mm512_setzero_si512(), v.r, 2); \
  /* [alr[15], *v[:15], alr[31], *v[16:31], alr[47], v[32:47], alr[63], v[48:63]] = [0, *v[1:]] */ \
  return {.r = _mm512_alignr_epi8(alr, v.r, GREX_DIVIDE(BITS, 8))};

#define GREX_VDSHINGLE_DBLALIGN(KIND, BITS, SIZE, ...) \
  const __m512i xval = broadcast(back, type_tag<NativeVector<KIND##BITS, SIZE>>).r; \
  const __m512i alr = _mm512_alignr_epi64(xval, v.r, 2); \
  return {.r = _mm512_alignr_epi8(alr, v.r, GREX_DIVIDE(BITS, 8))};

//==================================================================================================
// ZU: up with zero front
//==================================================================================================

// 128-bit vectors
#define GREX_ZUSHINGLE_64_2 GREX_ZUSHINGLE_SHIFT128
#define GREX_ZUSHINGLE_32_4 GREX_ZUSHINGLE_SHIFT128
#define GREX_ZUSHINGLE_16_8 GREX_ZUSHINGLE_SHIFT128
#define GREX_ZUSHINGLE_8_16 GREX_ZUSHINGLE_SHIFT128

// 256-bit vectors
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

// 512-bit vectors
#define GREX_ZUSHINGLE_64_8 GREX_ZUSHINGLE_ALIGNR_AVX512
#define GREX_ZUSHINGLE_32_16 GREX_ZUSHINGLE_ALIGNR_AVX512
#define GREX_ZUSHINGLE_16_32 GREX_ZUSHINGLE_DBLALIGN
#define GREX_ZUSHINGLE_8_64 GREX_ZUSHINGLE_DBLALIGN

//==================================================================================================
// VU: up with scalar front
//==================================================================================================

//--------------------------------------------------------------------------------------------------
// 128 bits
//--------------------------------------------------------------------------------------------------

// 2×64-bit: build [front, v[0]] via unpack.
#define GREX_VUSHINGLE_64_2(KIND, ...) \
  const __m128i xval = GREX_KINDCAST(KIND, i, 64, 128, expand_any(front, index_tag<2>).r); \
  const __m128i ivec = GREX_KINDCAST(KIND, i, 64, 128, v.r); \
  return {.r = GREX_KINDCAST(i, KIND, 64, 128, _mm_unpacklo_epi64(xval, ivec))};

// Other 128-bit widths use ZU shingle followed by `insert`.
#define GREX_VUSHINGLE_32_4 GREX_VUSHINGLE_SHINSERT
#define GREX_VUSHINGLE_16_8 GREX_VUSHINGLE_SHINSERT
#define GREX_VUSHINGLE_8_16 GREX_VUSHINGLE_SHINSERT

//--------------------------------------------------------------------------------------------------
// 256 bits
//--------------------------------------------------------------------------------------------------

// On Zen 4, GREX_VUSHINGLE_ALIGNR_AVX is faster than GREX_VUSHINGLE_ALIGNR_AVX512,
// and both are equally fast on Tigerlake.
#define GREX_VUSHINGLE_64_4 GREX_VUSHINGLE_SHUFFLEPD_AVX
#define GREX_VUSHINGLE_32_8 GREX_VUSHINGLE_ALIGNR_AVX
#define GREX_VUSHINGLE_16_16 GREX_VUSHINGLE_ALIGNR_AVX
#define GREX_VUSHINGLE_8_32 GREX_VUSHINGLE_ALIGNR_AVX

//--------------------------------------------------------------------------------------------------
// 512 bits
//--------------------------------------------------------------------------------------------------

#define GREX_VUSHINGLE_64_8 GREX_VUSHINGLE_ALIGNR_AVX512
#define GREX_VUSHINGLE_32_16 GREX_VUSHINGLE_ALIGNR_AVX512
#define GREX_VUSHINGLE_16_32 GREX_VUSHINGLE_DBLALIGN
#define GREX_VUSHINGLE_8_64 GREX_VUSHINGLE_DBLALIGN

//==================================================================================================
// ZD: down with zero front
//==================================================================================================

//--------------------------------------------------------------------------------------------------
// 128 bits
//--------------------------------------------------------------------------------------------------

#define GREX_ZDSHINGLE_64_2 GREX_ZDSHINGLE_SHIFT128
#define GREX_ZDSHINGLE_32_4 GREX_ZDSHINGLE_SHIFT128
#define GREX_ZDSHINGLE_16_8 GREX_ZDSHINGLE_SHIFT128
#define GREX_ZDSHINGLE_8_16 GREX_ZDSHINGLE_SHIFT128

//--------------------------------------------------------------------------------------------------
// 256 bits
//--------------------------------------------------------------------------------------------------

#if GREX_X86_64_LEVEL >= 4
#define GREX_ZDSHINGLE_64_4 GREX_ZDSHINGLE_ALIGNR_AVX512
#define GREX_ZDSHINGLE_32_8 GREX_ZDSHINGLE_ALIGNR_AVX512
#else
#define GREX_ZDSHINGLE_64_4 GREX_ZDSHINGLE_ALIGNR_AVX
#define GREX_ZDSHINGLE_32_8 GREX_ZDSHINGLE_ALIGNR_AVX
#endif
#define GREX_ZDSHINGLE_16_16 GREX_ZDSHINGLE_ALIGNR_AVX
#define GREX_ZDSHINGLE_8_32 GREX_ZDSHINGLE_ALIGNR_AVX

//--------------------------------------------------------------------------------------------------
// 512 bits
//--------------------------------------------------------------------------------------------------

#define GREX_ZDSHINGLE_64_8 GREX_ZDSHINGLE_ALIGNR_AVX512
#define GREX_ZDSHINGLE_32_16 GREX_ZDSHINGLE_ALIGNR_AVX512
#define GREX_ZDSHINGLE_16_32 GREX_ZDSHINGLE_DBLALIGN
#define GREX_ZDSHINGLE_8_64 GREX_ZDSHINGLE_DBLALIGN

//==================================================================================================
// VD: down with scalar front
//==================================================================================================

//--------------------------------------------------------------------------------------------------
// Sub-native
//--------------------------------------------------------------------------------------------------

// 2×32-bit: `movd` if integer, move `back` into the upper 64 bits, and shuffle everything down by
// one lane.
#define GREX_VDSHINGLE_32_2_BASE(KIND, ...) \
  const __m128i ivec = GREX_KINDCAST(KIND, i, 32, 128, v.registr()); \
  const auto xval = expand_any(back, index_tag<4>).r; \
  const __m128i aval = GREX_KINDCAST(KIND, i, 32, 128, xval); \
  const __m128i unpk = _mm_unpacklo_epi64(ivec, aval); \
  const __m128i shif = _mm_shuffle_epi32(unpk, 0b11'11'10'01); \
  return SubVector<KIND##32, 2>{GREX_KINDCAST(i, KIND, 32, 128, shif)};

// 4×32-bit integer: shuffle everything down with `pshufd` and insert into the last lane with
// `pinsrd`.
#define GREX_VDSHINGLE_32_INSERT(KIND, BITS, SIZE, ...) \
  const __m128i ivec = GREX_KINDCAST(KIND, i, 32, 128, v.registr()); \
  const __m128i sh = _mm_shuffle_epi32(ivec, 0b11'11'10'01); \
  const __m128i in = _mm_insert_epi32(sh, GREX_SIGNED_CAST(KIND, 32, back), GREX_DECR(SIZE)); \
  return VectorFor<KIND##BITS, SIZE>{GREX_KINDCAST(i, KIND, 32, 128, in)};

// 4×f32: `alignr` by 4 bytes with `v` at the bottom and `back` at the top.
#define GREX_VDSHINGLE_32_4_ALIGNR(KIND, BITS, SIZE, ...) \
  const __m128i ivec = GREX_KINDCAST(KIND, i, 32, 128, v.registr()); \
  const auto xval = expand_any(back, index_tag<4>).r; \
  const __m128i aval = GREX_KINDCAST(KIND, i, 32, 128, xval); \
  const __m128i algn = _mm_alignr_epi8(aval, ivec, 4); \
  return VectorFor<KIND##BITS, SIZE>{GREX_KINDCAST(i, KIND, 32, 128, algn)};

#if GREX_X86_64_LEVEL >= 2
// Choose type-specific 2×32-bit implementation on level 2+.
#define GREX_VDSHINGLE_f32_2 GREX_VDSHINGLE_32_2_BASE
#define GREX_VDSHINGLE_i32_2 GREX_VDSHINGLE_32_INSERT
#define GREX_VDSHINGLE_u32_2 GREX_VDSHINGLE_32_INSERT
#define GREX_VDSHINGLE_32_2(KIND, ...) GREX_VDSHINGLE_##KIND##32_2(KIND, __VA_ARGS__)
#else
#define GREX_VDSHINGLE_32_2 GREX_VDSHINGLE_32_2_BASE
#endif

// 16-bit sub-vector: shift right and insert `back` into last active lane.
#define GREX_VDSHINGLE_16_SUB(KIND, BITS, PART, ...) \
  const SubVector<KIND##16, PART> shif{GREX_BSRLI(GREX_MULTIPLY(BITS, PART), v.registr(), 2)}; \
  return insert(shif, index_tag<GREX_DECR(PART)>, back);

#define GREX_VDSHINGLE_16_4 GREX_VDSHINGLE_16_SUB
#define GREX_VDSHINGLE_16_2 GREX_VDSHINGLE_16_SUB

#if GREX_X86_64_LEVEL >= 2
// 8-bit sub-vector: shift by one byte and `pinsrb` for `back`.
#define GREX_VDSHINGLE_8_SUB(KIND, BITS, PART, ...) \
  const __m128i sh = GREX_CAT(_mm_srli_epi, GREX_MULTIPLY(BITS, PART))(v.registr(), 8); \
  return SubVector<KIND##8, PART>{mm::insert_epi8(sh, back, int_tag<GREX_DECR(PART)>)};
#else
// Fallback on level 1: insert `back` into the first inactive lane and shift down by one byte.
#define GREX_VDSHINGLE_8_SUB(KIND, BITS, PART, ...) \
  const i16 back16 = expand_bits<i16>(back); \
  const __m128i ins = mm::insert_epi16(v.registr(), back16, int_tag<GREX_DIVIDE(PART, 2)>); \
  return SubVector<KIND##8, PART>{GREX_BSRLI(GREX_MULTIPLY(GREX_MULTIPLY(BITS, PART), 2), ins, 1)};
#endif

#define GREX_VDSHINGLE_8_2 GREX_VDSHINGLE_8_SUB
#define GREX_VDSHINGLE_8_4 GREX_VDSHINGLE_8_SUB
#define GREX_VDSHINGLE_8_8 GREX_VDSHINGLE_8_SUB

//--------------------------------------------------------------------------------------------------
// 128 bits
//--------------------------------------------------------------------------------------------------

// 4×32-bit: use two `shuffle_ps` to construct [v[1], v[2], v[3], back].
#define GREX_VDSHINGLE_32_4_BASE(KIND, ...) \
  const __m128 xval = GREX_KINDCAST(KIND, f, 32, 128, expand_any(back, index_tag<4>).r); \
  const __m128 fvec = GREX_KINDCAST(KIND, f, 32, 128, v.r); \
  /* [fvec[2], fvec[3], xval[0], xval[1]] = [v[2], v[3], back, 0] */ \
  const __m128 shuf = _mm_shuffle_ps(fvec, xval, 0b01'00'11'10); \
  /* [fvec[1], fvec[2], shuf[1], shuf[2]] = [v[1], v[2], v[3], back] */ \
  const __m128 merged = _mm_shuffle_ps(fvec, shuf, 0b10'01'10'01); \
  return {.r = GREX_KINDCAST(f, KIND, 32, 128, merged)};

#if GREX_X86_64_LEVEL == 1
// 2×64-bit on level 1: broadcast `back` and `punpckhqdq`.
#define GREX_VDSHINGLE_64_2(KIND, ...) \
  const auto xval = broadcast(back, type_tag<NativeVector<KIND##64, 2>>).r; \
  const __m128i ival = GREX_KINDCAST(KIND, i, 64, 128, xval); \
  const __m128i ivec = GREX_KINDCAST(KIND, i, 64, 128, v.r); \
  return {.r = GREX_KINDCAST(i, KIND, 64, 128, _mm_unpackhi_epi64(ivec, ival))};

#define GREX_VDSHINGLE_32_4 GREX_VDSHINGLE_32_4_BASE
#define GREX_VDSHINGLE_16_8 GREX_VDSHINGLE_INSERT

// 16×8-bit: shift `back` to the last lane, shift `v` down by one line, and merge using bitwise OR.
#define GREX_VDSHINGLE_8_16(KIND, ...) \
  const __m128i xval = GREX_KINDCAST(KIND, i, 8, 128, expand_any(back, index_tag<16>).r); \
  const __m128i shval = _mm_bslli_si128(xval, 15); \
  const __m128i ivec = GREX_KINDCAST(KIND, i, 8, 128, v.r); \
  const __m128i shvec = _mm_bsrli_si128(ivec, 1); \
  return {.r = GREX_KINDCAST(i, KIND, BITS, 128, _mm_or_si128(shvec, shval))};
#else
// 2×f64: simple `shufpd` with expanded `back`.
#define GREX_VDSHINGLE_f64_2(...) \
  return {.r = _mm_shuffle_pd(v.r, expand_any(back, index_tag<2>).r, 0b0'1)};

// 2×64-bit integers: shuffle the upper half of `v` into the lower half with `pshufd` and insert
// `back` into the upper half.
#define GREX_VDSHINGLE_i64_2(KIND, ...) \
  const __m128i sh = _mm_shuffle_epi32(v.r, 0b11'10'11'10); \
  return {.r = _mm_insert_epi64(sh, GREX_SIGNED_CAST(KIND, 64, back), 1)};

#define GREX_VDSHINGLE_u64_2 GREX_VDSHINGLE_i64_2
#define GREX_VDSHINGLE_64_2(KIND, ...) GREX_VDSHINGLE_##KIND##64_2(KIND, __VA_ARGS__)

// 4×32-bit dispatch.
#define GREX_VDSHINGLE_i32_4 GREX_VDSHINGLE_32_INSERT
#define GREX_VDSHINGLE_u32_4 GREX_VDSHINGLE_i32_4
#define GREX_VDSHINGLE_f32_4 GREX_VDSHINGLE_32_4_ALIGNR
#define GREX_VDSHINGLE_32_4(KIND, ...) GREX_VDSHINGLE_##KIND##32_4(KIND, __VA_ARGS__)

// 8×16-bit, 16×8-bit via generic INSERT-based implementation.
#define GREX_VDSHINGLE_16_8 GREX_VDSHINGLE_INSERT
#define GREX_VDSHINGLE_8_16 GREX_VDSHINGLE_INSERT
#endif

//--------------------------------------------------------------------------------------------------
// 256 bits
//--------------------------------------------------------------------------------------------------

// Zen 4 prefers the AVX-512 variant by 2 cycles, Tigerlake the AVX variant by 1 cycle.
#if GREX_X86_64_LEVEL >= 4
#define GREX_VDSHINGLE_64_4 GREX_VDSHINGLE_ALIGNR_AVX512
#define GREX_VDSHINGLE_32_8 GREX_VDSHINGLE_ALIGNR_AVX512
#else
#define GREX_VDSHINGLE_64_4 GREX_VDSHINGLE_SHUFFLEPD_AVX
#define GREX_VDSHINGLE_32_8 GREX_VDSHINGLE_ALIGNR_AVX
#endif
#define GREX_VDSHINGLE_16_16 GREX_VDSHINGLE_ALIGNR_AVX
#define GREX_VDSHINGLE_8_32 GREX_VDSHINGLE_ALIGNR_AVX

//--------------------------------------------------------------------------------------------------
// 512 bits
//--------------------------------------------------------------------------------------------------

#define GREX_VDSHINGLE_64_8 GREX_VDSHINGLE_ALIGNR_AVX512
#define GREX_VDSHINGLE_32_16 GREX_VDSHINGLE_ALIGNR_AVX512
#define GREX_VDSHINGLE_16_32 GREX_VDSHINGLE_DBLALIGN
#define GREX_VDSHINGLE_8_64 GREX_VDSHINGLE_DBLALIGN

//==================================================================================================
// Shingle generators
//==================================================================================================

#define GREX_SHINGLE(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  GREX_ALWAYS_INLINE inline NativeVector<KIND##BITS, SIZE> shingle_up( \
    NativeVector<KIND##BITS, SIZE> v) { \
    GREX_CAT(GREX_ZUSHINGLE_, BITS, _, SIZE)(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS); \
  } \
  template<std::same_as<KIND##BITS> T> \
  GREX_ALWAYS_INLINE inline NativeVector<T, SIZE> shingle_up(T front, NativeVector<T, SIZE> v) { \
    GREX_CAT(GREX_VUSHINGLE_, BITS, _, SIZE)(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS); \
  } \
  GREX_ALWAYS_INLINE inline NativeVector<KIND##BITS, SIZE> shingle_down( \
    NativeVector<KIND##BITS, SIZE> v) { \
    GREX_CAT(GREX_ZDSHINGLE_, BITS, _, SIZE)(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS); \
  } \
  template<std::same_as<KIND##BITS> T> \
  GREX_ALWAYS_INLINE inline NativeVector<T, SIZE> shingle_down(NativeVector<T, SIZE> v, T back) { \
    GREX_CAT(GREX_VDSHINGLE_, BITS, _, SIZE)(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS); \
  }
#define GREX_SHINGLE_ALL(REGISTERBITS, BITPREFIX) \
  GREX_FOREACH_TYPE_EXT(GREX_SHINGLE, REGISTERBITS, BITPREFIX, REGISTERBITS)
GREX_FOREACH_X86_64_LEVEL(GREX_SHINGLE_ALL)

#define GREX_SHINGLE_SUB_I(KIND, BITS, PART, SIZE, BITPREFIX, REGISTERBITS, RKIND) \
  GREX_ALWAYS_INLINE inline SubVector<KIND##BITS, PART> shingle_up( \
    SubVector<KIND##BITS, PART> v) { \
    const __m128i iv = GREX_KINDCAST(RKIND, i, BITS, REGISTERBITS, v.registr()); \
    const auto slli = GREX_CAT(_mm_slli_epi, GREX_MULTIPLY(BITS, PART))(iv, BITS); \
    return SubVector<KIND##BITS, PART>{GREX_KINDCAST(i, RKIND, BITS, REGISTERBITS, slli)}; \
  } \
  template<std::same_as<KIND##BITS> T> \
  GREX_ALWAYS_INLINE inline SubVector<T, PART> shingle_up(T front, SubVector<T, PART> v) { \
    return SubVector<T, PART>{insert(shingle_up(v).full, index_tag<0>, front)}; \
  } \
  GREX_ALWAYS_INLINE inline SubVector<KIND##BITS, PART> shingle_down( \
    SubVector<KIND##BITS, PART> v) { \
    const __m128i iv = GREX_KINDCAST(RKIND, i, BITS, REGISTERBITS, v.registr()); \
    const auto srli = GREX_CAT(_mm_srli_epi, GREX_MULTIPLY(BITS, PART))(iv, BITS); \
    return SubVector<KIND##BITS, PART>{GREX_KINDCAST(i, RKIND, BITS, REGISTERBITS, srli)}; \
  } \
  template<std::same_as<KIND##BITS> T> \
  GREX_ALWAYS_INLINE inline SubVector<T, PART> shingle_down(SubVector<T, PART> v, T back) { \
    GREX_CAT(GREX_VDSHINGLE_, BITS, _, PART)(KIND, BITS, PART, SIZE, BITPREFIX, REGISTERBITS) \
  }
#define GREX_SHINGLE_SUB(KIND, BITS, PART, SIZE, BITPREFIX, REGISTERBITS) \
  GREX_SHINGLE_SUB_I(KIND, BITS, PART, SIZE, BITPREFIX, REGISTERBITS, GREX_REGKIND(KIND, BITS))
GREX_FOREACH_SUB_EXT(GREX_SHINGLE_SUB, _mm, 128)
} // namespace grex::backend

#include "grex/backend/shared/operations/shingle.hpp" // IWYU pragma: export

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_SHINGLE_HPP
