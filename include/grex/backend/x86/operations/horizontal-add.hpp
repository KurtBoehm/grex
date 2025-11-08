// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_HORIZONTAL_ADD_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_HORIZONTAL_ADD_HPP

#include <immintrin.h>

#include "grex/backend/defs.hpp"
#include "grex/backend/x86/instruction-sets.hpp"
#include "grex/backend/x86/macros/for-each.hpp"
#include "grex/backend/x86/macros/intrinsics.hpp"
#include "grex/backend/x86/operations/arithmetic.hpp"
#include "grex/backend/x86/types.hpp"
#include "grex/base/defs.hpp" // IWYU pragma: keep

#if GREX_X86_64_LEVEL >= 3
#include "grex/backend/x86/operations/split.hpp"
#endif

namespace grex::backend {
// Baseline recursive definition for 256 and 512 bits
#define GREX_HADD_HALVES(...) \
  return horizontal_add(add(split(v, index_tag<0>), split(v, index_tag<1>)));
// f32
#define GREX_HADD_f32x2(...) \
  /* [v1, -, -, -] */ \
  const __m128 shuf = _mm_shuffle_ps(vf.r, vf.r, 1); \
  /* [v0 + v1, -, -, -][0] */ \
  return _mm_cvtss_f32(_mm_add_ss(vf.r, shuf));
#define GREX_HADD_f32x4(...) \
  /* [v0 + v2, v1 + v3, -, -] */ \
  const __m128 pairs = _mm_add_ps(v.r, _mm_movehl_ps(v.r, v.r)); \
  /* [v1 + v3, -, -, -] */ \
  const __m128 shuf = _mm_shuffle_ps(pairs, pairs, 1); \
  /* [v0 + v2 + v1 + v3, -, -, -][0] */ \
  return _mm_cvtss_f32(_mm_add_ss(pairs, shuf));
#define GREX_HADD_f32x8 GREX_HADD_HALVES
#define GREX_HADD_f32x16 GREX_HADD_HALVES
// f64
#define GREX_HADD_f64x2(...) \
  /* [v1, v0] */ \
  const __m128d rev = _mm_unpackhi_pd(v.r, v.r); \
  /* [v0 + v1, -][0] */ \
  return _mm_cvtsd_f64(_mm_add_sd(v.r, rev));
#define GREX_HADD_f64x4 GREX_HADD_HALVES
#define GREX_HADD_f64x8 GREX_HADD_HALVES
// i8/u8
#define GREX_HADD_i8_SUB(PART, KIND, BITS, ...) \
  /* mask out all components above PART */ \
  const __m128i masked = _mm_and_si128(vf.r, _mm_set1_epi64x((1ULL << (BITS * PART)) - 1)); \
  /* [v0 + … + v7, -, -, -, -, -, -, -] as u16x8 */ \
  const __m128i sad = _mm_sad_epu8(masked, _mm_setzero_si128()); \
  /* extract low 32 bits and cast the upper 24 bits away */ \
  return KIND##BITS(_mm_cvtsi128_si32(sad));
#define GREX_HADD_i8x2(...) GREX_HADD_i8_SUB(2, __VA_ARGS__)
#define GREX_HADD_i8x4(...) GREX_HADD_i8_SUB(4, __VA_ARGS__)
#define GREX_HADD_i8x8(KIND, BITS, ...) \
  /* [v0 + … + v7, -, -, -, -, -, -, -] as u16x8 */ \
  const __m128i sad = _mm_sad_epu8(vf.r, _mm_setzero_si128()); \
  /* extract low 32 bits and cast the upper 24 bits away */ \
  return KIND##BITS(_mm_cvtsi128_si32(sad));
#define GREX_HADD_i8x16(KIND, BITS, ...) \
  /* [v0 + … + v7, -, -, -, v8 + … + v15, -, -, -] as u16x8 */ \
  const __m128i sad = _mm_sad_epu8(v.r, _mm_setzero_si128()); \
  /* [sad1, -, -, -, -, -, -, -] as u16x8 */ \
  const __m128i unpackhi = _mm_unpackhi_epi64(sad, sad); \
  /* [v0 + … + v15, -, -, -, -, -, -, -] as u16x8 */ \
  const __m128i add = _mm_add_epi16(sad, unpackhi); \
  /* extract low 32 bits and cast the upper 24 bits away */ \
  return KIND##BITS(_mm_cvtsi128_si32(add));
#define GREX_HADD_i8x32 GREX_HADD_HALVES
#define GREX_HADD_i8x64 GREX_HADD_HALVES
// i16/u16
#define GREX_HADD_i16x2(KIND, BITS, ...) \
  /* [v1, -, -, -, -, -, -, -] */ \
  const __m128i shuf = _mm_shufflelo_epi16(vf.r, 1); \
  /* [v0 + v1, -, -, -, -, -, -, -][0] */ \
  return KIND##BITS(_mm_cvtsi128_si32(_mm_add_epi16(vf.r, shuf)));
#define GREX_HADD_i16x4(KIND, BITS, ...) \
  /* [v2, v3, -, -, -, -, -, -] */ \
  const __m128i shuf1 = _mm_shuffle_epi32(vf.r, 1); \
  /* [v0 + v2, v1 + v3, -, -, -, -, -, -] */ \
  const __m128i pairs = _mm_add_epi16(vf.r, shuf1); \
  /* [v1 + v3, -, -, -, -, -, -, -] */ \
  const __m128i shuf2 = _mm_shufflelo_epi16(pairs, 1); \
  /* [v1 + … + v3, -, -, -, -, -, -, -][0] */ \
  return KIND##BITS(_mm_cvtsi128_si32(_mm_add_epi16(pairs, shuf2)));
#define GREX_HADD_i16x8(KIND, BITS, ...) \
  /* [v4, v5, v6, v7, v4, v5, v6, v7] */ \
  const __m128i unpackhi = _mm_unpackhi_epi64(v.r, v.r); \
  /* [v0 + v4, v1 + v5, v2 + v6, v3 + v7, -, -, -, -] */ \
  const __m128i pairs = _mm_add_epi16(v.r, unpackhi); \
  /* [v2 + v6, v3 + v7, -, -, -, -, -, -] */ \
  const __m128i spairs = _mm_shuffle_epi32(pairs, 1); \
  /* [v0 + v4 + v2 + v6, v1 + v5 + v3 + v7, -, -, -, -, -, -] */ \
  const __m128i quads = _mm_add_epi16(pairs, spairs); \
  /* [v1 + v5 + v3 + v7, -, -, -, -, -, -, -] */ \
  const __m128i squads = _mm_shufflelo_epi16(quads, 1); \
  /* [v1 + … + v7, -, -, -, -, -, -, -][0] */ \
  return KIND##BITS(_mm_cvtsi128_si32(_mm_add_epi16(quads, squads)));
#define GREX_HADD_i16x16 GREX_HADD_HALVES
#define GREX_HADD_i16x32 GREX_HADD_HALVES
// i32/u32
#define GREX_HADD_i32x2(KIND, BITS, ...) \
  /* [v1, -, -, -] */ \
  const __m128i shuf = _mm_shuffle_epi32(vf.r, 1); \
  /* [v0 + v1, -, -, -][0] */ \
  return GREX_KINDCAST_SINGLE(i, KIND, BITS, _mm_cvtsi128_si32(_mm_add_epi32(vf.r, shuf)));
#define GREX_HADD_i32x4(KIND, BITS, ...) \
  /* [v2, v3, v2, v3] */ \
  const __m128i unpackhi = _mm_unpackhi_epi64(v.r, v.r); \
  /* [v0 + v2, v1 + v3, -, -] */ \
  const __m128i pairs = _mm_add_epi32(v.r, unpackhi); \
  /* [v1 + v3, -, -, -] */ \
  const __m128i shuf = _mm_shuffle_epi32(pairs, 1); \
  /* [v0 + v2 + v1 + v3, -, -, -][0] */ \
  return GREX_KINDCAST_SINGLE(i, KIND, BITS, _mm_cvtsi128_si32(_mm_add_epi32(pairs, shuf)));
#define GREX_HADD_i32x8 GREX_HADD_HALVES
#define GREX_HADD_i32x16 GREX_HADD_HALVES
// i64/u64
#define GREX_HADD_i64x2(KIND, BITS, ...) \
  /* [v1, v1] */ \
  const __m128i unpack = _mm_unpackhi_epi64(v.r, v.r); \
  /* [v0 + v1, v1 + v1][0] */ \
  return GREX_KINDCAST_SINGLE(i, KIND, BITS, _mm_cvtsi128_si64(_mm_add_epi64(v.r, unpack)));
#define GREX_HADD_i64x4 GREX_HADD_HALVES
#define GREX_HADD_i64x8 GREX_HADD_HALVES
// Conversion wrappers
#define GREX_HADD_f(KIND, BITS, SIZE, ...) GREX_HADD_f##BITS##x##SIZE(KIND, BITS, SIZE, __VA_ARGS__)
#define GREX_HADD_i(KIND, BITS, SIZE, ...) GREX_HADD_i##BITS##x##SIZE(KIND, BITS, SIZE, __VA_ARGS__)
#define GREX_HADD_u(KIND, BITS, SIZE, ...) GREX_HADD_i##BITS##x##SIZE(KIND, BITS, SIZE, __VA_ARGS__)

// Wrapper macros
#define GREX_HADD(KIND, BITS, SIZE) \
  inline KIND##BITS horizontal_add(Vector<KIND##BITS, SIZE> v) { \
    GREX_HADD_##KIND(KIND, BITS, SIZE) \
  }
#define GREX_HADD_SUB(KIND, BITS, PART, SIZE) \
  inline KIND##BITS horizontal_add(SubVector<KIND##BITS, PART, SIZE> v) { \
    const auto vf = v.full; \
    GREX_HADD_##KIND(KIND, BITS, PART) \
  }

#define GREX_HADD_ALL(REGISTERBITS, BITPREFIX) GREX_FOREACH_TYPE(GREX_HADD, REGISTERBITS)
GREX_FOREACH_X86_64_LEVEL(GREX_HADD_ALL)

// SubVector
GREX_FOREACH_SUB(GREX_HADD_SUB)

// SuperVector
template<typename THalf>
inline THalf::Value horizontal_add(SuperVector<THalf> v) {
  return horizontal_add(add(v.lower, v.upper));
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_HORIZONTAL_ADD_HPP
