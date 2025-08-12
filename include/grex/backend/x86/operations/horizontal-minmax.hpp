// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_HORIZONTAL_MINMAX_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_HORIZONTAL_MINMAX_HPP

#include <immintrin.h>

#include "grex/backend/defs.hpp"
#include "grex/backend/x86/instruction-sets.hpp"
#include "grex/backend/x86/macros/for-each.hpp"
#include "grex/backend/x86/macros/intrinsics.hpp"
#include "grex/backend/x86/operations/minmax.hpp"
#include "grex/backend/x86/types.hpp"
#include "grex/base/defs.hpp" // IWYU pragma: keep

#if GREX_X86_64_LEVEL >= 3
#include "grex/backend/x86/operations/split.hpp"
#endif

namespace grex::backend {
// Baseline recursive definition for 256 and 512 bits
#define GREX_HMINMAX_HALVES(OP, ...) \
  return horizontal_##OP(OP(split(v, index_tag<0>), split(v, index_tag<1>)));
// f32
#define GREX_HMINMAX_f32x2(OP, ...) \
  /* [v1, -, -, -] */ \
  const __m128 shuf = _mm_shuffle_ps(vf.r, vf.r, 1); \
  /* [op(v0, v1), -, -, -][0] */ \
  return _mm_cvtss_f32(_mm_##OP##_ss(vf.r, shuf));
#define GREX_HMINMAX_f32x4(OP, ...) \
  /* [op(v0, v2), op(v1, v3), -, -] */ \
  const __m128 pairs = _mm_##OP##_ps(v.r, _mm_movehl_ps(v.r, v.r)); \
  /* [op(v1, v3), -, -, -] */ \
  const __m128 shuf = _mm_shuffle_ps(pairs, pairs, 1); \
  /* [op(v0, v2, v1, v3), -, -, -][0] */ \
  return _mm_cvtss_f32(_mm_##OP##_ss(pairs, shuf));
#define GREX_HMINMAX_f32x8 GREX_HMINMAX_HALVES
#define GREX_HMINMAX_f32x16 GREX_HMINMAX_HALVES
// f64
#define GREX_HMINMAX_f64x2(OP, ...) \
  /* [v1, v0] */ \
  const __m128d rev = _mm_unpackhi_pd(v.r, v.r); \
  /* [op(v0, v1), -][0] */ \
  return _mm_cvtsd_f64(_mm_##OP##_sd(v.r, rev));
#define GREX_HMINMAX_f64x4 GREX_HMINMAX_HALVES
#define GREX_HMINMAX_f64x8 GREX_HMINMAX_HALVES
// i8/u8
#define GREX_HMINMAX_i8x2(OP, KIND, BITS, ...) \
  /* [v1, -, …, -] */ \
  const __m128i srli = _mm_srli_epi16(vf.r, 8); \
  /* [op(v0, v1), -, …, -] */ \
  const auto full = OP(vf, {.r = srli}); \
  /* extract low 32 bits and cast the upper 24 bits away */ \
  return KIND##BITS(_mm_cvtsi128_si32(full.r));
#define GREX_HMINMAX_i8x4(OP, KIND, BITS, ...) \
  /* [v2, v3, -, …, -] */ \
  const __m128i srli16 = _mm_srli_epi32(vf.r, 16); \
  /* [op(v0, v2), op(v1, v3), -, …, -] */ \
  const auto pairs = OP(vf, {.r = srli16}); \
  /* [op(v1, v3), -, …, -] */ \
  const __m128i srli8 = _mm_srli_epi16(pairs.r, 8); \
  /* [op(v0, v2, v1, v3), -, …, -] */ \
  const auto full = OP(pairs, {.r = srli8}); \
  /* extract low 32 bits and cast the upper 24 bits away */ \
  return KIND##BITS(_mm_cvtsi128_si32(full.r));
#define GREX_HMINMAX_i8x8(OP, KIND, BITS, ...) \
  /* [v4, …, v7, -, …, -] */ \
  const __m128i srli32 = _mm_srli_epi64(vf.r, 32); \
  /* [op(v0, v4), …, op(v3, v7), -, …, -] */ \
  const auto pairs = OP(vf, {.r = srli32}); \
  /* [op(v2, v6), op(v3, v7), -, …, -] */ \
  const __m128i srli16 = _mm_srli_epi32(pairs.r, 16); \
  /* [op(v0, v4, v2, v6), op(v1, v5, v3, v7), -, …, -] */ \
  const auto quads = OP(pairs, {.r = srli16}); \
  /* [op(v1, v5, v3, v7), -, …, -] */ \
  const __m128i srli8 = _mm_srli_epi16(quads.r, 8); \
  /* [op(v0, v4, v2, v6, v1, v5, v3, v7), -, …, -] */ \
  const auto full = OP(quads, {.r = srli8}); \
  /* extract low 32 bits and cast the upper 24 bits away */ \
  return KIND##BITS(_mm_cvtsi128_si32(full.r));
#define GREX_HMINMAX_i8x16(OP, KIND, BITS, ...) \
  /* [v8, …, v15, -, …, -] */ \
  const __m128i srli64 = _mm_srli_si128(v.r, 8); \
  /* [op(v0, v8), …, op(v7, v15), -, …, -] */ \
  const auto pairs = OP(v, {.r = srli64}); \
  /* [op(v4, v12), …, op(v7, v15), -, …, -] */ \
  const __m128i srli32 = _mm_srli_epi64(pairs.r, 32); \
  /* [op(v0, v8, v4, v12), …, op(v3, v11, v7, v15), -, …, -] */ \
  const auto quads = OP(pairs, {.r = srli32}); \
  /* [op(v2, v10, v6, v14), op(v3, v11, v7, v15), -, …, -] */ \
  const __m128i srli16 = _mm_srli_epi32(quads.r, 16); \
  /* [op(v0, v8, v4, v12, v2, v10, v6, v14), op(v1, v9, v5, v13, v3, v11, v7, v15), -, …, -] */ \
  const auto octs = OP(quads, {.r = srli16}); \
  /* [op(v1, v9, v5, v13, v3, v11, v7, v15), -, …, -] */ \
  const __m128i srli8 = _mm_srli_epi16(octs.r, 8); \
  /* [op(v0, v8, v4, v12, v2, v10, v6, v14, v1, v9, v5, v13, v3, v11, v7, v15), -, …, -] */ \
  const auto full = OP(octs, {.r = srli8}); \
  /* extract low 32 bits and cast the upper 24 bits away */ \
  return KIND##BITS(_mm_cvtsi128_si32(full.r));
#define GREX_HMINMAX_i8x32 GREX_HMINMAX_HALVES
#define GREX_HMINMAX_i8x64 GREX_HMINMAX_HALVES
// i16/u16
#define GREX_HMINMAX_i16x2(OP, KIND, BITS, ...) \
  /* [v1, -, -, -, -, -, -, -] */ \
  const __m128i srli16 = _mm_srli_epi32(vf.r, 16); \
  /* [op(v0, v1), -, -, -, -, -, -, -][0] */ \
  return KIND##BITS(_mm_cvtsi128_si32(OP(vf, {.r = srli16}).r));
#define GREX_HMINMAX_i16x4(OP, KIND, BITS, ...) \
  /* [v2, v3, -, -, -, -, -, -] */ \
  const __m128i srli32 = _mm_srli_epi64(vf.r, 32); \
  /* [op(v0, v2), op(v1, v3), -, -, -, -, -, -] */ \
  const auto pairs = OP(vf, {.r = srli32}); \
  /* [op(v1, v3), -, -, -, -, -, -, -] */ \
  const __m128i srli16 = _mm_srli_epi32(pairs.r, 16); \
  /* [op(v1, …, v7), -, -, -, -, -, -, -][0] */ \
  return KIND##BITS(_mm_cvtsi128_si32(OP(pairs, {.r = srli16}).r));
#define GREX_HMINMAX_i16x8(OP, KIND, BITS, ...) \
  /* [v4, v5, v6, v7, -, -, -, -] */ \
  const __m128i srli64 = _mm_srli_si128(v.r, 8); \
  /* [op(v0, v4), op(v1, v5), op(v2, v6), op(v3, v7), -, -, -, -] */ \
  const auto pairs = OP(v, {.r = srli64}); \
  /* [op(v2, v6), op(v3, v7), -, -, -, -, -, -] */ \
  const __m128i srli32 = _mm_srli_epi64(pairs.r, 32); \
  /* [op(v0, v4, v2, v6), op(v1, v5, v3, v7), -, -, -, -, -, -] */ \
  const auto quads = OP(pairs, {.r = srli32}); \
  /* [op(v1, v5, v3, v7), -, -, -, -, -, -, -] */ \
  const __m128i srli16 = _mm_srli_epi32(quads.r, 16); \
  /* [op(v1, …, v7), -, -, -, -, -, -, -][0] */ \
  return KIND##BITS(_mm_cvtsi128_si32(OP(quads, {.r = srli16}).r));
#define GREX_HMINMAX_i16x16 GREX_HMINMAX_HALVES
#define GREX_HMINMAX_i16x32 GREX_HMINMAX_HALVES
// i32/u32
#define GREX_HMINMAX_i32x2(OP, KIND, BITS, ...) \
  /* [v1, -, -, -] */ \
  const __m128i srli32 = _mm_srli_epi64(vf.r, 32); \
  /* [op(v0, v1), -, -, -][0] */ \
  return GREX_KINDCAST_SINGLE(i, KIND, 32, _mm_cvtsi128_si32(OP(vf, {.r = srli32}).r));
#define GREX_HMINMAX_i32x4(OP, KIND, BITS, ...) \
  /* [v2, v3, -, -] */ \
  const __m128i srli64 = _mm_srli_si128(v.r, 8); \
  /* [op(v0, v2), op(v1, v3), -, -] */ \
  const auto pairs = OP(v, {.r = srli64}); \
  /* [op(v1, v3), -, -, -] */ \
  const __m128i srli32 = _mm_srli_epi64(pairs.r, 32); \
  /* [op(v0, v2, v1, v3), -, -, -][0] */ \
  return GREX_KINDCAST_SINGLE(i, KIND, 32, _mm_cvtsi128_si32(OP(pairs, {.r = srli32}).r));
#define GREX_HMINMAX_i32x8 GREX_HMINMAX_HALVES
#define GREX_HMINMAX_i32x16 GREX_HMINMAX_HALVES
// i64/u64
#define GREX_HMINMAX_i64x2(OP, KIND, BITS, ...) \
  /* [v1, v1] */ \
  const __m128i unpack = _mm_unpackhi_epi64(v.r, v.r); \
  /* [op(v0, v1), op(v1, v1)][0] */ \
  return GREX_KINDCAST_SINGLE(i, KIND, 64, _mm_cvtsi128_si64(OP(v, {.r = unpack}).r));
#define GREX_HMINMAX_i64x4 GREX_HMINMAX_HALVES
#define GREX_HMINMAX_i64x8 GREX_HMINMAX_HALVES
// Conversion wrappers
#define GREX_HMINMAX_f(OP, KIND, BITS, SIZE, ...) \
  GREX_HMINMAX_f##BITS##x##SIZE(OP, KIND, BITS, SIZE, __VA_ARGS__)
#define GREX_HMINMAX_i(OP, KIND, BITS, SIZE, ...) \
  GREX_HMINMAX_i##BITS##x##SIZE(OP, KIND, BITS, SIZE, __VA_ARGS__)
#define GREX_HMINMAX_u(OP, KIND, BITS, SIZE, ...) \
  GREX_HMINMAX_i##BITS##x##SIZE(OP, KIND, BITS, SIZE, __VA_ARGS__)

// Wrapper macros
#define GREX_HMINMAX(KIND, BITS, SIZE, OP) \
  inline KIND##BITS horizontal_##OP(Vector<KIND##BITS, SIZE> v) { \
    GREX_HMINMAX_##KIND(OP, KIND, BITS, SIZE) \
  }
#define GREX_HMINMAX_SUB(KIND, BITS, PART, SIZE, OP) \
  inline KIND##BITS horizontal_##OP(SubVector<KIND##BITS, PART, SIZE> v) { \
    const auto vf = v.full; \
    GREX_HMINMAX_##KIND(OP, KIND, BITS, PART) \
  }

#define GREX_HMINMAX_ALL(REGISTERBITS, BITPREFIX) \
  GREX_FOREACH_TYPE(GREX_HMINMAX, REGISTERBITS, min) \
  GREX_FOREACH_TYPE(GREX_HMINMAX, REGISTERBITS, max)
GREX_FOREACH_X86_64_LEVEL(GREX_HMINMAX_ALL)

// SubVector
#define GREX_HMINMAX_SUB_ALL(KIND, BITS, PART, SIZE) \
  GREX_HMINMAX_SUB(KIND, BITS, PART, SIZE, min) \
  GREX_HMINMAX_SUB(KIND, BITS, PART, SIZE, max)
GREX_FOREACH_SUB(GREX_HMINMAX_SUB_ALL)

template<typename THalf>
inline THalf::Value horizontal_min(SuperVector<THalf> v) {
  return horizontal_min(min(v.lower, v.upper));
}
template<typename THalf>
inline THalf::Value horizontal_max(SuperVector<THalf> v) {
  return horizontal_max(max(v.lower, v.upper));
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_HORIZONTAL_MINMAX_HPP
