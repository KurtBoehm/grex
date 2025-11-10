// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_INSERT_STATIC_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_INSERT_STATIC_HPP

#include "grex/backend/defs.hpp"
#include "grex/backend/macros/base.hpp"
#include "grex/backend/macros/cast.hpp"
#include "grex/backend/macros/for-each.hpp"
#include "grex/backend/macros/math.hpp"
#include "grex/backend/x86/instruction-sets.hpp"
#include "grex/backend/x86/macros/for-each.hpp"
#include "grex/backend/x86/macros/intrinsics.hpp"
#include "grex/backend/x86/operations/expand-scalar.hpp"
#include "grex/backend/x86/types.hpp"
#include "grex/base/defs.hpp"

namespace grex::backend {
#define GREX_VEC_SINSERT_INTRINSIC(KIND, BITS, SIZE, BITPREFIX) \
  return {.r = GREX_CAT(BITPREFIX##_insert_, GREX_EPI_SUFFIX(KIND, BITS))( \
            v.r, GREX_SIGNED_CAST(KIND, BITS, value), index.value)};
#define GREX_VEC_SINSERT_FALLBACK(KIND, BITS, SIZE, BITPREFIX) return insert(v, index.value, value);

#define GREX_VEC_SINSERT_AVX512(KIND, BITS, SIZE, BITPREFIX) \
  if constexpr (BITS * index < 128) { \
    const auto v128 = GREX_CAT(_mm512_cast, GREX_SIR_SUFFIX(KIND, BITS, 512), _, \
                               GREX_SIR_SUFFIX(KIND, BITS, 128))(v.r); \
    const auto ins128 = insert(Vector<KIND##BITS, 128 / BITS>{v128}, index, value).r; \
    const auto ins512 = GREX_CAT(_mm512_insert, GREX_SIGNED_KIND(KIND), GREX_MAX(BITS, 32), x, \
                                 GREX_DIVIDE(128, GREX_MAX(BITS, 32)))(v.r, ins128, 0); \
    return {.r = ins512}; \
  } \
  return insert(v, index.value, value);

#define GREX_VEC_SINSERT_f64x2(KIND, BITS, SIZE, BITPREFIX) \
  if constexpr (index == 0) { \
    return {.r = _mm_move_sd(v.r, expand_any(Scalar{value}, index_tag<2>).r)}; \
  } else { \
    return {.r = _mm_unpacklo_pd(v.r, expand_any(Scalar{value}, index_tag<2>).r)}; \
  }
#if GREX_X86_64_LEVEL >= 2
#define GREX_VEC_SINSERT_i64x2 GREX_VEC_SINSERT_INTRINSIC
#define GREX_VEC_SINSERT_f32x4(KIND, BITS, SIZE, BITPREFIX) \
  if constexpr (index == 0) { \
    return {.r = _mm_move_ss(v.r, expand_any(Scalar{value}, index_tag<4>).r)}; \
  } else { \
    return {.r = _mm_insert_ps(v.r, expand_any(Scalar{value}, index_tag<4>).r, index.value << 4)}; \
  }
#define GREX_VEC_SINSERT_i32x4 GREX_VEC_SINSERT_INTRINSIC
#define GREX_VEC_SINSERT_i16x8 GREX_VEC_SINSERT_INTRINSIC
#define GREX_VEC_SINSERT_i8x16 GREX_VEC_SINSERT_INTRINSIC
#else
#define GREX_VEC_SINSERT_i64x2 GREX_VEC_SINSERT_FALLBACK
#define GREX_VEC_SINSERT_f32x4(KIND, BITS, SIZE, BITPREFIX) \
  const __m128 vec = expand_any(Scalar{value}, index_tag<4>).r; \
  if constexpr (index == 0) { \
    return {.r = _mm_move_ss(v.r, vec)}; \
  } else if (index == 1) { \
    return {.r = _mm_shuffle_ps(_mm_movelh_ps(vec, v.r), v.r, 0b11'10'00'10)}; \
  } else { \
    /* [value, value, v[2], v[3]] */ \
    const auto a = _mm_shuffle_ps(vec, v.r, 0b11'10'00'00); \
    constexpr int imm8 = 0b11'10'01'00 & (0xFF - (0b11 << (2 * index))); \
    return {.r = _mm_shuffle_ps(v.r, a, imm8)}; \
  }
#define GREX_VEC_SINSERT_i32x4 GREX_VEC_SINSERT_FALLBACK
#define GREX_VEC_SINSERT_i16x8 GREX_VEC_SINSERT_INTRINSIC
#define GREX_VEC_SINSERT_i8x16 GREX_VEC_SINSERT_FALLBACK
#endif

#define GREX_VEC_SINSERT_f64x4(KIND, BITS, SIZE, BITPREFIX) \
  __m256d ins; \
  if constexpr (index == 0) { \
    ins = expand_any(Scalar{value}, index_tag<4>).r; \
  } else if (index == 1) { \
    ins = _mm256_castpd128_pd256( \
      _mm_unpacklo_pd(_mm256_castpd256_pd128(v.r), expand_any(Scalar{value}, index_tag<2>).r)); \
  } else { \
    ins = _mm256_broadcastsd_pd(expand_any(Scalar{value}, index_tag<2>).r); \
  } \
  return {.r = _mm256_blend_pd(v.r, ins, 1 << index.value)};
#define GREX_VEC_SINSERT_i64x4(KIND, BITS, SIZE, BITPREFIX) \
  __m256i ins; \
  if constexpr (index == 0) { \
    ins = _mm256_castsi128_si256(_mm_cvtsi64_si128(GREX_SIGNED_CAST(KIND, BITS, value))); \
  } else if constexpr (index == 1) { \
    ins = _mm256_castsi128_si256(_mm_insert_epi64( \
      _mm256_castsi256_si128(v.r), GREX_SIGNED_CAST(KIND, BITS, value), index.value)); \
  } else { \
    ins = _mm256_set1_epi64x(GREX_SIGNED_CAST(KIND, BITS, value)); \
  } \
  return {.r = _mm256_blend_epi32(v.r, ins, 0b11 << (2 * index.value))};
#define GREX_VEC_SINSERT_f32x8(KIND, BITS, SIZE, BITPREFIX) \
  __m256 ins; \
  if constexpr (index == 0) { \
    ins = expand_any(Scalar{value}, index_tag<8>).r; \
  } else if (index < 4) { \
    ins = _mm256_castps128_ps256(_mm_insert_ps( \
      _mm256_castps256_ps128(v.r), expand_any(Scalar{value}, index_tag<4>).r, index.value << 4)); \
  } else { \
    ins = _mm256_broadcastss_ps(expand_any(Scalar{value}, index_tag<4>).r); \
  } \
  return {.r = _mm256_blend_ps(v.r, ins, 1 << index.value)};
#define GREX_VEC_SINSERT_i32x8(KIND, BITS, SIZE, BITPREFIX) \
  __m256i ins; \
  if constexpr (index == 0) { \
    ins = _mm256_castsi128_si256(_mm_cvtsi32_si128(GREX_SIGNED_CAST(KIND, BITS, value))); \
  } else if constexpr (index < 4) { \
    ins = _mm256_castsi128_si256(_mm_insert_epi32( \
      _mm256_castsi256_si128(v.r), GREX_SIGNED_CAST(KIND, BITS, value), index.value)); \
  } else { \
    ins = _mm256_set1_epi32(GREX_SIGNED_CAST(KIND, BITS, value)); \
  } \
  return {.r = _mm256_blend_epi32(v.r, ins, 1 << index.value)};
#define GREX_VEC_SINSERT_i16x16(KIND, BITS, SIZE, BITPREFIX) \
  __m256i ins; \
  if constexpr (index < 8) { \
    ins = _mm256_castsi128_si256(_mm_insert_epi16( \
      _mm256_castsi256_si128(v.r), GREX_SIGNED_CAST(KIND, BITS, value), index.value)); \
  } else { \
    ins = _mm256_blend_epi16(v.r, _mm256_set1_epi16(GREX_SIGNED_CAST(KIND, BITS, value)), \
                             1 << (index.value - 8)); \
  } \
  return {.r = _mm256_blend_epi32(v.r, ins, 1 << (index.value / 2))};
#define GREX_VEC_SINSERT_i8x32(KIND, BITS, SIZE, BITPREFIX) \
  if constexpr (index < 16) { \
    const __m128i ins = _mm_insert_epi8(_mm256_castsi256_si128(v.r), \
                                        GREX_SIGNED_CAST(KIND, BITS, value), index.value); \
    return {.r = _mm256_blend_epi32(v.r, _mm256_castsi128_si256(ins), 1 << (index.value / 4))}; \
  } else { \
    __m128i upper = _mm256_extracti128_si256(v.r, 1); \
    const __m128i ins = \
      _mm_insert_epi8(upper, GREX_SIGNED_CAST(KIND, BITS, value), index.value - 16); \
    return {.r = _mm256_inserti128_si256(v.r, ins, 1)}; \
  }

#define GREX_VEC_SINSERT_f64x8 GREX_VEC_SINSERT_AVX512
#define GREX_VEC_SINSERT_i64x8 GREX_VEC_SINSERT_AVX512
#define GREX_VEC_SINSERT_f32x16 GREX_VEC_SINSERT_AVX512
#define GREX_VEC_SINSERT_i32x16 GREX_VEC_SINSERT_AVX512
#define GREX_VEC_SINSERT_i16x32 GREX_VEC_SINSERT_AVX512
#define GREX_VEC_SINSERT_i8x64 GREX_VEC_SINSERT_AVX512

#if GREX_X86_64_LEVEL >= 4
#define GREX_MASK_SINSERT(KIND, BITS, SIZE, BITPREFIX) \
  inline Mask<KIND##BITS, SIZE> insert(Mask<KIND##BITS, SIZE> m, AnyIndexTag auto index, \
                                       bool value) { \
    return insert(m, index.value, value); \
  }
#else
#define GREX_MASK_SINSERT(KIND, BITS, SIZE, BITPREFIX) \
  inline Mask<KIND##BITS, SIZE> insert(Mask<KIND##BITS, SIZE> m, AnyIndexTag auto index, \
                                       bool value) { \
    const i##BITS entry = GREX_OPCAST(i, BITS, -i##BITS(value)); \
    return {.r = insert(Vector<i##BITS, SIZE>{.r = m.r}, index, entry).r}; \
  }
#endif

#define GREX_VEC_SINSERT(KIND, BITS, SIZE, BITPREFIX) \
  inline Vector<KIND##BITS, SIZE> insert(Vector<KIND##BITS, SIZE> v, AnyIndexTag auto index, \
                                         KIND##BITS value) { \
    GREX_CAT(GREX_VEC_SINSERT_, GREX_SIGNED_KIND(KIND), BITS, x, SIZE)(KIND, BITS, SIZE, \
                                                                       BITPREFIX) \
  }

#define GREX_SINSERT_ALL(REGISTERBITS, BITPREFIX) \
  GREX_FOREACH_TYPE(GREX_VEC_SINSERT, REGISTERBITS, BITPREFIX) \
  GREX_FOREACH_TYPE(GREX_MASK_SINSERT, REGISTERBITS, BITPREFIX)
GREX_FOREACH_X86_64_LEVEL(GREX_SINSERT_ALL)
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_INSERT_STATIC_HPP
