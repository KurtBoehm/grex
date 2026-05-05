// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_EXTRACT_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_EXTRACT_HPP

#include <cstddef>

#include <immintrin.h>

#include "grex/backend/base.hpp"
#include "grex/backend/macros/base.hpp"
#include "grex/backend/macros/for-each.hpp"
#include "grex/backend/macros/math.hpp"
#include "grex/backend/x86/instruction-sets.hpp"
#include "grex/backend/x86/macros/for-each.hpp"
#include "grex/backend/x86/macros/intrinsics.hpp"
#include "grex/backend/x86/types.hpp"
#include "grex/base.hpp"

#if GREX_X86_64_LEVEL >= 2
#include "grex/backend/macros/cast.hpp"
#endif

#if GREX_X86_64_LEVEL >= 3
#include "grex/backend/x86/operations/split.hpp"
#endif

#if GREX_X86_64_LEVEL >= 4
#include "grex/backend/x86/operations/bit.hpp"
#include "grex/backend/x86/operations/extract-single.hpp"
#include "grex/backend/x86/operations/mask-index.hpp"
#endif

#if GREX_X86_64_LEVEL < 4 || !GREX_HAS_AVX512VBMI2
#include <array>

#include "grex/backend/x86/operations/store.hpp"
#endif

namespace grex::backend {
// Generic scalar fallback: store to array, pick element (wrapping index)
#define GREX_EXTRACT_BASIC_FALLBACK(ELEMENT, SIZE, CALL, CONVERT) \
  std::array<ELEMENT, SIZE> x{}; \
  store(x.data(), v); \
  return x[i % SIZE];
// AVX-512: use mask-compress + scalar convert
#define GREX_EXTRACT_BASIC_AVX512(ELEMENT, SIZE, CALL, CONVERT) \
  const auto x = CALL; \
  return CONVERT;
// Compress v by mask selecting the i‑th element
#define GREX_MASKZ_CMPR(KIND, BITS, SIZE, BITPREFIX) \
  GREX_CAT(BITPREFIX##_maskz_compress_, GREX_EPI_SUFFIX(KIND, BITS)) \
  (single_mask(i, type_tag<NativeMask<KIND##BITS, SIZE>>).r, v.r)

#if GREX_X86_64_LEVEL >= 4
#define GREX_EXTRACT_BASIC_64 GREX_EXTRACT_BASIC_AVX512
#define GREX_EXTRACT_BASIC_32 GREX_EXTRACT_BASIC_AVX512
#else
#define GREX_EXTRACT_BASIC_64 GREX_EXTRACT_BASIC_FALLBACK
#define GREX_EXTRACT_BASIC_32 GREX_EXTRACT_BASIC_FALLBACK
#endif

#if GREX_X86_64_LEVEL >= 4 && GREX_HAS_AVX512VBMI2
#define GREX_EXTRACT_BASIC_16 GREX_EXTRACT_BASIC_AVX512
#define GREX_EXTRACT_BASIC_8 GREX_EXTRACT_BASIC_AVX512
#else
#define GREX_EXTRACT_BASIC_16 GREX_EXTRACT_BASIC_FALLBACK
#define GREX_EXTRACT_BASIC_8 GREX_EXTRACT_BASIC_FALLBACK
#endif

// Floating-point extraction
#define GREX_EXTRACT_FP_COMPRESS(REGISTERBITS, BITPREFIX, BITS, SIZE, ELETTER) \
  GREX_EXTRACT_BASIC_##BITS(f##BITS, SIZE, GREX_MASKZ_CMPR(f, BITS, SIZE, BITPREFIX), \
                            BITPREFIX##_cvts##ELETTER##_f##BITS(x))
#define GREX_EXTRACT_F32(...) GREX_EXTRACT_FP_COMPRESS(__VA_ARGS__)
// Special-case f64x2: use unpackhi + cvtsd
#define GREX_EXTRACT_F64_128(REGISTERBITS, BITPREFIX, BITS, ...) \
  GREX_EXTRACT_BASIC_##BITS(f64, 2, _mm_mask_unpackhi_pd(v.r, __mmask8(i), v.r, v.r), \
                            _mm_cvtsd_f64(x))
#define GREX_EXTRACT_F64_256(...) GREX_EXTRACT_FP_COMPRESS(__VA_ARGS__)
#define GREX_EXTRACT_F64_512(...) GREX_EXTRACT_FP_COMPRESS(__VA_ARGS__)
#define GREX_EXTRACT_F64(REGISTERBITS, ...) \
  GREX_EXTRACT_F64_##REGISTERBITS(REGISTERBITS, __VA_ARGS__)

// Integer extraction: convert low 128 bits to scalar
#define GREX_EXTRACT_CVTSI128_8 _mm_cvtsi128_si32
#define GREX_EXTRACT_CVTSI128_16 _mm_cvtsi128_si32
#define GREX_EXTRACT_CVTSI128_32 _mm_cvtsi128_si32
#define GREX_EXTRACT_CVTSI128_64 _mm_cvtsi128_si64

#define GREX_EXTRACT_VALUE_128(BITS) GREX_EXTRACT_CVTSI128_##BITS(x)
#define GREX_EXTRACT_VALUE_256(BITS) GREX_EXTRACT_CVTSI128_##BITS(_mm256_castsi256_si128(x))
#define GREX_EXTRACT_VALUE_512(BITS) GREX_EXTRACT_CVTSI128_##BITS(_mm512_castsi512_si128(x))

// Integer extraction
#define GREX_EXTRACT_INT_IMPL(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  GREX_EXTRACT_BASIC_##BITS(KIND##BITS, SIZE, GREX_MASKZ_CMPR(KIND, BITS, SIZE, BITPREFIX), \
                            GREX_KINDCAST_SINGLE_EXT(KIND, BITS, i, GREX_MAX(BITS, 32), \
                                                     GREX_EXTRACT_VALUE_##REGISTERBITS(BITS)))

// Dispatch for vector kinds
#define GREX_EXTRACT_VEC_f(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  GREX_EXTRACT_F##BITS(REGISTERBITS, BITPREFIX, BITS, SIZE, GREX_FP_LETTER(BITS))
#define GREX_EXTRACT_VEC_i GREX_EXTRACT_INT_IMPL
#define GREX_EXTRACT_VEC_u GREX_EXTRACT_INT_IMPL

// Generic extract for a given (kind, bits, size)
#define GREX_EXTRACT_VEC(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  inline KIND##BITS extract(NativeVector<KIND##BITS, SIZE> v, std::size_t i) { \
    GREX_EXTRACT_VEC_##KIND(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  }

// Instantiate for all types for this register width
#define GREX_EXTRACT_VEC_ALL(REGISTERBITS, BITPREFIX) \
  GREX_FOREACH_TYPE(GREX_EXTRACT_VEC, REGISTERBITS, BITPREFIX, REGISTERBITS)

// Extraction with compile-time index

// i8x16: use _mm_extract_epi8 if available, otherwise extract via 16‑bit lane
inline i8 extract(NativeVector<i8, 16> v, AnyIndexTag auto i) {
  static_assert(i < 16);
#if GREX_X86_64_LEVEL >= 2
  return i8(_mm_extract_epi8(v.r, i.value));
#else
  return i8(_mm_extract_epi16(v.r, i.value / 2) >> (8 * (i.value % 2)));
#endif
}

inline i16 extract(NativeVector<i16, 8> v, AnyIndexTag auto i) {
  static_assert(i < 8);
  return i16(_mm_extract_epi16(v.r, i.value));
}

// i32x4: use _mm_extract_epi32 if available, otherwise shuffle into lane 0
inline i32 extract(NativeVector<i32, 4> v, AnyIndexTag auto i) {
  static_assert(i < 4);
#if GREX_X86_64_LEVEL >= 2
  return _mm_extract_epi32(v.r, i.value);
#else
  return _mm_cvtsi128_si32(_mm_shuffle_epi32(v.r, i.value));
#endif
}

// i64x2: use _mm_extract_epi64 if available, otherwise unpack hi for index 1
inline i64 extract(NativeVector<i64, 2> v, AnyIndexTag auto i) {
  static_assert(i < 2);
#if GREX_X86_64_LEVEL >= 2
  return _mm_extract_epi64(v.r, i.value);
#else
  return _mm_cvtsi128_si64((i == 1) ? _mm_unpackhi_epi64(v.r, v.r) : v.r);
#endif
}

// f32x4: shuffle requested lane into lane 0 then cvtss
inline f32 extract(NativeVector<f32, 4> v, AnyIndexTag auto i) {
  static_assert(i < 4);
  const __m128i shuf = _mm_shuffle_epi32(_mm_castps_si128(v.r), i.value);
  return _mm_cvtss_f32(_mm_castsi128_ps(shuf));
}

// f64x2: for lane 1, unpackhi; then cvtsd
inline f64 extract(NativeVector<f64, 2> v, AnyIndexTag auto i) {
  static_assert(i < 2);
  const __m128d shuf = (i == 1) ? _mm_unpackhi_pd(v.r, v.r) : v.r;
  return _mm_cvtsd_f64(shuf);
}

#if GREX_X86_64_LEVEL >= 3
// AVX2: use _mm256_extract for integer vectors
#define GREX_STRACT_I256(KIND, BITS, SIZE) \
  inline KIND##BITS extract(NativeVector<KIND##BITS, SIZE> v, AnyIndexTag auto i) { \
    static_assert(i < SIZE); \
    return GREX_RETCAST(KIND, BITS, _mm256_extract_epi##BITS(v.r, i.value)); \
  }
GREX_FOREACH_INT_TYPE(GREX_STRACT_I256, 256)

// f32x8: just extract from low or high 128‑bit half
inline f32 extract(f32x8 v, AnyIndexTag auto i) {
  static_assert(i < 8);
  if constexpr (i < 4) {
    return extract(get_low(v), i);
  } else {
    return extract(get_high(v), index_tag<i - 4>);
  }
}

// f64x4: low half via get_low, high half via 4×64 permutation+cvtsd
inline f64 extract(f64x4 v, AnyIndexTag auto i) {
  static_assert(i < 4);
  if constexpr (i < 2) {
    return extract(get_low(v), i);
  } else {
    return _mm256_cvtsd_f64(_mm256_permute4x64_pd(v.r, i.value));
  }
}
#endif

#if GREX_X86_64_LEVEL >= 4
// Helpers to extract 128‑bit lanes from 512‑bit vectors
#define GREX_TRACT_512_INT(A, IMM8) _mm512_extracti32x4_epi32(A, IMM8)
#define GREX_TRACT_512_F32(A, IMM8) _mm512_extractf32x4_ps(A, IMM8)
#define GREX_TRACT_512_F64(A, IMM8) _mm512_extractf64x2_pd(A, IMM8)

#define GREX_TRACT_512_f(BITS, A, IMM8) GREX_TRACT_512_F##BITS(A, IMM8)
#define GREX_TRACT_512_i(BITS, A, IMM8) GREX_TRACT_512_INT(A, IMM8)
#define GREX_TRACT_512_u(BITS, A, IMM8) GREX_TRACT_512_INT(A, IMM8)
#define GREX_TRACT_512(KIND, BITS, A, IMM8) GREX_TRACT_512_##KIND(BITS, A, IMM8)

// 512‑bit integer: split into 4 lanes of equal size and recurse
#define GREX_STRACT_512_INT(KIND, BITS, SIZE) \
  if constexpr (lane_idx < 2) { \
    return extract(get_low(v), i); \
  } else { \
    const auto x = GREX_TRACT_512(KIND, BITS, v.r, lane_idx); \
    return extract(NativeVector<KIND##BITS, lane_size>{x}, \
                   index_tag<i.value - lane_idx * lane_size>); \
  }

// 512‑bit floating-point: rotate bits so desired element is at position 0, then extract_single
#define GREX_STRACT_512_FP(KIND, BITS, SIZE) \
  if constexpr (lane_idx < 2) { \
    return extract(get_low(v), i); \
  } else { \
    const auto iv = GREX_KINDCAST(KIND, i, BITS, 512, v.r); \
    const auto ix = _mm512_alignr_epi##BITS(iv, iv, i.value); \
    const auto x = GREX_KINDCAST(i, KIND, BITS, 512, ix); \
    return extract_single(NativeVector<KIND##BITS, SIZE>{x}); \
  }

#define GREX_STRACT_512_f GREX_STRACT_512_FP
#define GREX_STRACT_512_i GREX_STRACT_512_INT
#define GREX_STRACT_512_u GREX_STRACT_512_INT

// Generic 512‑bit extract: compute lane index, delegate to INT/FP variant
#define GREX_STRACT_512(KIND, BITS, SIZE) \
  inline KIND##BITS extract(NativeVector<KIND##BITS, SIZE> v, AnyIndexTag auto i) { \
    static_assert(i < SIZE); \
    constexpr std::size_t lane_size = GREX_DIVIDE(SIZE, 4); \
    constexpr std::size_t lane_idx = i.value / lane_size; \
    GREX_STRACT_512_##KIND(KIND, BITS, SIZE) \
  }

GREX_FOREACH_TYPE(GREX_STRACT_512, 512)
#endif

// Mask extraction
#if GREX_X86_64_LEVEL >= 4
// Compact masks: test bit i
#define GREX_EXTRACT_MASK_IMPL(KIND, BITS, SIZE, UMMASK) \
  inline bool extract(NativeMask<KIND##BITS, SIZE> v, std::size_t i) { \
    return bit_test(v.r, UMMASK(i)); \
  }
#define GREX_EXTRACT_MASK(KIND, BITS, SIZE) \
  GREX_EXTRACT_MASK_IMPL(KIND, BITS, SIZE, GREX_CAT(u, GREX_MAX(SIZE, 8)))
#else
// Broad masks: load mask as vector and test element != 0
#define GREX_EXTRACT_MASK(KIND, BITS, SIZE) \
  inline bool extract(NativeMask<KIND##BITS, SIZE> v, std::size_t i) { \
    return extract(NativeVector<u##BITS, SIZE>{v.r}, i) != 0; \
  }
#endif

#define GREX_EXTRACT_MASK_ALL(REGISTERBITS, BITPREFIX) \
  GREX_FOREACH_TYPE(GREX_EXTRACT_MASK, REGISTERBITS)

// Instantiate scalar extract and mask extract for each x86-64 level
GREX_FOREACH_X86_64_LEVEL(GREX_EXTRACT_VEC_ALL)
GREX_FOREACH_X86_64_LEVEL(GREX_EXTRACT_MASK_ALL)
} // namespace grex::backend

#include "grex/backend/shared/operations/extract.hpp" // IWYU pragma: export

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_EXTRACT_HPP
