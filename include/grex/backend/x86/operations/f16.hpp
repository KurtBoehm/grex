// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_F16_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_F16_HPP

#include <cstddef>

#include "grex/backend/base.hpp"
#include "grex/backend/choosers.hpp"
#include "grex/backend/defs.hpp"
#include "grex/backend/x86/types.hpp"
#include "grex/base.hpp"

#if GREX_X86_64_LEVEL < 3
#include <bit>
#else
#include "grex/backend/x86/operations/intrinsics.hpp"
#endif

#if GREX_F16_NATIVE_ARITHMETIC
#include <concepts>

#include <immintrin.h>
#else
#include "grex/backend/x86/operations/merge.hpp"
#include "grex/backend/x86/operations/split.hpp"
#endif

namespace grex::backend {
#if GREX_F16_NATIVE_ARITHMETIC
// Binary16: Reinterpret to/from the same-width unsigned-integer register which a binary16 vector is
// stored in.
inline __m128h as_ph(__m128i r) {
  return _mm_castsi128_ph(r);
}
inline __m256h as_ph(__m256i r) {
  return _mm256_castsi256_ph(r);
}
inline __m512h as_ph(__m512i r) {
  return _mm512_castsi512_ph(r);
}

inline __m128i as_u16(__m128h r) {
  return _mm_castph_si128(r);
}
inline __m256i as_u16(__m256h r) {
  return _mm256_castph_si256(r);
}
inline __m512i as_u16(__m512h r) {
  return _mm512_castph_si512(r);
}
#endif

/**
 * Convert an x86 register to its stored type, which maps `f16` to the same-width unsigned-integer
 * register (see `as_u16` above) and returns the input unchanged otherwise.
 */
template<typename TValue>
GREX_ALWAYS_INLINE inline auto to_stored(auto r) {
#if GREX_F16_NATIVE_ARITHMETIC
  if constexpr (std::same_as<TValue, f16>) {
    return as_u16(r);
  } else
#endif
  {
    return r;
  }
}

/**
 * Convert an x86 register from its stored type, which maps the same-width unsigned-integer register
 * to `f16` (see `as_ph` above) if `TValue` is `f16` and returns the input unchanged otherwise.
 */
template<typename TValue>
GREX_ALWAYS_INLINE inline auto from_stored(auto r) {
#if GREX_F16_NATIVE_ARITHMETIC
  if constexpr (std::same_as<TValue, f16>) {
    return as_ph(r);
  } else
#endif
  {
    return r;
  }
}

// Native conversions are only available through F16C, which is part of level 3, or AVX512-FP16,
// which is handled separately.
#if GREX_X86_64_LEVEL >= 3
template<std::size_t tSize>
requires(tSize < 8)
GREX_ALWAYS_INLINE inline VectorFor<f32, tSize> f16_to_f32(SubVector<f16, tSize> v) {
  return VectorFor<f32, tSize>{_mm_cvtph_ps(v.full.r)};
}
template<TypedVector<f32> TVec>
requires(size_of<TVec> < 8)
GREX_ALWAYS_INLINE inline SubVector<f16, size_of<TVec>> f32_to_f16(TVec v) {
  return SubVector<f16, size_of<TVec>>{mm::cvtps_ph(v.registr())};
}

GREX_ALWAYS_INLINE inline VectorFor<f32, 8> f16_to_f32(NativeVector<f16, 8> v) {
  return {.r = _mm256_cvtph_ps(v.r)};
}
GREX_ALWAYS_INLINE inline NativeVector<f16, 8> f32_to_f16(VectorFor<f32, 8> v) {
  return {.r = mm256::cvtps_ph(v.r)};
}

GREX_ALWAYS_INLINE inline VectorFor<f32, 16> f16_to_f32(NativeVector<f16, 16> v) {
#if GREX_X86_64_LEVEL >= 4
  return {.r = _mm512_cvtph_ps(v.r)};
#else
  return {
    .lower = {.r = _mm256_cvtph_ps(_mm256_castsi256_si128(v.r))},
    .upper = {.r = _mm256_cvtph_ps(_mm256_extracti128_si256(v.r, 1))},
  };
#endif
}
GREX_ALWAYS_INLINE inline NativeVector<f16, 16> f32_to_f16(VectorFor<f32, 16> v) {
#if GREX_X86_64_LEVEL >= 4
  return {.r = mm512::cvtps_ph(v.r)};
#else
  const __m128i lower = mm256::cvtps_ph(v.lower.r);
  const __m128i upper = mm256::cvtps_ph(v.upper.r);
  return {.r = _mm256_inserti128_si256(_mm256_castsi128_si256(lower), upper, 1)};
#endif
}
#else
//==================================================================================================
// Binary16 ↔ binary32 (both ways)
//==================================================================================================

// Without F16C (below x86-64-v3), binary16 ↔ binary32 conversions are emulated directly with SSE2
// intrinsics, using x86-64-v2 equivalents where that is more efficient. The bit manipulations
// follow the well-known branch-free formulations that turn the exponent adjustment into an integer
// addition and handle subnormals with a single floating-point addition.
//
// 8 lanes (one 128-bit register) is the widest binary16 vector converted this way, since F16C is
// unconditionally available from x86-64-v3 onward.
namespace f16_convert {
/** Zero-extend the low four `u16` lanes of `v` to `u32`. */
GREX_ALWAYS_INLINE inline __m128i widen_lo(__m128i v) {
#if GREX_X86_64_LEVEL >= 2
  return _mm_cvtepu16_epi32(v);
#else
  return _mm_unpacklo_epi16(v, _mm_setzero_si128());
#endif
}
/** Zero-extend the high four `u16` lanes of `v` to `u32`. */
GREX_ALWAYS_INLINE inline __m128i widen_hi(__m128i v) {
  return _mm_unpackhi_epi16(v, _mm_setzero_si128());
}

/** Select `v1` where `mask` is all-ones and `v0` where it is all-zero, per 32-bit lane. */
GREX_ALWAYS_INLINE inline __m128i select(__m128i mask, __m128i v0, __m128i v1) {
#if GREX_X86_64_LEVEL >= 2
  return _mm_blendv_epi8(v0, v1, mask);
#else
  return _mm_or_si128(_mm_andnot_si128(mask, v0), _mm_and_si128(mask, v1));
#endif
}

/** Narrow four low and four high `u32` lanes, each holding a value below `0x10000`, to `u16`. */
GREX_ALWAYS_INLINE inline __m128i narrow(__m128i lo, __m128i hi) {
#if GREX_X86_64_LEVEL >= 2
  return _mm_packus_epi32(lo, hi);
#else
  const __m128i bias = _mm_set1_epi32(0x8000);
  const __m128i packed = _mm_packs_epi32(_mm_sub_epi32(lo, bias), _mm_sub_epi32(hi, bias));
  return _mm_xor_si128(packed, _mm_set1_epi16(-0x8000));
#endif
}

/**
 * Convert four zero-extended binary16 bit patterns (`u32` lanes, as produced by `widen_lo`/
 * `widen_hi`) into the binary32 vector with the same values, which is exact.
 */
GREX_ALWAYS_INLINE inline __m128 to_f32(__m128i bits) {
  const __m128i sign = _mm_slli_epi32(_mm_and_si128(bits, _mm_set1_epi32(0x8000)), 16);
  // Exponent and mantissa moved into their binary32 positions
  const __m128i rest = _mm_slli_epi32(_mm_and_si128(bits, _mm_set1_epi32(0x7FFF)), 13);
  const __m128i expo = _mm_and_si128(rest, _mm_set1_epi32(0x0F800000));

  // Re-bias the exponent from 15 to 127.
  const __m128i biased = _mm_add_epi32(rest, _mm_set1_epi32(0x38000000));
  // Infinity/not-a-number: The exponent must be all ones, which needs another (128 - 16) << 23.
  const __m128i infinite = _mm_add_epi32(biased, _mm_set1_epi32(0x38000000));
  // Subnormal binary16 values become normal binary32 values: Adding 2⁻¹⁴ (as an integer increment
  // of the exponent) and subtracting it again as a floating-point value normalizes the mantissa.
  const __m128i shifted = _mm_add_epi32(biased, _mm_set1_epi32(0x00800000));
  const __m128i subnormal = _mm_castps_si128(
    _mm_sub_ps(_mm_castsi128_ps(shifted), _mm_castsi128_ps(_mm_set1_epi32(0x38800000))));

  __m128i out = select(_mm_cmpeq_epi32(expo, _mm_set1_epi32(0x0F800000)), biased, infinite);
  out = select(_mm_cmpeq_epi32(expo, _mm_setzero_si128()), out, subnormal);
  return _mm_castsi128_ps(_mm_or_si128(out, sign));
}

/**
 * Convert a binary32 vector into four `u32` lanes, each holding the bit pattern of the nearest
 * binary16 value (below `0x10000`, ready for `narrow`), rounding ties to even.
 */
GREX_ALWAYS_INLINE inline __m128i from_f32(__m128 v) {
  const __m128i bits = _mm_castps_si128(v);
  const __m128i sign = _mm_and_si128(bits, _mm_set1_epi32(std::bit_cast<int>(0x80000000)));
  // The magnitude, i.e. the input without its sign.
  const __m128i mag = _mm_xor_si128(bits, sign);

  // |x| ≥ 2¹⁶ rounds to infinity, unless the input is a not-a-number, which stays one
  const __m128i is_nan = _mm_cmpgt_epi32(mag, _mm_set1_epi32(0x7F800000));
  const __m128i saturated = select(is_nan, _mm_set1_epi32(0x7C00), _mm_set1_epi32(0x7E00));

  // |x| < 2⁻¹⁴ becomes a binary16 subnormal, whose value is m · 2⁻²⁴ for a mantissa m < 2¹⁰:
  // Adding ½ makes the binary32 mantissa field equal to m, with correct rounding, since
  // ½ + m · 2⁻²⁴ = 2⁻¹ · (1 + m · 2⁻²³), so subtracting the bits of ½ leaves exactly m.
  const __m128i magic = _mm_set1_epi32(0x3F000000);
  const __m128i subnormal = _mm_sub_epi32(
    _mm_castps_si128(_mm_add_ps(_mm_castsi128_ps(mag), _mm_castsi128_ps(magic))), magic);

  // Normal result: Re-bias the exponent from 127 to 15 and round to nearest, ties to even.
  const __m128i odd = _mm_and_si128(_mm_srli_epi32(mag, 13), _mm_set1_epi32(1));
  const __m128i rounded =
    _mm_add_epi32(_mm_add_epi32(mag, _mm_set1_epi32(std::bit_cast<int>(0xC8000FFF))), odd);
  const __m128i normal = _mm_srli_epi32(rounded, 13);

  __m128i out = select(_mm_cmpgt_epi32(_mm_set1_epi32(0x38800000), mag), normal, subnormal);
  out = select(_mm_cmpgt_epi32(_mm_set1_epi32(0x47800000), mag), saturated, out);
  return _mm_or_si128(out, _mm_srli_epi32(sign, 16));
}
} // namespace f16_convert

// Sub-native binary16 vectors are converted through the native register they wrap, which is the
// only one that is ever actually populated.
template<std::size_t tSize>
requires(tSize == 2 || tSize == 4)
GREX_ALWAYS_INLINE inline VectorFor<f32, tSize> f16_to_f32(SubVector<f16, tSize> v) {
  return VectorFor<f32, tSize>{f16_convert::to_f32(f16_convert::widen_lo(v.registr()))};
}
GREX_ALWAYS_INLINE inline SubVector<f16, 4> f32_to_f16(f32x4 v) {
  const __m128i bits = f16_convert::from_f32(v.r);
  return SubVector<f16, 4>{f16_convert::narrow(bits, bits)};
}
GREX_ALWAYS_INLINE inline SubVector<f16, 2> f32_to_f16(SubVector<f32, 2> v) {
  const __m128i bits = f16_convert::from_f32(v.registr());
  return SubVector<f16, 2>{f16_convert::narrow(bits, bits)};
}

GREX_ALWAYS_INLINE inline VectorFor<f32, 8> f16_to_f32(NativeVector<f16, 8> v) {
  return {
    .lower = {.r = f16_convert::to_f32(f16_convert::widen_lo(v.r))},
    .upper = {.r = f16_convert::to_f32(f16_convert::widen_hi(v.r))},
  };
}
GREX_ALWAYS_INLINE inline NativeVector<f16, 8> f32_to_f16(VectorFor<f32, 8> v) {
  return {
    .r = f16_convert::narrow(f16_convert::from_f32(v.lower.r), f16_convert::from_f32(v.upper.r)),
  };
}
#endif

#if GREX_X86_64_LEVEL >= 4
// Binary16 ↔ binary32: The binary32 result of 32 binary16 lanes needs two 512-bit registers.
GREX_ALWAYS_INLINE inline VectorFor<f32, 32> f16_to_f32(NativeVector<f16, 32> v) {
  return {
    .lower = {.r = _mm512_cvtph_ps(_mm512_castsi512_si256(v.r))},
    .upper = {.r = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(v.r, 1))},
  };
}
GREX_ALWAYS_INLINE inline NativeVector<f16, 32> f32_to_f16(VectorFor<f32, 32> v) {
  const __m256i lower = mm512::cvtps_ph(v.lower.r);
  const __m256i upper = mm512::cvtps_ph(v.upper.r);
  return {.r = _mm512_inserti64x4(_mm512_castsi256_si512(lower), upper, 1)};
}
#endif

//==================================================================================================
// Binary16 ↔ binary64 (both ways)
//==================================================================================================

#if GREX_F16_NATIVE_ARITHMETIC
// AVX512-FP16 converts between binary16 and binary64 in a single instruction, which `convert` uses
// directly (see grex/backend/x86/operations/convert/binary16.hpp), so no detour is required.
#else
// Without AVX512-FP16, binary64 is reached through binary32, reusing the conversions above rather
// than repeating their bit manipulation for the wider format. Narrowing twice with round-to-nearest
// would round twice, which no amount of intermediate precision repairs for an arbitrary binary64
// input, so the first step rounds to odd instead and the second one is then correctly rounded:
// A binary16 rounding boundary is a midpoint between two binary16 values and therefore has an even
// significand in every format with at least p + 2 = 13 bits, so an odd intermediate can neither
// land on such a boundary nor, being a neighbour of the input, cross one. Binary32 offers 24 bits,
// well above that bound.
namespace f16_convert {
// A binary64 significand stores 52 bits and a binary32 one 23, so narrowing discards the low 29.
inline constexpr i64 f64_discarded = (i64{1} << 29) - 1;

/**
 * Round the binary64 lanes of `v` to the significand of binary32, breaking ties towards an odd
 * significand: Adding the mask of the discarded bits to those bits carries into the lowest
 * surviving bit exactly if any of them is set, so the disjunction forces that bit to one precisely
 * if the narrowing loses information.
 *
 * Infinities and not-a-numbers survive this unchanged, the latter because the sticky bit keeps a
 * significand whose surviving bits are all zero from turning into an infinity.
 */
GREX_ALWAYS_INLINE inline __m128d round_odd(__m128d v) {
  const __m128i bits = _mm_castpd_si128(v);
  const __m128i mask = _mm_set1_epi64x(f64_discarded);
  const __m128i sticky = _mm_and_si128(_mm_add_epi64(_mm_and_si128(bits, mask), mask),
                                       _mm_set1_epi64x(f64_discarded + 1));
  return _mm_castsi128_pd(_mm_or_si128(_mm_andnot_si128(mask, bits), sticky));
}
#if GREX_X86_64_LEVEL >= 3
GREX_ALWAYS_INLINE inline __m256d round_odd(__m256d v) {
  const __m256i bits = _mm256_castpd_si256(v);
  const __m256i mask = _mm256_set1_epi64x(f64_discarded);
  const __m256i sticky = _mm256_and_si256(_mm256_add_epi64(_mm256_and_si256(bits, mask), mask),
                                          _mm256_set1_epi64x(f64_discarded + 1));
  return _mm256_castsi256_pd(_mm256_or_si256(_mm256_andnot_si256(mask, bits), sticky));
}
#endif
#if GREX_X86_64_LEVEL >= 4
GREX_ALWAYS_INLINE inline __m512d round_odd(__m512d v) {
  const __m512i bits = _mm512_castpd_si512(v);
  const __m512i mask = _mm512_set1_epi64(f64_discarded);
  const __m512i sticky = _mm512_and_si512(_mm512_add_epi64(_mm512_and_si512(bits, mask), mask),
                                          _mm512_set1_epi64(f64_discarded + 1));
  return _mm512_castsi512_pd(_mm512_or_si512(_mm512_andnot_si512(mask, bits), sticky));
}
#endif
} // namespace f16_convert

/** Narrow a binary64 vector to binary32, rounding to odd (see `f16_convert::round_odd`). */
GREX_ALWAYS_INLINE inline VectorFor<f32, 2> f64_to_f32_odd(NativeVector<f64, 2> v) {
  return VectorFor<f32, 2>{_mm_cvtpd_ps(f16_convert::round_odd(v.r))};
}
#if GREX_X86_64_LEVEL >= 3
/** Narrow a binary64 vector to binary32, rounding to odd (see `f16_convert::round_odd`). */
GREX_ALWAYS_INLINE inline NativeVector<f32, 4> f64_to_f32_odd(NativeVector<f64, 4> v) {
  return {.r = _mm256_cvtpd_ps(f16_convert::round_odd(v.r))};
}
#endif
#if GREX_X86_64_LEVEL >= 4
/** Narrow a binary64 vector to binary32, rounding to odd (see `f16_convert::round_odd`). */
GREX_ALWAYS_INLINE inline NativeVector<f32, 8> f64_to_f32_odd(NativeVector<f64, 8> v) {
  return {.r = _mm512_cvtpd_ps(f16_convert::round_odd(v.r))};
}
#endif
/** Narrow a binary64 vector to binary32, rounding to odd (see `f16_convert::round_odd`). */
template<TypedVector<f64> THalf>
GREX_ALWAYS_INLINE inline VectorFor<f32, 2 * THalf::size> f64_to_f32_odd(SuperVector<THalf> v) {
  return merge(f64_to_f32_odd(v.lower), f64_to_f32_odd(v.upper));
}

/** Widen a binary32 vector to binary64, which is always exact. */
GREX_ALWAYS_INLINE inline NativeVector<f64, 2> f32_to_f64(VectorFor<f32, 2> v) {
  return {.r = _mm_cvtps_pd(v.registr())};
}
#if GREX_X86_64_LEVEL >= 3
/** Widen a binary32 vector to binary64, which is always exact. */
GREX_ALWAYS_INLINE inline NativeVector<f64, 4> f32_to_f64(NativeVector<f32, 4> v) {
  return {.r = _mm256_cvtps_pd(v.r)};
}
#endif
#if GREX_X86_64_LEVEL >= 4
/** Widen a binary32 vector to binary64, which is always exact. */
GREX_ALWAYS_INLINE inline NativeVector<f64, 8> f32_to_f64(NativeVector<f32, 8> v) {
  return {.r = _mm512_cvtps_pd(v.r)};
}
#endif
/** Widen a binary32 vector to binary64, which is always exact. */
template<TypedVector<f32> TVec>
requires(is_supernative<f64, size_of<TVec>>)
GREX_ALWAYS_INLINE inline VectorFor<f64, size_of<TVec>> f32_to_f64(TVec v) {
  return {.lower = f32_to_f64(get_low(v)), .upper = f32_to_f64(get_high(v))};
}

/**
 * Convert a binary16 vector into the binary64 vector with the same values, which is always
 * exact.
 */
template<Float16Vector TVec>
GREX_ALWAYS_INLINE inline VectorFor<f64, size_of<TVec>> f16_to_f64(TVec v) {
  return f32_to_f64(f16_to_f32(v));
}
/** Convert a binary64 vector into the binary16 vector with the nearest values, ties to even. */
template<TypedVector<f64> TVec>
GREX_ALWAYS_INLINE inline VectorFor<f16, size_of<TVec>> f64_to_f16(TVec v) {
  return f32_to_f16(f64_to_f32_odd(v));
}
#endif
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_F16_HPP
