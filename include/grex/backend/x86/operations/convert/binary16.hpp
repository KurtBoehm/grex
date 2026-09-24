// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_CONVERT_BINARY16_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_CONVERT_BINARY16_HPP

#include <cstddef>

#include <immintrin.h>

#include "grex/backend/base.hpp"
#include "grex/backend/choosers.hpp"
#include "grex/backend/defs.hpp"
#include "grex/backend/x86/instruction-sets.hpp"
#include "grex/backend/x86/types.hpp" // IWYU pragma: keep
#include "grex/base.hpp"

#if GREX_F16_NATIVE_ARITHMETIC
#include "grex/backend/macros/math.hpp"
#include "grex/backend/x86/operations/intrinsics.hpp"
#include "grex/backend/x86/operations/split.hpp"
#else
#include <concepts>
#endif

namespace grex::backend {
#if GREX_F16_NATIVE_ARITHMETIC
//==================================================================================================
// Binary16 with AVX512-FP16: direct conversions between binary16 and every other numeric type
//==================================================================================================

//--------------------------------------------------------------------------------------------------
// Binary16 ↔ 16-bit integers
//--------------------------------------------------------------------------------------------------

#define GREX_F16_CVT16(REGISTERBITS, BITPREFIX, KIND, SIGN) \
  inline NativeVector<KIND##16, GREX_DIVIDE(REGISTERBITS, 16)> convert( \
    NativeVector<f16, GREX_DIVIDE(REGISTERBITS, 16)> v, TypeTag<KIND##16>) { \
    return {.r = BITPREFIX##_cvttph_e##SIGN##16(BITPREFIX##_castsi##REGISTERBITS##_ph(v.r))}; \
  } \
  inline NativeVector<f16, GREX_DIVIDE(REGISTERBITS, 16)> convert( \
    NativeVector<KIND##16, GREX_DIVIDE(REGISTERBITS, 16)> v, TypeTag<f16> /*tag*/) { \
    return {.r = BITPREFIX##_castph_si##REGISTERBITS(BITPREFIX##_cvte##SIGN##16_ph(v.r))}; \
  }
GREX_F16_CVT16(128, _mm, i, pi)
GREX_F16_CVT16(128, _mm, u, pu)
GREX_F16_CVT16(256, _mm256, i, pi)
GREX_F16_CVT16(256, _mm256, u, pu)
GREX_F16_CVT16(512, _mm512, i, pi)
GREX_F16_CVT16(512, _mm512, u, pu)
#undef GREX_F16_CVT16

#define GREX_F16_CVT16_SUB(PART, KIND, SIGN) \
  inline SubVector<KIND##16, PART> convert(SubVector<f16, PART> v, TypeTag<KIND##16>) { \
    return SubVector<KIND##16, PART>{_mm_cvttph_e##SIGN##16(_mm_castsi128_ph(v.full.r))}; \
  } \
  inline SubVector<f16, PART> convert(SubVector<KIND##16, PART> v, TypeTag<f16> /*tag*/) { \
    return SubVector<f16, PART>{_mm_castph_si128(_mm_cvte##SIGN##16_ph(v.full.r))}; \
  }
GREX_F16_CVT16_SUB(2, i, pi)
GREX_F16_CVT16_SUB(2, u, pu)
GREX_F16_CVT16_SUB(4, i, pi)
GREX_F16_CVT16_SUB(4, u, pu)
#undef GREX_F16_CVT16_SUB

//--------------------------------------------------------------------------------------------------
// Binary16 ↔ 32-bit values
//--------------------------------------------------------------------------------------------------

template<std::size_t N>
inline VectorFor<f32, N> convert(SubVector<f16, N> v, TypeTag<f32> /*tag*/) {
  return VectorFor<f32, N>{_mm_cvtph_ps(v.full.r)};
}
template<std::size_t N>
inline VectorFor<i32, N> convert(SubVector<f16, N> v, TypeTag<i32> /*tag*/) {
  return VectorFor<i32, N>{_mm_cvttph_epi32(_mm_castsi128_ph(v.full.r))};
}
template<std::size_t N>
inline VectorFor<u32, N> convert(SubVector<f16, N> v, TypeTag<u32> /*tag*/) {
  return VectorFor<u32, N>{_mm_cvttph_epu32(_mm_castsi128_ph(v.full.r))};
}

inline SubVector<f16, 4> convert(f32x4 v, TypeTag<f16> /*tag*/) {
  return SubVector<f16, 4>{mm::cvtps_ph(v.r)};
}
inline SubVector<f16, 4> convert(i32x4 v, TypeTag<f16> /*tag*/) {
  return SubVector<f16, 4>{_mm_castph_si128(_mm_cvtepi32_ph(v.r))};
}
inline SubVector<f16, 4> convert(u32x4 v, TypeTag<f16> /*tag*/) {
  return SubVector<f16, 4>{_mm_castph_si128(_mm_cvtepu32_ph(v.r))};
}

inline f32x8 convert(f16x8 v, TypeTag<f32> /*tag*/) {
  return {.r = _mm256_cvtph_ps(v.r)};
}
inline i32x8 convert(f16x8 v, TypeTag<i32> /*tag*/) {
  return {.r = _mm256_cvttph_epi32(_mm_castsi128_ph(v.r))};
}
inline u32x8 convert(f16x8 v, TypeTag<u32> /*tag*/) {
  return {.r = _mm256_cvttph_epu32(_mm_castsi128_ph(v.r))};
}

inline f16x8 convert(f32x8 v, TypeTag<f16> /*tag*/) {
  return {.r = mm256::cvtps_ph(v.r)};
}
inline f16x8 convert(i32x8 v, TypeTag<f16> /*tag*/) {
  return {.r = _mm_castph_si128(_mm256_cvtepi32_ph(v.r))};
}
inline f16x8 convert(u32x8 v, TypeTag<f16> /*tag*/) {
  return {.r = _mm_castph_si128(_mm256_cvtepu32_ph(v.r))};
}

inline f32x16 convert(f16x16 v, TypeTag<f32> /*tag*/) {
  return {.r = _mm512_cvtph_ps(v.r)};
}
inline i32x16 convert(f16x16 v, TypeTag<i32> /*tag*/) {
  return {.r = _mm512_cvttph_epi32(_mm256_castsi256_ph(v.r))};
}
inline u32x16 convert(f16x16 v, TypeTag<u32> /*tag*/) {
  return {.r = _mm512_cvttph_epu32(_mm256_castsi256_ph(v.r))};
}

inline f16x16 convert(f32x16 v, TypeTag<f16> /*tag*/) {
  return {.r = mm512::cvtps_ph(v.r)};
}
inline f16x16 convert(i32x16 v, TypeTag<f16> /*tag*/) {
  return {.r = _mm256_castph_si256(_mm512_cvtepi32_ph(v.r))};
}
inline f16x16 convert(u32x16 v, TypeTag<f16> /*tag*/) {
  return {.r = _mm256_castph_si256(_mm512_cvtepu32_ph(v.r))};
}

//--------------------------------------------------------------------------------------------------
// Binary16 ↔ 64-bit values
//--------------------------------------------------------------------------------------------------

#define GREX_F16_CVT64_LOWN(N, REGISTERBITS, BITPREFIX) \
  inline NativeVector<i64, N> convert(SubVector<f16, N> v, TypeTag<i64> /*tag*/) { \
    return {.r = BITPREFIX##_cvttph_epi64(_mm_castsi128_ph(v.full.r))}; \
  } \
  inline NativeVector<u64, N> convert(SubVector<f16, N> v, TypeTag<u64> /*tag*/) { \
    return {.r = BITPREFIX##_cvttph_epu64(_mm_castsi128_ph(v.full.r))}; \
  } \
  inline SubVector<f16, N> convert(NativeVector<i64, N> v, TypeTag<f16> /*tag*/) { \
    return SubVector<f16, N>{_mm_castph_si128(BITPREFIX##_cvtepi64_ph(v.r))}; \
  } \
  inline SubVector<f16, N> convert(NativeVector<u64, N> v, TypeTag<f16> /*tag*/) { \
    return SubVector<f16, N>{_mm_castph_si128(BITPREFIX##_cvtepu64_ph(v.r))}; \
  } \
  inline NativeVector<f64, N> convert(SubVector<f16, N> v, TypeTag<f64> /*tag*/) { \
    return {.r = BITPREFIX##_cvtph_pd(_mm_castsi128_ph(v.full.r))}; \
  } \
  inline SubVector<f16, N> convert(NativeVector<f64, N> v, TypeTag<f16> /*tag*/) { \
    return SubVector<f16, N>{_mm_castph_si128(BITPREFIX##_cvtpd_ph(v.r))}; \
  }
GREX_F16_CVT64_LOWN(2, 128, _mm)
GREX_F16_CVT64_LOWN(4, 256, _mm256)
#undef GREX_F16_CVT64_LOWN

inline f64x8 convert(f16x8 v, TypeTag<f64> /*tag*/) {
  return {.r = _mm512_cvtph_pd(_mm_castsi128_ph(v.r))};
}
inline i64x8 convert(f16x8 v, TypeTag<i64> /*tag*/) {
  return {.r = _mm512_cvttph_epi64(_mm_castsi128_ph(v.r))};
}
inline u64x8 convert(f16x8 v, TypeTag<u64> /*tag*/) {
  return {.r = _mm512_cvttph_epu64(_mm_castsi128_ph(v.r))};
}

inline f16x8 convert(f64x8 v, TypeTag<f16> /*tag*/) {
  return {.r = _mm_castph_si128(_mm512_cvtpd_ph(v.r))};
}
inline f16x8 convert(i64x8 v, TypeTag<f16> /*tag*/) {
  return {.r = _mm_castph_si128(_mm512_cvtepi64_ph(v.r))};
}
inline f16x8 convert(u64x8 v, TypeTag<f16> /*tag*/) {
  return {.r = _mm_castph_si128(_mm512_cvtepu64_ph(v.r))};
}

//==================================================================================================
// General conversion rules
//==================================================================================================

// Native binary16 → super-native: split the input vector and go from there.
template<Float16Vector Src, Vectorizable Dst>
requires(AnyNativeVector<Src> && is_supernative<Dst, size_of<Src>>)
inline VectorFor<Dst, size_of<Src>> convert(Src v, TypeTag<Dst> /*tag*/) {
  return {.lower = convert(get_low(v)), .upper = convert(get_high(v))};
}

// Binary16 → 8-bit integers: convert with 16-bit integers as the intermediary.
template<Float16Vector Src, Int8 Dst>
inline VectorFor<Dst, size_of<Src>> convert(Src v, TypeTag<Dst> tag) {
  return convert(convert(v, type_tag<CopySignInt<Dst, 2>>), tag);
}

// 8-bit integers → native binary16: convert with 16-bit integers as the intermediary.
template<Int8Vector Src>
requires(is_native<f16, size_of<Src>>)
inline VectorFor<f16, size_of<Src>> convert(Src v, TypeTag<f16> tag) {
  return convert(convert(v, type_tag<CopySignInt<ValueOf<Src>, 2>>), tag);
}
#else
//==================================================================================================
// Binary16 with AVX512-FP16: direct conversions between binary16 and every other numeric type
//==================================================================================================

//--------------------------------------------------------------------------------------------------
// Binary16 ↔ binary32: native with F16C, emulated otherwise
//--------------------------------------------------------------------------------------------------
// These overloads fix both the source and the destination value type, which makes them more
// specialized than the generic conversions above and thus unambiguously preferred.

template<std::size_t N>
GREX_ALWAYS_INLINE inline VectorFor<f32, N> convert(NativeVector<f16, N> v, TypeTag<f32> /*tag*/) {
  return f16_to_f32(v);
}
template<std::size_t N>
GREX_ALWAYS_INLINE inline VectorFor<f32, N> convert(SubVector<f16, N> v, TypeTag<f32> /*tag*/) {
  return f16_to_f32(v);
}

template<std::size_t N>
GREX_ALWAYS_INLINE inline VectorFor<f16, N> convert(NativeVector<f32, N> v, TypeTag<f16> /*tag*/) {
  return f32_to_f16(v);
}
template<std::size_t N>
GREX_ALWAYS_INLINE inline VectorFor<f16, N> convert(SubVector<f32, N> v, TypeTag<f16> /*tag*/) {
  return f32_to_f16(v);
}
template<typename Half>
requires(std::same_as<ValueOf<Half>, f32> && !is_supernative<f16, 2 * Half::size>)
GREX_ALWAYS_INLINE inline VectorFor<f16, 2 * Half::size> convert(SuperVector<Half> v,
                                                                 TypeTag<f16> /*tag*/) {
  return f32_to_f16(v);
}

//--------------------------------------------------------------------------------------------------
// Binary16 ↔ any type other than binary32: routed through binary32
//--------------------------------------------------------------------------------------------------
// Binary64 is the exception: `f16_to_f64`/`f64_to_f16` also pass through binary32, but round only
// once, whereas two plain conversions would round twice.

template<Vectorizable Dst, std::size_t N>
requires(!std::same_as<Dst, f16>)
GREX_ALWAYS_INLINE inline VectorFor<Dst, N> convert(NativeVector<f16, N> v, TypeTag<Dst> tag) {
  if constexpr (std::same_as<Dst, f64>) {
    return f16_to_f64(v);
  } else {
    return convert(f16_to_f32(v), tag);
  }
}
template<Vectorizable Dst, std::size_t N>
requires(!std::same_as<Dst, f16>)
GREX_ALWAYS_INLINE inline VectorFor<Dst, N> convert(SubVector<f16, N> v, TypeTag<Dst> tag) {
  if constexpr (std::same_as<Dst, f64>) {
    return f16_to_f64(v);
  } else {
    return convert(f16_to_f32(v), tag);
  }
}

template<typename Half>
requires(!std::same_as<ValueOf<Half>, f16> && !std::same_as<ValueOf<Half>, f32> &&
         !is_supernative<f16, 2 * Half::size>)
GREX_ALWAYS_INLINE inline VectorFor<f16, 2 * Half::size> convert(SuperVector<Half> v,
                                                                 TypeTag<f16> tag) {
  if constexpr (std::same_as<ValueOf<Half>, f64>) {
    return f64_to_f16(v);
  } else {
    return convert(convert(v, type_tag<f32>), tag);
  }
}

template<Vectorizable Src, std::size_t N>
requires(!std::same_as<Src, f16> && !std::same_as<Src, f32>)
GREX_ALWAYS_INLINE inline VectorFor<f16, N> convert(NativeVector<Src, N> v, TypeTag<f16> tag) {
  if constexpr (std::same_as<Src, f64>) {
    return f64_to_f16(v);
  } else {
    return convert(convert(v, type_tag<f32>), tag);
  }
}
template<Vectorizable Src, std::size_t N>
requires(!std::same_as<Src, f16> && !std::same_as<Src, f32>)
GREX_ALWAYS_INLINE inline VectorFor<f16, N> convert(SubVector<Src, N> v, TypeTag<f16> tag) {
  if constexpr (std::same_as<Src, f64>) {
    return f64_to_f16(v);
  } else {
    return convert(convert(v, type_tag<f32>), tag);
  }
}

template<IntVectorizable Dst, Float16Vector Half>
requires(!is_supernative<Dst, 2 * Half::size>)
GREX_ALWAYS_INLINE inline VectorFor<Dst, 2 * Half::size> convert(SuperVector<Half> v,
                                                                 TypeTag<Dst> tag) {
  return convert(convert(v, type_tag<f32>), tag);
}
#endif
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_CONVERT_BINARY16_HPP
