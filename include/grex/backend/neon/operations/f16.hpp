// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_NEON_OPERATIONS_F16_HPP
#define INCLUDE_GREX_BACKEND_NEON_OPERATIONS_F16_HPP

#include <bit>
#include <concepts>
#include <cstddef>

#include "grex/backend/base.hpp"
#include "grex/backend/choosers.hpp"
#include "grex/backend/neon/operations/expand64.hpp"
#include "grex/backend/neon/operations/merge.hpp"
#include "grex/backend/neon/types.hpp"
#include "grex/base.hpp"

namespace grex::backend {
// Binary16: Reinterpret to/from `u16`.
inline float16x4_t as_f16(uint16x4_t r) {
  return vreinterpret_f16_u16(r);
}
inline float16x8_t as_f16(uint16x8_t r) {
  return vreinterpretq_f16_u16(r);
}

inline uint16x4_t as_u16(float16x4_t r) {
  return vreinterpret_u16_f16(r);
}
inline uint16x8_t as_u16(float16x8_t r) {
  return vreinterpretq_u16_f16(r);
}

// Binary16 ↔ binary32: ARM64 always provides the FCVTL/FCVTN instructions, so the software fallback
// in grex/backend/shared/operations/convert.hpp is never needed here.

/** Widen a binary16 vector into the binary32 vector. */
GREX_ALWAYS_INLINE inline VectorFor<f32, 8> f16_to_f32(NativeVector<f16, 8> v) {
  const float16x8_t half = vreinterpretq_f16_u16(v.r);
  return {
    .lower = {.r = vcvt_f32_f16(vget_low_f16(half))},
    .upper = {.r = vcvt_high_f32_f16(half)},
  };
}
/** Widen a binary16 vector into the binary32 vector. */
template<std::size_t tSize>
requires(std::has_single_bit(tSize) && tSize < 8)
GREX_ALWAYS_INLINE inline VectorFor<f32, tSize> f16_to_f32(SubVector<f16, tSize> v) {
  const float16x4_t half = vget_low_f16(vreinterpretq_f16_u16(v.full.r));
  return VectorFor<f32, tSize>{vcvt_f32_f16(half)};
}

/** Convert a binary32 vector into the binary16 vector, rounding ties to even. */
GREX_ALWAYS_INLINE inline NativeVector<f16, 8> f32_to_f16(VectorFor<f32, 8> v) {
  const float16x4_t lower = vcvt_f16_f32(v.lower.r);
  return {.r = vreinterpretq_u16_f16(vcvt_high_f16_f32(lower, v.upper.r))};
}
/** Convert a binary32 vector into the binary16 vector, rounding ties to even. */
GREX_ALWAYS_INLINE inline SubVector<f16, 4> f32_to_f16(NativeVector<f32, 4> v) {
  return SubVector<f16, 4>{vreinterpretq_u16_f16(expand64(vcvt_f16_f32(v.r)))};
}
/** Convert a binary32 vector into the binary16 vector, rounding ties to even. */
GREX_ALWAYS_INLINE inline SubVector<f16, 2> f32_to_f16(SubVector<f32, 2> v) {
  return SubVector<f16, 2>{vreinterpretq_u16_f16(expand64(vcvt_f16_f32(v.full.r)))};
}

// Binary16 ↔ binary64: ARM64 has no instruction for either direction, so binary64 is reached
// through binary32, reusing the conversions above rather than repeating their bit manipulation for
// the wider format. Narrowing twice with round-to-nearest would round twice, which no amount of
// intermediate precision repairs for an arbitrary binary64 input, so the narrowing step rounds to
// odd instead and the binary16 rounding that follows is then correctly rounded: A binary16 rounding
// boundary is a midpoint between two binary16 values and therefore has an even significand in every
// format with at least p + 2 = 13 bits, so an odd intermediate can neither land on such a boundary
// nor, being a neighbour of the input, cross one. Binary32 offers 24 bits, well above that bound.

namespace f16_convert {
// A binary64 significand stores 52 bits and a binary32 one 23, so narrowing discards the low 29.
inline constexpr u64 f64_discarded = (u64{1} << 29) - 1;

/**
 * Round the binary64 lanes of `v` to the significand of binary32, breaking ties towards an odd
 * significand: Adding the mask of the discarded bits to those bits carries into the lowest
 * surviving bit exactly if any of them is set, so the disjunction forces that bit to one precisely
 * if the narrowing loses information.
 *
 * Infinities and not-a-numbers survive this unchanged, the latter because the sticky bit keeps a
 * significand whose surviving bits are all zero from turning into an infinity.
 */
GREX_ALWAYS_INLINE inline float64x2_t round_odd(float64x2_t v) {
  const uint64x2_t bits = vreinterpretq_u64_f64(v);
  const uint64x2_t mask = vdupq_n_u64(f64_discarded);
  const uint64x2_t sticky =
    vandq_u64(vaddq_u64(vandq_u64(bits, mask), mask), vdupq_n_u64(f64_discarded + 1));
  return vreinterpretq_f64_u64(vorrq_u64(vbicq_u64(bits, mask), sticky));
}
} // namespace f16_convert

/** Narrow a binary64 vector to binary32, rounding to odd (see `f16_convert::round_odd`). */
GREX_ALWAYS_INLINE inline VectorFor<f32, 2> f64_to_f32_odd(NativeVector<f64, 2> v) {
  return VectorFor<f32, 2>{expand64(vcvt_f32_f64(f16_convert::round_odd(v.r)))};
}
/** Narrow a binary64 vector to binary32, rounding to odd (see `f16_convert::round_odd`). */
GREX_ALWAYS_INLINE inline NativeVector<f32, 4> f64_to_f32_odd(SuperVector<NativeVector<f64, 2>> v) {
  const float32x2_t lower = vcvt_f32_f64(f16_convert::round_odd(v.lower.r));
  return {.r = vcvt_high_f32_f64(lower, f16_convert::round_odd(v.upper.r))};
}
/** Narrow a binary64 vector to binary32, rounding to odd (see `f16_convert::round_odd`). */
template<TypedVector<f64> THalf>
GREX_ALWAYS_INLINE inline VectorFor<f32, 2 * size_of<THalf>> f64_to_f32_odd(SuperVector<THalf> v) {
  return merge(f64_to_f32_odd(v.lower), f64_to_f32_odd(v.upper));
}

/** Widen a binary32 vector to binary64, which is always exact. */
GREX_ALWAYS_INLINE inline NativeVector<f64, 2> f32_to_f64(SubVector<f32, 2> v) {
  return {.r = vcvt_f64_f32(vget_low_f32(v.full.r))};
}
/** Widen a binary32 vector to binary64, which is always exact. */
GREX_ALWAYS_INLINE inline VectorFor<f64, 4> f32_to_f64(NativeVector<f32, 4> v) {
  return {
    .lower = {.r = vcvt_f64_f32(vget_low_f32(v.r))},
    .upper = {.r = vcvt_high_f64_f32(v.r)},
  };
}
/** Widen a binary32 vector to binary64, which is always exact. */
template<TypedVector<f32> THalf>
GREX_ALWAYS_INLINE inline VectorFor<f64, 2 * THalf::size> f32_to_f64(SuperVector<THalf> v) {
  return {.lower = f32_to_f64(v.lower), .upper = f32_to_f64(v.upper)};
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

/**
 * Convert a Neon type to its stored type, which maps `f16` to `u16` and returns the input unchanged
 * otherwise.
 */
template<typename TValue, typename TVec>
GREX_ALWAYS_INLINE inline auto to_stored(TVec v) {
  if constexpr (std::same_as<TValue, f16>) {
    return as_u16(v);
  } else {
    return v;
  }
}

/**
 * Convert a Neon type from its stored type, which maps `u16` to `f16` if `TValue` is `f16` and
 * returns the input unchanged otherwise.
 */
template<typename TValue, typename TVec>
GREX_ALWAYS_INLINE inline auto from_stored(TVec v) {
  if constexpr (std::same_as<TValue, f16>) {
    return as_f16(v);
  } else {
    return v;
  }
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_F16_HPP
