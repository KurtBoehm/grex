// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_OPERATIONS_HPP
#define INCLUDE_GREX_OPERATIONS_HPP

#include <concepts>

#include "grex/backend/active/operations.hpp"
#include "grex/backend/base.hpp"
#include "grex/backend/defs.hpp" // IWYU pragma: keep
#include "grex/backend/operations.hpp"
#include "grex/base.hpp"

#if !GREX_BACKEND_SCALAR
#include "grex/types.hpp"
#endif

namespace grex {
/**
  Indicates whether the backend supports fused multiply-add.

  If `false`, fused multiply-addition is emulated:
  - `f64`/`f32`: multiplication, addition/subtraction, and negation (if required).
  - `f16`: widen to `f32`, perform `f32` FMA (which may be emulated), and round back.
*/
template<FloatVectorizable T>
inline constexpr bool has_fma = std::same_as<T, f16> ? backend::has_f16_fma : backend::has_fma;

template<IntVectorizable TDst>
inline TDst expand_any(IntVectorizable auto value) {
  return backend::expand_any<TDst>(value);
}
template<UnsignedIntVectorizable T>
inline bool bit_test(T a, T b) {
  return backend::bit_test(a, b);
}

inline bool andnot(bool a, bool b) {
  return backend::logical_andnot(a, b);
}

#define GREX_MATH_FMA(NAME, TAG) \
  template<FloatVectorizable T> \
  inline T NAME(T a, T b, T c) { \
    return backend::fused(a, b, c, backend::TAG{}); \
  }
GREX_MATH_FMA(fmadd, MultiplyAdd)
GREX_MATH_FMA(fmsub, MultiplySubtract)
GREX_MATH_FMA(fnmadd, NegatedMultiplyAdd)
GREX_MATH_FMA(fnmsub, NegatedMultiplySubtract)
#undef GREX_MATH_FMA

template<FloatVectorizable T>
inline T sqrt(T a) {
  return backend::sqrt(a);
}

template<Vectorizable T>
inline T abs(T a) {
  return backend::abs(a);
}
template<Vectorizable T>
inline T min(T a, T b) {
  return backend::min(a, b);
}
template<Vectorizable T>
inline T max(T a, T b) {
  return backend::max(a, b);
}

#define GREX_MATH_MASKARITH(NAME) \
  template<Vectorizable T> \
  inline T NAME(bool mask, T a, T b) { \
    return backend::NAME(mask, a, b); \
  }
GREX_MATH_MASKARITH(mask_add)
GREX_MATH_MASKARITH(mask_subtract)
GREX_MATH_MASKARITH(mask_multiply)
GREX_MATH_MASKARITH(mask_divide)
#undef GREX_MATH_MASKARITH

template<Vectorizable T>
inline T extract_single(T v) {
  return backend::extract_single(v);
}
template<Vectorizable T>
inline T blend_zero(bool selector, T v1) {
  return backend::blend_zero(selector, v1);
}
template<Vectorizable T>
inline T blend(bool selector, T v0, T v1) {
  return backend::blend(selector, v0, v1);
}

template<FloatVectorizable T>
inline bool is_finite(T a) {
  return backend::is_finite(a);
}
template<FloatVectorizable T>
inline T make_finite(T a) {
  return backend::make_finite(a);
}

// To determine whether a conversion is safe, i.e. guaranteed not to change finite values,
// there are two cases to consider:
// - floating-point → integer: Always unsafe, since max(f32) ≈ 2^128
// - otherwise: digits(TDst) >= digits(TSrc), signed(Dst) || unsigned(Src)
// One of the underlying assumptions is that the number of bits for the mantissa and the exponent
// grow/shrink together, which is true for f16/f32/f64 (there is no support for bf16, whose exponent
// is as wide as that of f32 while its mantissa is narrower than that of f16). `grex::NumericTrait`
// is used instead of `std::numeric_limits`, which is not specialized for `f16`.
template<typename TDst, typename TSrc>
concept SafeConversion = (!FloatVectorizable<TSrc> || FloatVectorizable<TDst>) &&
                         (SignedVectorizable<TDst> || UnsignedVectorizable<TSrc>) &&
                         NumericTrait<TDst>::digits >= NumericTrait<TSrc>::digits;

// convert
template<Vectorizable TDst, Vectorizable TSrc>
inline TDst convert(TSrc src) {
  return TDst(src);
}
template<Vectorizable TDst, Vectorizable TSrc, bool tSafe>
requires(!tSafe || SafeConversion<TDst, TSrc>)
inline TDst convert(TSrc src, CastTag<tSafe> /*tag*/) {
  return convert<TDst>(src);
}

#if !GREX_BACKEND_SCALAR
template<Vectorizable TDst, AnyVector TSrc>
inline Vector<TDst, TSrc::size> convert(TSrc src) {
  return src.convert(type_tag<TDst>);
}
template<Vectorizable TDst, AnyVector TSrc, bool tSafe>
requires(!tSafe || SafeConversion<TDst, typename TSrc::Value>)
inline Vector<TDst, TSrc::size> convert(TSrc src, CastTag<tSafe> /*tag*/) {
  return src.convert(type_tag<TDst>);
}

// Mask conversions are always safe if each entry is filled with 0 or 1
// (which the provided operations ensure)
template<Vectorizable TDst>
inline bool convert(bool src) {
  return src;
}
template<Vectorizable TDst>
inline bool convert(bool src, AnyBoolTag auto /*tag*/) {
  return src;
}

template<Vectorizable TDst, AnyMask TSrc>
inline Mask<TDst, TSrc::size> convert(TSrc src) {
  return src.convert(type_tag<TDst>);
}
template<Vectorizable TDst, AnyMask TSrc>
inline Mask<TDst, TSrc::size> convert(TSrc src, AnyBoolTag auto /*tag*/) {
  return src.convert(type_tag<TDst>);
}
#endif

GREX_ALWAYS_INLINE inline auto add(auto... values) {
  return backend::nary_add(values...);
}
GREX_ALWAYS_INLINE inline auto subtract(auto... values) {
  return backend::nary_subtract(values...);
}

#if GREX_BACKEND_X86_64
using backend::runtime_x86_64_level;
#endif
} // namespace grex

#endif // INCLUDE_GREX_OPERATIONS_HPP
