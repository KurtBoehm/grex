// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_OPERATIONS_HPP
#define INCLUDE_GREX_OPERATIONS_HPP

#include <limits>

#include "grex/backend/active/operations.hpp"
#include "grex/backend/defs.hpp"
#include "grex/backend/operations.hpp"
#include "grex/base/defs.hpp"
#include "grex/types.hpp"

namespace grex {
#define GREX_MATH_FMA(NAME) \
  template<FloatVectorizable T> \
  inline T NAME(T a, T b, T c) { \
    return backend::NAME(backend::Scalar{a}, backend::Scalar{b}, backend::Scalar{c}).value; \
  }
GREX_MATH_FMA(fmadd)
GREX_MATH_FMA(fmsub)
GREX_MATH_FMA(fnmadd)
GREX_MATH_FMA(fnmsub)
#undef GREX_MATH_FMA

template<FloatVectorizable T>
inline T sqrt(T a) {
  return backend::sqrt(backend::Scalar{a}).value;
}

template<Vectorizable T>
inline T abs(T a) {
  return backend::abs(backend::Scalar{a}).value;
}
template<Vectorizable T>
inline T min(T a, T b) {
  return backend::min(backend::Scalar{a}, backend::Scalar{b}).value;
}
template<Vectorizable T>
inline T max(T a, T b) {
  return backend::max(backend::Scalar{a}, backend::Scalar{b}).value;
}

#define GREX_MATH_MASKARITH(NAME) \
  template<Vectorizable T> \
  inline T NAME(bool mask, T a, T b) { \
    return backend::NAME(mask, backend::Scalar{a}, backend::Scalar{b}).value; \
  }
GREX_MATH_MASKARITH(mask_add)
GREX_MATH_MASKARITH(mask_subtract)
GREX_MATH_MASKARITH(mask_multiply)
GREX_MATH_MASKARITH(mask_divide)
#undef GREX_MATH_MASKARITH

template<Vectorizable T>
inline T blend_zero(bool selector, T v1) {
  return backend::blend_zero(selector, backend::Scalar{v1}).value;
}
template<Vectorizable T>
inline T blend(bool selector, T v0, T v1) {
  return backend::blend(selector, backend::Scalar{v0}, backend::Scalar{v1}).value;
}

template<Vectorizable T>
inline bool is_finite(T a) {
  return backend::is_finite(backend::Scalar{a});
}

// To determine whether a conversion is safe, i.e. guaranteed not to change finite values,
// there are two cases to consider:
// - floating-point → integer: Always unsafe, since max(f32) ≈ 2^128
// - otherwise: digits(TDst) >= digits(TSrc), signed(Dst) || unsigned(Src)
// One of the underlying assumptions is that the number of bits for the mantissa and the exponent
// grow/shrink together, which is true for f32/f64 (there is no support for f16/bf16)
template<typename TDst, typename TSrc>
concept SafeConversion = (!FloatVectorizable<TSrc> || FloatVectorizable<TDst>) &&
                         (SignedVectorizable<TDst> || UnsignedVectorizable<TSrc>) &&
                         std::numeric_limits<TDst>::digits >= std::numeric_limits<TSrc>::digits;

// convert
template<Vectorizable TDst, Vectorizable TSrc, bool tSafe>
requires(!tSafe || SafeConversion<TDst, TSrc>)
inline TDst convert(TSrc src, BoolTag<tSafe> /*tag*/) {
  return TDst(src);
}
template<Vectorizable TDst, AnyVector TSrc, bool tSafe>
requires(!tSafe || SafeConversion<TDst, typename TSrc::Value>)
inline Vector<TDst, TSrc::size> convert(TSrc src, BoolTag<tSafe> /*tag*/) {
  return src.convert(type_tag<TDst>);
}
// Mask conversions are always safe if each entry is filled with 0 or 1
// (which the provided operations ensure)
template<Vectorizable TDst, AnyMask TSrc>
inline Mask<TDst, TSrc::size> convert(TSrc src, AnyBoolTag auto /*tag*/) {
  return src.convert(type_tag<TDst>);
}

template<Vectorizable TDst, typename TSrc>
inline auto convert_unsafe(TSrc src) {
  return convert<TDst>(src, false_tag);
}
template<Vectorizable TDst, typename TSrc>
inline auto convert_safe(TSrc src) {
  return convert<TDst>(src, true_tag);
}

// mask conversions are always safe
template<Vectorizable TDst>
inline bool convert(bool src) {
  return src;
}
template<Vectorizable TDst, AnyMask TSrc>
inline Mask<TDst, TSrc::size> convert(TSrc src) {
  return src.convert(type_tag<TDst>);
}
} // namespace grex

#endif // INCLUDE_GREX_OPERATIONS_HPP
