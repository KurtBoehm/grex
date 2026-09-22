// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_NEON_OPERATIONS_FMADD_FAMILY_HPP
#define INCLUDE_GREX_BACKEND_NEON_OPERATIONS_FMADD_FAMILY_HPP

#include <concepts>
#include <cstddef>

#include <arm_fp16.h>
#include <arm_neon.h>

#include "grex/backend/base.hpp"
#include "grex/backend/defs.hpp" // IWYU pragma: keep
#include "grex/backend/macros/for-each.hpp"
#include "grex/backend/neon/macros/types.hpp"
#include "grex/backend/neon/operations/arithmetic.hpp"
#include "grex/backend/neon/types.hpp"
#include "grex/base.hpp"

#if !GREX_F16_NATIVE_ARITHMETIC
#include "grex/backend/neon/operations/f16.hpp"
#endif

#if GREX_F16_NATIVE_ARITHMETIC
#include "grex/backend/neon/operations/f16.hpp"
#endif

// With FP16, native binary16 FMA is defined together with the other versions, otherwise the
// round-trip through binary32 is used.

namespace grex::backend {
inline constexpr bool has_fma = true;

#define GREX_FMADDF(KIND, BITS, SIZE, TAG, INTRINSIC) \
  GREX_ALWAYS_INLINE inline NativeVector<KIND##BITS, SIZE> fused( \
    NativeVector<KIND##BITS, SIZE> a, NativeVector<KIND##BITS, SIZE> b, \
    NativeVector<KIND##BITS, SIZE> c, TAG) { \
    const auto ar = from_stored<KIND##BITS>(a.r); \
    const auto br = from_stored<KIND##BITS>(b.r); \
    const auto cr = from_stored<KIND##BITS>(c.r); \
    return {.r = to_stored<KIND##BITS>(GREX_ISUFFIXED(INTRINSIC, KIND, BITS)(cr, ar, br))}; \
  }

GREX_FOREACH_FP_TYPE_OPT_EXT(GREX_FMADDF, 128, MultiplyAdd, vfmaq)
GREX_FOREACH_FP_TYPE_OPT_EXT(GREX_FMADDF, 128, NegatedMultiplyAdd, vfmsq)

template<NativeFloatVectorizable T, std::size_t tSize>
GREX_ALWAYS_INLINE inline NativeVector<T, tSize>
fused(NativeVector<T, tSize> a, NativeVector<T, tSize> b, NativeVector<T, tSize> c,
      MultiplySubtract /*tag*/) {
  return fused(a, b, negate(c), MultiplyAdd{});
}
template<NativeFloatVectorizable T, std::size_t tSize>
GREX_ALWAYS_INLINE inline NativeVector<T, tSize>
fused(NativeVector<T, tSize> a, NativeVector<T, tSize> b, NativeVector<T, tSize> c,
      NegatedMultiplySubtract /*tag*/) {
  return fused(a, b, negate(c), NegatedMultiplyAdd{});
}

template<typename THalf>
GREX_ALWAYS_INLINE inline SuperVector<THalf> fused(SuperVector<THalf> a, SuperVector<THalf> b,
                                                   SuperVector<THalf> c, FusedTag auto tag) {
  return {
    .lower = fused(a.lower, b.lower, c.lower, tag),
    .upper = fused(a.upper, b.upper, c.upper, tag),
  };
}
template<NativeFloatVectorizable T, std::size_t tSize>
GREX_ALWAYS_INLINE inline SubVector<T, tSize> fused(SubVector<T, tSize> a, SubVector<T, tSize> b,
                                                    SubVector<T, tSize> c, FusedTag auto tag) {
  return SubVector<T, tSize>{fused(a.full, b.full, c.full, tag)};
}

#if !GREX_F16_NATIVE_ARITHMETIC
// Binary16 without FP16: round-trip through binary32.
template<Float16Vector TVec>
GREX_ALWAYS_INLINE inline TVec fused(TVec a, TVec b, TVec c, FusedTag auto tag) {
  return f32_to_f16(fused(f16_to_f32(a), f16_to_f32(b), f16_to_f32(c), tag));
}
#endif

#define GREX_FFMA(TYPE, BSUFFIX) \
  template<std::same_as<TYPE> T> \
  GREX_ALWAYS_INLINE inline T fused(T a, T b, T c, MultiplyAdd) { \
    return __builtin_fma##BSUFFIX(a, b, c); \
  } \
  template<std::same_as<TYPE> T> \
  GREX_ALWAYS_INLINE inline T fused(T a, T b, T c, MultiplySubtract) { \
    return __builtin_fma##BSUFFIX(a, b, -c); \
  } \
  template<std::same_as<TYPE> T> \
  GREX_ALWAYS_INLINE inline T fused(T a, T b, T c, NegatedMultiplyAdd) { \
    return __builtin_fma##BSUFFIX(-a, b, c); \
  } \
  template<std::same_as<TYPE> T> \
  GREX_ALWAYS_INLINE inline T fused(T a, T b, T c, NegatedMultiplySubtract) { \
    return __builtin_fma##BSUFFIX(-a, b, -c); \
  }

GREX_FFMA(f64, )
GREX_FFMA(f32, f)

#if GREX_F16_NATIVE_ARITHMETIC
GREX_FFMA(f16, f16)
#else
template<std::same_as<f16> T>
GREX_ALWAYS_INLINE inline T fused(T a, T b, T c, FusedTag auto tag) {
  return grex::f32_to_f16(
    fused(grex::f16_to_f32(a), grex::f16_to_f32(b), grex::f16_to_f32(c), tag));
}
#endif
#undef GREX_FFMA
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_FMADD_FAMILY_HPP
