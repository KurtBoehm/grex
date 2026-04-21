// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_EXPAND_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_EXPAND_HPP

#include <cstddef>

#include <immintrin.h>

#include "grex/backend/base.hpp"
#include "grex/backend/choosers.hpp"
#include "grex/backend/macros/base.hpp"
#include "grex/backend/macros/math.hpp"
#include "grex/backend/shared/operations/expand.hpp" // IWYU pragma: export
#include "grex/backend/x86/instruction-sets.hpp"
#include "grex/backend/x86/macros/intrinsics.hpp"
#include "grex/backend/x86/operations/merge.hpp"
#include "grex/backend/x86/operations/set.hpp"
#include "grex/backend/x86/types.hpp"
#include "grex/base.hpp"

#if GREX_X86_64_LEVEL >= 3
#include "grex/backend/macros/for-each.hpp"
#endif

namespace grex::backend {
////////////
// Scalar //
////////////

template<bool tZero>
inline f32x4 expand(Scalar<f32> x, IndexTag<4> /*tag*/, BoolTag<tZero> /*tag*/) {
  if constexpr (!tZero) {
#if GREX_GCC
    __m128 retval;
    asm("" : "=x"(retval) : "0"(x.value));
    return {.r = retval};
#elif GREX_CLANG
    f32 data[4];
    data[0] = x.value;
    return {.r = _mm_load_ps(static_cast<const f32*>(data))};
#endif
  }
  return {.r = _mm_set_ss(x.value)};
}
template<bool tZero>
inline f64x2 expand(Scalar<f64> x, IndexTag<2> /*tag*/, BoolTag<tZero> /*tag*/) {
  if constexpr (!tZero) {
#if GREX_GCC
    __m128d retval;
    asm("" : "=x"(retval) : "0"(x.value));
    return {.r = retval};
#elif GREX_CLANG
    f64 data[2];
    data[0] = x.value;
    return {.r = _mm_load_pd(static_cast<const f64*>(data))};
#endif
  }
  return {.r = _mm_set_sd(x.value)};
}
// Integers with at most 32 bits: Cast to i32
template<IntVectorizable T, bool tZero>
requires(sizeof(T) <= 4)
inline NativeVector<T, min_native_size<T>> expand(Scalar<T> x, IndexTag<min_native_size<T>> /*tag*/,
                                                  BoolTag<tZero> /*tag*/) {
  // force zero extension
  using Unsigned = UnsignedOf<T>;
  return {.r = _mm_cvtsi32_si128(i32(Unsigned(x.value)))};
}
// Integers with 64 bits: Cast to i64
template<IntVectorizable T, bool tZero>
requires(sizeof(T) == 8)
inline NativeVector<T, 2> expand(Scalar<T> x, IndexTag<2> /*tag*/, BoolTag<tZero> /*tag*/) {
  return {.r = _mm_cvtsi64_si128(i64(x.value))};
}

////////////
// Vector //
////////////

// native → native: use cast/zext intrinsics
#define GREX_EXPANDV_INTRINSIC(KIND, BITS, DSTSIZE, SRCRBITS, DSTRBITS) \
  inline NativeVector<KIND##BITS, DSTSIZE> expand( \
    NativeVector<KIND##BITS, GREX_DIVIDE(SRCRBITS, BITS)> v, IndexTag<DSTSIZE>, BoolTag<false>) { \
    return {.r = GREX_CAT(GREX_BITPREFIX(DSTRBITS), _cast, GREX_SIR_SUFFIX(KIND, BITS, SRCRBITS), \
                          _, GREX_SIR_SUFFIX(KIND, BITS, DSTRBITS))(v.r)}; \
  } \
  inline NativeVector<KIND##BITS, DSTSIZE> expand( \
    NativeVector<KIND##BITS, GREX_DIVIDE(SRCRBITS, BITS)> v, IndexTag<DSTSIZE>, BoolTag<true>) { \
    return {.r = GREX_CAT(GREX_BITPREFIX(DSTRBITS), _zext, GREX_SIR_SUFFIX(KIND, BITS, SRCRBITS), \
                          _, GREX_SIR_SUFFIX(KIND, BITS, DSTRBITS))(v.r)}; \
  }
#if GREX_X86_64_LEVEL >= 3
GREX_FOREACH_TYPE(GREX_EXPANDV_INTRINSIC, 256, 128, 256)
#endif
#if GREX_X86_64_LEVEL >= 4
GREX_FOREACH_TYPE(GREX_EXPANDV_INTRINSIC, 512, 128, 512)
GREX_FOREACH_TYPE(GREX_EXPANDV_INTRINSIC, 512, 256, 512)
#endif

// native/super-native → super-native
template<AnyVector TVec, std::size_t tDstSize, bool tZero>
requires(tDstSize > TVec::size && is_supernative<typename TVec::Value, tDstSize> &&
         (AnyNativeVector<TVec> || AnySuperNativeVector<TVec>))
inline VectorFor<typename TVec::Value, tDstSize> expand(TVec v, IndexTag<tDstSize> /*size*/,
                                                        BoolTag<tZero> zero_tag) {
  using Value = TVec::Value;
  using Half = VectorFor<Value, tDstSize / 2>;
  if constexpr (tZero) {
    return merge(expand(v, index_tag<Half::size>, zero_tag), zeros(type_tag<Half>));
  } else {
    return merge(expand(v, index_tag<Half::size>, zero_tag), undefined(type_tag<Half>));
  }
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_EXPAND_HPP
