// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_EXPAND_SCALAR_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_EXPAND_SCALAR_HPP

#include <concepts>
#include <cstddef>

#include <immintrin.h>

#include "grex/backend/choosers.hpp"
#include "grex/backend/defs.hpp"
#include "grex/backend/x86/operations/expand-vector.hpp"
#include "grex/backend/x86/types.hpp"
#include "grex/base/defs.hpp"

namespace grex::backend {
template<bool tZero>
inline f32x4 expand(ValWrap<f32> x, IndexTag<4> /*tag*/, BoolTag<tZero> /*tag*/) {
  if constexpr (!tZero) {
#if defined(__GNUC__) && !defined(__clang__)
    __m128 retval;
    asm("" : "=x"(retval) : "0"(x.value));
    return {.r = retval};
#elif defined(__clang__)
    f32 data[4];
    data[0] = x.value;
    return {.r = _mm_load_ps(static_cast<const f32*>(data))};
#endif
  }
  return {.r = _mm_set_ss(x.value)};
}
template<bool tZero>
inline f64x2 expand(ValWrap<f64> x, IndexTag<2> /*tag*/, BoolTag<tZero> /*tag*/) {
  if constexpr (!tZero) {
#if defined(__GNUC__) && !defined(__clang__)
    __m128d retval;
    asm("" : "=x"(retval) : "0"(x.value));
    return {.r = retval};
#elif defined(__clang__)
    f64 data[2];
    data[0] = x.value;
    return {.r = _mm_load_pd(static_cast<const f64*>(data))};
#endif
  }
  return {.r = _mm_set_sd(x.value)};
}
// Integers with at most 32 bits: Cast to i32
template<Vectorizable T, bool tZero>
requires(std::integral<T> && sizeof(T) <= 4)
inline Vector<T, native_sizes<T>.front()>
expand(ValWrap<T> x, IndexTag<native_sizes<T>.front()> /*tag*/, BoolTag<tZero> /*tag*/) {
  // force zero extension
  using Unsigned = UnsignedOf<T>;
  return {.r = _mm_cvtsi32_si128(i32(Unsigned(x.value)))};
}
// Integers with 64 bits: Cast to i64
template<Vectorizable T, bool tZero>
requires(std::integral<T> && sizeof(T) == 8)
inline Vector<T, 2> expand(ValWrap<T> x, IndexTag<2> /*tag*/, BoolTag<tZero> /*tag*/) {
  return {.r = _mm_cvtsi64_si128(i64(x.value))};
}

// Sub-native: Delegate to the native version
template<Vectorizable T, std::size_t tSize, bool tZero>
requires(tSize < native_sizes<T>.front())
inline VectorFor<T, tSize> expand(ValWrap<T> x, IndexTag<tSize> /*tag*/, BoolTag<tZero> zero) {
  return VectorFor<T, tSize>{expand(x, index_tag<native_sizes<T>.front()>, zero)};
}

// Larger than the smallest native size: Merge with zero/undefined
template<Vectorizable T, std::size_t tSize, bool tZero>
requires(tSize > native_sizes<T>.front())
inline VectorFor<T, tSize> expand(ValWrap<T> x, IndexTag<tSize> /*tag*/, BoolTag<tZero> zero) {
  constexpr std::size_t half = tSize / 2;
  return expand(expand(x, index_tag<half>, zero), index_tag<tSize>, zero);
}

template<Vectorizable T, std::size_t tSize>
inline VectorFor<T, tSize> expand_any(T x, IndexTag<tSize> size) {
  return expand(ValWrap{x}, size, bool_tag<false>);
}
template<Vectorizable T, std::size_t tSize>
inline VectorFor<T, tSize> expand_zero(T x, IndexTag<tSize> size) {
  return expand(ValWrap{x}, size, bool_tag<true>);
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_EXPAND_SCALAR_HPP
