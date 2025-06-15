// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_EXPAND_SCALAR_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_EXPAND_SCALAR_HPP

#include <concepts>
#include <immintrin.h>
#include <type_traits>

#include "grex/backend/defs.hpp"
#include "grex/backend/x86/types.hpp"
#include "grex/base/defs.hpp"

namespace grex::backend {
template<bool tZero>
inline f32x4 expand_impl(f32 x, std::bool_constant<tZero> /*tag*/) {
  if constexpr (!tZero) {
#if defined(__GNUC__) && !defined(__clang__)
    __m128 retval;
    asm("" : "=x"(retval) : "0"(x));
    return {.r = retval};
#elif defined(__clang__)
    f32 data[4];
    data[0] = x;
    return {.r = _mm_load_ps(static_cast<const f32*>(data))};
#endif
  }
  return {.r = _mm_set_ss(x)};
}
template<bool tZero>
inline f64x2 expand_impl(f64 x, std::bool_constant<tZero> /*tag*/) {
  if constexpr (!tZero) {
#if defined(__GNUC__) && !defined(__clang__)
    __m128d retval;
    asm("" : "=x"(retval) : "0"(x));
    return {.r = retval};
#elif defined(__clang__)
    f64 data[2];
    data[0] = x;
    return {.r = _mm_load_pd(static_cast<const f64*>(data))};
#endif
  }
  return {.r = _mm_set_sd(x)};
}
// Integers with at most 32 bits: Cast to i32
template<Vectorizable T, bool tZero>
requires(std::integral<T> && sizeof(T) <= 4)
inline Vector<T, native_sizes<T>.front()> expand_impl(T x, std::bool_constant<tZero> /*tag*/) {
  // force zero extension
  using Unsigned = UnsignedOf<T>;
  return {.r = _mm_cvtsi32_si128(i32(Unsigned(x)))};
}
// Integers with 64 bits: Cast to i64
template<Vectorizable T, bool tZero>
requires(std::integral<T> && sizeof(T) == 8)
inline Vector<T, 2> expand_impl(T x, std::bool_constant<tZero> /*tag*/) {
  return {.r = _mm_cvtsi64_si128(i64(x))};
}

template<Vectorizable T>
inline Vector<T, native_sizes<T>.front()> expand_any(T x) {
  return expand_impl(x, std::bool_constant<false>{});
}
template<Vectorizable T>
inline Vector<T, native_sizes<T>.front()> expand_zero(T x) {
  return expand_impl(x, std::bool_constant<true>{});
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_EXPAND_SCALAR_HPP
