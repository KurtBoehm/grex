// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_EXPAND_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_EXPAND_HPP

#include <concepts>
#include <cstddef>
#include <cstring>

#include <immintrin.h>

#include "grex/backend/base.hpp"
#include "grex/backend/choosers.hpp"
#include "grex/backend/defs.hpp"
#include "grex/backend/macros/base.hpp"
#include "grex/backend/macros/math.hpp"
#include "grex/backend/macros/types.hpp"
#include "grex/backend/x86/instruction-sets.hpp"
#include "grex/backend/x86/macros/intrinsics.hpp"
#include "grex/backend/x86/operations/merge.hpp"
#include "grex/backend/x86/sizes.hpp"
#include "grex/backend/x86/types.hpp"
#include "grex/base.hpp"

#if GREX_X86_64_LEVEL >= 3
#include "grex/backend/macros/for-each.hpp"
#endif

namespace grex::backend {
//==================================================================================================
// Bits
//==================================================================================================

// Cast Src to Dst with arbitrary values in the upper bits
template<IntVectorizable Dst, IntVectorizable Src>
inline Dst expand_bits(Src src) {
  if (__builtin_constant_p(src)) {
    return Dst(src);
  }
  Dst dst;
  asm("" : "=r"(dst) : "0"(src)); // NOLINT
  return dst;
}

//==================================================================================================
// Scalar
//==================================================================================================

template<std::same_as<f16> T, bool Zero>
inline f16x8 expand(T x, IndexTag<8> size, BoolTag<Zero> /*zero*/) {
  if constexpr (!Zero) {
    // A compile-time value has to stay recognizable as one: The `asm` block below is opaque to
    // GCC, which would keep callers such as `set` from folding a constant argument list into a
    // single vector constant. Zeroing the upper lanes is permitted, as they are arbitrary anyway.
    if (__builtin_constant_p(x)) {
      return {.r = _mm_cvtsi32_si128(i32(f16_bits(x)))};
    }
#if GREX_GCC
    __m128i retval;
    asm("" : "=x"(retval) : "0"(x)); // NOLINT
    return {.r = retval};
#elif GREX_CLANG
    // Clang rejects tying a binary16 input to a `__m128i` output, so the value is routed through
    // memory, which Clang folds away. The empty `asm` afterwards makes the result opaque, matching
    // GCC: Without it, Clang tracks the sole lane back to the scalar and rewrites consumers such as
    // the zeroing below into `vpextrw`/`movzx`/`vmovd`, taking the detour this expansion avoids.
    f16 data[8];
    data[0] = x;
    __m128i retval = _mm_load_si128(reinterpret_cast<const __m128i*>(data));
    asm("" : "+x"(retval)); // NOLINT
    return {.r = retval};
#endif
  }
  const __m128i any = expand(x, size, bool_tag<false>).r;
#if GREX_F16_NATIVE_ARITHMETIC
  return {.r = _mm_castph_si128(_mm_move_sh(_mm_setzero_ph(), _mm_castsi128_ph(any)))};
#elif GREX_X86_64_LEVEL >= 2
  return {.r = _mm_blend_epi16(_mm_setzero_si128(), any, 1)};
#else
  return {.r = _mm_bsrli_si128(_mm_bslli_si128(any, 14), 14)};
#endif
}
template<std::same_as<f32> T, bool Zero>
inline f32x4 expand(T x, IndexTag<4> /*tag*/, BoolTag<Zero> /*tag*/) {
  if constexpr (!Zero) {
#if GREX_GCC
    __m128 retval;
    asm("" : "=x"(retval) : "0"(x));
    return {.r = retval};
#elif GREX_CLANG
    f32 data[4];
    data[0] = x;
    return {.r = _mm_load_ps(static_cast<const f32*>(data))};
#endif
  }
  return {.r = _mm_set_ss(x)};
}
template<std::same_as<f64> T, bool Zero>
inline f64x2 expand(T x, IndexTag<2> /*tag*/, BoolTag<Zero> /*tag*/) {
  if constexpr (!Zero) {
#if GREX_GCC
    __m128d retval;
    asm("" : "=x"(retval) : "0"(x));
    return {.r = retval};
#elif GREX_CLANG
    f64 data[2];
    data[0] = x;
    return {.r = _mm_load_pd(static_cast<const f64*>(data))};
#endif
  }
  return {.r = _mm_set_sd(x)};
}
// Integers with at most 32 bits: Cast to i32
template<IntVectorizable T, bool Zero>
requires(sizeof(T) <= 4)
inline NativeVector<T, min_native_size<T>> expand(T x, IndexTag<min_native_size<T>> /*tag*/,
                                                  BoolTag<Zero> /*tag*/) {
  // force zero extension
  using Unsigned = UnsignedOf<T>;
  if constexpr (Zero) {
    return {.r = _mm_cvtsi32_si128(i32(Unsigned(x)))};
  } else {
#if GREX_GCC
    return {.r = _mm_cvtsi32_si128(expand_bits<i32>(x))};
#else
    __m128i dst;
    std::memcpy(&dst, &x, sizeof(x));
    return {.r = dst};
#endif
  }
}
// Integers with 64 bits: Cast to i64
template<IntVectorizable T, bool Zero>
requires(sizeof(T) == 8)
inline NativeVector<T, 2> expand(T x, IndexTag<2> /*tag*/, BoolTag<Zero> /*tag*/) {
  return {.r = _mm_cvtsi64_si128(i64(x))};
}

//==================================================================================================
// Vector
//==================================================================================================

// native → native: use cast/zext intrinsics
#define GREX_EXPANDV_INTRINSIC_I(KIND, BITS, DSTSIZE, SRCRBITS, DSTRBITS, SRCSFX, DSTSFX) \
  inline NativeVector<KIND##BITS, DSTSIZE> expand( \
    NativeVector<KIND##BITS, GREX_DIVIDE(SRCRBITS, BITS)> v, IndexTag<DSTSIZE>, BoolTag<false>) { \
    return {.r = GREX_CAT(GREX_BITPREFIX(DSTRBITS), _cast, SRCSFX, _, DSTSFX)(v.r)}; \
  } \
  inline NativeVector<KIND##BITS, DSTSIZE> expand( \
    NativeVector<KIND##BITS, GREX_DIVIDE(SRCRBITS, BITS)> v, IndexTag<DSTSIZE>, BoolTag<true>) { \
    const auto r = GREX_CAT(GREX_BITPREFIX(DSTRBITS), _zext, SRCSFX, _, DSTSFX)(v.r); \
    return {.r = r}; \
  }
#define GREX_EXPANDV_INTRINSIC(KIND, BITS, DSTSIZE, SRCRBITS, DSTRBITS) \
  GREX_EXPANDV_INTRINSIC_I(KIND, BITS, DSTSIZE, SRCRBITS, DSTRBITS, \
                           GREX_SIR_SUFFIX(GREX_REGKIND(KIND, BITS), BITS, SRCRBITS), \
                           GREX_SIR_SUFFIX(GREX_REGKIND(KIND, BITS), BITS, DSTRBITS))

#if GREX_X86_64_LEVEL >= 3
GREX_FOREACH_TYPE_EXT(GREX_EXPANDV_INTRINSIC, 256, 128, 256)
#endif
#if GREX_X86_64_LEVEL >= 4
GREX_FOREACH_TYPE_EXT(GREX_EXPANDV_INTRINSIC, 512, 128, 512)
GREX_FOREACH_TYPE_EXT(GREX_EXPANDV_INTRINSIC, 512, 256, 512)
#endif

// native/super-native → super-native
template<AnyVector Vec, std::size_t DstN, bool Zero>
requires(DstN > size_of<Vec> && is_supernative<ValueOf<Vec>, DstN> &&
         (AnyNativeVector<Vec> || AnySuperNativeVector<Vec>))
inline VectorFor<typename Vec::Value, DstN> expand(Vec v, IndexTag<DstN> /*size*/,
                                                   BoolTag<Zero> zero_tag) {
  using Value = Vec::Value;
  using Half = VectorFor<Value, DstN / 2>;
  if constexpr (Zero) {
    return merge(expand(v, index_tag<Half::size>, zero_tag), zeros(type_tag<Half>));
  } else {
    return merge(expand(v, index_tag<Half::size>, zero_tag), undefined(type_tag<Half>));
  }
}
} // namespace grex::backend

#include "grex/backend/shared/operations/expand.hpp" // IWYU pragma: export

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_EXPAND_HPP
