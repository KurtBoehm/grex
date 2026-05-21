// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_NEON_OPERATIONS_SET_HPP
#define INCLUDE_GREX_BACKEND_NEON_OPERATIONS_SET_HPP

#include <arm_neon.h>

#include "grex/backend/defs.hpp" // IWYU pragma: keep
#include "grex/backend/macros/cast.hpp"
#include "grex/backend/macros/conditional.hpp"
#include "grex/backend/macros/for-each.hpp"
#include "grex/backend/macros/repeat.hpp"
#include "grex/backend/neon/operations/arithmetic.hpp"
#include "grex/backend/neon/operations/expand.hpp"
#include "grex/backend/neon/operations/undefined.hpp"
#include "grex/backend/neon/types.hpp"
#include "grex/base.hpp"

namespace grex::backend {
#define GREX_SET_PARAM(CNT, IDX, TYPE) GREX_COMMA_IF(IDX) TYPE v##IDX
#define GREX_SET_ARG(CNT, IDX) GREX_COMMA_IF(IDX) v##IDX
#define GREX_BSET_VAL(CNT, IDX, TYPE) GREX_COMMA_IF(IDX) TYPE(v##IDX)

// u64
GREX_ALWAYS_INLINE inline u64x2 set64(u64x2 v0, u64x2 v1) {
  return u64x2{.r = vzip1q_u64(v0.r, v1.r)};
}
GREX_ALWAYS_INLINE inline i64x2 set64(i64x2 v0, i64x2 v1) {
  return i64x2{.r = vzip1q_s64(v0.r, v1.r)};
}
template<Int64 T>
GREX_ALWAYS_INLINE inline NativeVector<T, 2> set64(T v0, T v1) {
  return set64(expand_any(Scalar{v0}, index_tag<2>), expand_any(Scalar{v1}, index_tag<2>));
}

template<std::size_t tOffset, IntVectorizable TDst, IntVectorizable T>
GREX_ALWAYS_INLINE inline TDst merge_ints(T v0, T v1) {
  auto dst = TDst(v0);
  std::memcpy(reinterpret_cast<u8*>(&dst) + tOffset, &v1, tOffset);
  return dst;
}

template<IntVectorizable TDst, Int8 T>
GREX_ALWAYS_INLINE inline TDst merge8(T v0, T v1) {
  if (__builtin_constant_p(v0) == 0 && __builtin_constant_p(v1) == 0) {
    auto dst = TDst(v0);
    auto src = TDst(v1);
    asm("bfi %w0, %w1, %2, %3" : "+r"(dst) : "r"(src), "i"(8), "i"(8)); // NOLINT
    return dst;
  }
  return merge_ints<1, TDst>(v0, v1);
}
template<IntVectorizable T>
GREX_ALWAYS_INLINE inline CopySignInt<T, 4> merge16(T v0, T v1) {
  using Dst = CopySignInt<T, 4>;
  if (__builtin_constant_p(v0) == 0 && __builtin_constant_p(v1) == 0) {
    auto dst = Dst(v0);
    auto src = Dst(v1);
    asm("bfi %w0, %w1, %2, %3" : "+r"(dst) : "r"(src), "i"(16), "i"(16)); // NOLINT
    return dst;
  }
  return merge_ints<2, Dst>(v0, v1);
}
template<Int32 T>
GREX_ALWAYS_INLINE inline CopySignInt<T, 8> merge32(T v0, T v1) {
  using Dst = CopySignInt<T, 8>;
#if GREX_GCC
  if (__builtin_constant_p(v0) == 0 && __builtin_constant_p(v1) == 0) {
    auto dst = expand_bits<Dst>(v0);
    auto src = expand_bits<Dst>(v1);
    asm("bfi %0, %1, %2, %3" : "+r"(dst) : "r"(src), "i"(32), "i"(32)); // NOLINT
    return dst;
  }
#endif
  return merge_ints<4, Dst>(v0, v1);
}

// u32
// 2×[iu]32: fallback packing
template<Int32 T>
GREX_ALWAYS_INLINE inline NativeVector<T, 4> set32(T v0, T v1) {
  return as<T>(expand_any(Scalar{merge32(v0, v1)}, index_tag<2>));
}
template<Int32 T>
GREX_ALWAYS_INLINE inline NativeVector<T, 4> set32(T v0, T v1, T v2, T v3) {
  using I = CopySignInt<T, 8>;
  return as<T>(set64(as<I>(set32(v0, v1)), as<I>(set32(v2, v3))));
}

// u16
template<IntVectorizable T>
GREX_ALWAYS_INLINE inline NativeVector<CopySignInt<T, 2>, 8> set16(T v0, T v1) {
  return as<CopySignInt<T, 2>>(expand_any(Scalar{merge16(v0, v1)}, index_tag<4>));
}
template<IntVectorizable T>
GREX_ALWAYS_INLINE inline NativeVector<CopySignInt<T, 2>, 8> set16(T v0, T v1, T v2, T v3) {
  const auto a0 = merge16(v0, v1);
  const auto a1 = merge16(v2, v3);
  const auto b0 = merge32(a0, a1);
  return as<CopySignInt<T, 2>>(expand_any(Scalar{b0}, index_tag<2>));
}
template<IntVectorizable T>
GREX_ALWAYS_INLINE inline NativeVector<CopySignInt<T, 2>, 8> set16(T v0, T v1, T v2, T v3, T v4,
                                                                   T v5, T v6, T v7) {
  using I = CopySignInt<T, 8>;
  return as<CopySignInt<T, 2>>(set64(as<I>(set16(v0, v1, v2, v3)), as<I>(set16(v4, v5, v6, v7))));
}

// 8-bit
template<Int8 T>
GREX_ALWAYS_INLINE inline NativeVector<T, 16> set8(T v0, T v1) {
  return as<T>(expand_any(Scalar{merge8<CopySignInt<T, 4>>(v0, v1)}, index_tag<4>));
}
template<Int8 T>
GREX_ALWAYS_INLINE inline NativeVector<T, 16> set8(T v0, T v1, T v2, T v3) {
  using I = CopySignInt<T, 4>;
  return as<T>(set16(merge8<I>(v0, v1), merge8<I>(v2, v3)));
}
template<Int8 T>
GREX_ALWAYS_INLINE inline NativeVector<T, 16> set8(T v0, T v1, T v2, T v3, T v4, T v5, T v6, T v7) {
  using I = CopySignInt<T, 4>;
  const I a0 = merge8<I>(v0, v1);
  const I a1 = merge8<I>(v2, v3);
  const I a2 = merge8<I>(v4, v5);
  const I a3 = merge8<I>(v6, v7);
  return as<T>(set16(a0, a1, a2, a3));
}
template<Int8 T>
GREX_ALWAYS_INLINE inline NativeVector<T, 16> set8(T v0, T v1, T v2, T v3, T v4, T v5, T v6, T v7,
                                                   T v8, T v9, T v10, T v11, T v12, T v13, T v14,
                                                   T v15) {
  using I = CopySignInt<T, 4>;
  const I a0 = merge8<I>(v0, v1);
  const I a1 = merge8<I>(v2, v3);
  const I a2 = merge8<I>(v4, v5);
  const I a3 = merge8<I>(v6, v7);
  const I a4 = merge8<I>(v8, v9);
  const I a5 = merge8<I>(v10, v11);
  const I a6 = merge8<I>(v12, v13);
  const I a7 = merge8<I>(v14, v15);
  return as<T>(set16(a0, a1, a2, a3, a4, a5, a6, a7));
}

// f32
GREX_ALWAYS_INLINE inline f32x4 set32(f32 v0, f32 v1) {
  auto a0 = expand_any(Scalar{v0}, index_tag<4>);
  auto a1 = expand_any(Scalar{v1}, index_tag<4>);
  return f32x4{.r = vzip1q_f32(a0.r, a1.r)};
}
GREX_ALWAYS_INLINE inline f32x4 set32(f32 v0, f32 v1, f32 v2, f32 v3) {
  return f32x4{.r = vzip1q_f32(set32(v0, v2).r, set32(v1, v3).r)};
}

// f64
GREX_ALWAYS_INLINE inline f64x2 set64(f64 v0, f64 v1) {
  auto vec0 = expand_any(Scalar{v0}, index_tag<2>);
  auto vec1 = expand_any(Scalar{v1}, index_tag<2>);
  return f64x2{.r = vzip1q_f64(vec0.r, vec1.r)};
}

#define GREX_SET_i(KIND, BITS, SIZE) GREX_SET_I##BITS(KIND, BITS, SIZE)
#define GREX_SET_u(KIND, BITS, SIZE) GREX_SET_I##BITS(KIND, BITS, SIZE)
#define GREX_SET_f(KIND, BITS, SIZE) GREX_SET_F##BITS(KIND, BITS, SIZE)

#define GREX_SET(KIND, BITS, SIZE) \
  inline NativeVector<KIND##BITS, SIZE> set(TypeTag<NativeVector<KIND##BITS, SIZE>>, \
                                            GREX_REPEAT(SIZE, GREX_SET_PARAM, KIND##BITS)) { \
    return as<KIND##BITS>(set##BITS(GREX_REPEAT(SIZE, GREX_SET_ARG))); \
  }

GREX_FOREACH_TYPE(GREX_SET, 128)

#define GREX_CREATE(KIND, BITS, SIZE) \
  inline NativeVector<KIND##BITS, SIZE> zeros(TypeTag<NativeVector<KIND##BITS, SIZE>>) { \
    return {.r = GREX_ISUFFIXED(vdupq_n, KIND, BITS)(0)}; \
  } \
  inline NativeVector<KIND##BITS, SIZE> undefined(TypeTag<NativeVector<KIND##BITS, SIZE>>) { \
    return {.r = make_undefined<GREX_REGISTER(KIND, BITS, SIZE)>()}; \
  } \
  inline NativeVector<KIND##BITS, SIZE> broadcast(KIND##BITS value, \
                                                  TypeTag<NativeVector<KIND##BITS, SIZE>>) { \
    return {.r = GREX_ISUFFIXED(vdupq_n, KIND, BITS)(value)}; \
  } \
\
  inline NativeMask<KIND##BITS, SIZE> zeros(TypeTag<NativeMask<KIND##BITS, SIZE>>) { \
    return {.r = GREX_ISUFFIXED(vdupq_n, u, BITS)(0)}; \
  } \
  inline NativeMask<KIND##BITS, SIZE> ones(TypeTag<NativeMask<KIND##BITS, SIZE>>) { \
    return {.r = GREX_ISUFFIXED(vdupq_n, u, BITS)(u##BITS(-1))}; \
  } \
  inline NativeMask<KIND##BITS, SIZE> broadcast(bool value, \
                                                TypeTag<NativeMask<KIND##BITS, SIZE>>) { \
    const u##BITS entry = GREX_OPCAST(u, BITS, -u##BITS(value)); \
    return {.r = GREX_ISUFFIXED(vdupq_n, u, BITS)(entry)}; \
  } \
  /* Idea: Set to the Boolean values and use a greater-than comparison with zero */ \
  inline NativeMask<KIND##BITS, SIZE> set(TypeTag<NativeMask<KIND##BITS, SIZE>>, \
                                          GREX_REPEAT(SIZE, GREX_SET_PARAM, bool)) { \
    using R = NativeVector<u##BITS, SIZE>; \
    return {.r = negate(set(type_tag<R>, GREX_REPEAT(SIZE, GREX_BSET_VAL, u##BITS))).r}; \
  }

GREX_FOREACH_TYPE(GREX_CREATE, 128)

#define GREX_SET_SUB(KIND, BITS, PART, SIZE) \
  inline SubVector<KIND##BITS, PART, SIZE> set(TypeTag<SubVector<KIND##BITS, PART, SIZE>>, \
                                               GREX_REPEAT(PART, GREX_SET_PARAM, KIND##BITS)) { \
    const auto m = set##BITS(GREX_REPEAT(PART, GREX_SET_ARG)); \
    return SubVector<KIND##BITS, PART, SIZE>{as<KIND##BITS>(m)}; \
  }
GREX_FOREACH_SUB(GREX_SET_SUB)
} // namespace grex::backend

#include "grex/backend/shared/operations/set.hpp" // IWYU pragma: export

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_SET_HPP
