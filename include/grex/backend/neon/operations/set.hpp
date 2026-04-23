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
#include "grex/backend/neon/instructions.hpp"
#include "grex/backend/neon/macros/cast.hpp"
#include "grex/backend/neon/operations/arithmetic.hpp"
#include "grex/backend/neon/operations/expand.hpp"
#include "grex/backend/neon/operations/undefined.hpp"
#include "grex/backend/neon/types.hpp"
#include "grex/base.hpp"

namespace grex::backend {
#define GREX_SET_ARG(CNT, IDX, TYPE) GREX_COMMA_IF(IDX) TYPE v##IDX
#define GREX_BSET_VAL(CNT, IDX, TYPE) GREX_COMMA_IF(IDX) TYPE(v##IDX)

#define GREX_SET_KINDCAST_f f
#define GREX_SET_KINDCAST_i u
#define GREX_SET_KINDCAST_u u
#define GREX_SET_KINDCAST(CNT, IDX, KIND, BITS) \
  GREX_COMMA_IF(IDX) GREX_KINDCAST(GREX_SET_KINDCAST_##KIND, KIND, BITS, v##IDX)

// u64
GREX_ALWAYS_INLINE inline u64x2 set64(u64x2 v0, u64x2 v1) {
  return u64x2{.r = vzip1q_u64(v0.r, v1.r)};
}
GREX_ALWAYS_INLINE inline u64x2 set64(u64 v0, u64 v1) {
  return set64(expand_any(Scalar{v0}, index_tag<2>), expand_any(Scalar{v1}, index_tag<2>));
}

// u32
#if GREX_GCC
// 2×[iu]32: uses BFI
GREX_ALWAYS_INLINE inline u64x2 set32(u32 v0, u32 v1) {
  const u64 a0 = isa::bfi<32, 32>(expand_bits<u64>(v0), expand_bits<u64>(v1));
  return expand_any(Scalar{a0}, index_tag<2>);
}
#else
// 2×[iu]32: fallback packing
GREX_ALWAYS_INLINE inline u64x2 set32(u32 v0, u32 v1) {
  return expand_any(Scalar{v0 | (u64{v1} << 32)}, index_tag<2>);
}
#endif
GREX_ALWAYS_INLINE inline u32x4 set32(u32 v0, u32 v1, u32 v2, u32 v3) {
  return as<u32>(set64(set32(v0, v1), set32(v2, v3)));
}

// u16
GREX_ALWAYS_INLINE inline u16x8 set16(u32 v0, u32 v1) {
  return as<u16>(expand_any(Scalar{isa::orr32(v0, v1, isa::lsl<16>)}, index_tag<4>));
}
GREX_ALWAYS_INLINE inline u16x8 set16(u32 v0, u32 v1, u32 v2, u32 v3) {
  const u64 a0 = isa::orr64(v0, v1, isa::lsl<16>);
  const u64 a1 = isa::orr64(v2, v3, isa::lsl<16>);
  const u64 b0 = isa::orr64(a0, a1, isa::lsl<32>);
  return as<u16>(expand_any(Scalar{b0}, index_tag<2>));
}
GREX_ALWAYS_INLINE inline u16x8 set16(u32 v0, u32 v1, u32 v2, u32 v3, u32 v4, u32 v5, u32 v6,
                                      u32 v7) {
  return as<u16>(set64(as<u64>(set16(v0, v1, v2, v3)), as<u64>(set16(v4, v5, v6, v7))));
}

// u8
GREX_ALWAYS_INLINE inline u8x16 set8(u32 v0, u32 v1) {
  return as<u8>(expand_any(Scalar{isa::orr32(v0, v1, isa::lsl<8>)}, index_tag<4>));
}
GREX_ALWAYS_INLINE inline u8x16 set8(u32 v0, u32 v1, u32 v2, u32 v3) {
  return as<u8>(set16(isa::orr32(v0, v1, isa::lsl<8>), isa::orr32(v2, v3, isa::lsl<8>)));
}
GREX_ALWAYS_INLINE inline u8x16 set8(u32 v0, u32 v1, u32 v2, u32 v3, u32 v4, u32 v5, u32 v6,
                                     u32 v7) {
  const u32 a0 = isa::orr32(v0, v1, isa::lsl<8>);
  const u32 a1 = isa::orr32(v2, v3, isa::lsl<8>);
  const u32 a2 = isa::orr32(v4, v5, isa::lsl<8>);
  const u32 a3 = isa::orr32(v6, v7, isa::lsl<8>);
  return as<u8>(set16(a0, a1, a2, a3));
}
GREX_ALWAYS_INLINE inline u8x16 set8(u32 v0, u32 v1, u32 v2, u32 v3, u32 v4, u32 v5, u32 v6, u32 v7,
                                     u32 v8, u32 v9, u32 v10, u32 v11, u32 v12, u32 v13, u32 v14,
                                     u32 v15) {
  const u32 a0 = isa::orr32(v0, v1, isa::lsl<8>);
  const u32 a1 = isa::orr32(v2, v3, isa::lsl<8>);
  const u32 a2 = isa::orr32(v4, v5, isa::lsl<8>);
  const u32 a3 = isa::orr32(v6, v7, isa::lsl<8>);
  const u32 a4 = isa::orr32(v8, v9, isa::lsl<8>);
  const u32 a5 = isa::orr32(v10, v11, isa::lsl<8>);
  const u32 a6 = isa::orr32(v12, v13, isa::lsl<8>);
  const u32 a7 = isa::orr32(v14, v15, isa::lsl<8>);
  return as<u8>(set16(a0, a1, a2, a3, a4, a5, a6, a7));
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
                                            GREX_REPEAT(SIZE, GREX_SET_ARG, KIND##BITS)) { \
    return as<KIND##BITS>(set##BITS(GREX_REPEAT(SIZE, GREX_SET_KINDCAST, KIND, BITS))); \
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
                                          GREX_REPEAT(SIZE, GREX_SET_ARG, bool)) { \
    using R = NativeVector<u##BITS, SIZE>; \
    return {.r = negate(set(type_tag<R>, GREX_REPEAT(SIZE, GREX_BSET_VAL, u##BITS))).r}; \
  }

GREX_FOREACH_TYPE(GREX_CREATE, 128)

#define GREX_SET_SUB(KIND, BITS, PART, SIZE) \
  inline SubVector<KIND##BITS, PART, SIZE> set(TypeTag<SubVector<KIND##BITS, PART, SIZE>>, \
                                               GREX_REPEAT(PART, GREX_SET_ARG, KIND##BITS)) { \
    const auto m = set##BITS(GREX_REPEAT(PART, GREX_SET_KINDCAST, KIND, BITS)); \
    return SubVector<KIND##BITS, PART, SIZE>{as<KIND##BITS>(m)}; \
  }
GREX_FOREACH_SUB(GREX_SET_SUB)
} // namespace grex::backend

#include "grex/backend/shared/operations/set.hpp" // IWYU pragma: export

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_SET_HPP
