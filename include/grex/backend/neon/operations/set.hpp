// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_NEON_OPERATIONS_SET_HPP
#define INCLUDE_GREX_BACKEND_NEON_OPERATIONS_SET_HPP

#include <arm_neon.h>

#include "grex/backend/defs.hpp" // IWYU pragma: keep
#include "grex/backend/macros/base.hpp"
#include "grex/backend/macros/cast.hpp"
#include "grex/backend/macros/conditional.hpp"
#include "grex/backend/macros/equals.hpp"
#include "grex/backend/macros/for-each.hpp"
#include "grex/backend/macros/repeat.hpp"
#include "grex/backend/neon/instructions.hpp"
#include "grex/backend/neon/macros/cast.hpp"
#include "grex/backend/neon/operations/arithmetic.hpp"
#include "grex/backend/neon/operations/expand-register.hpp"
#include "grex/backend/neon/operations/expand.hpp"
#include "grex/backend/neon/operations/undefined.hpp"
#include "grex/backend/neon/types.hpp"
#include "grex/base.hpp"

namespace grex::backend {
#define GREX_SET_ARG(CNT, IDX, TYPE) GREX_COMMA_IF(IDX) TYPE v##IDX
#define GREX_SET_VAL(CNT, IDX) GREX_COMMA_IF(IDX) v##IDX
#define GREX_BSET_VAL(CNT, IDX, TYPE) GREX_COMMA_IF(IDX) TYPE(v##IDX)

#define GREX_SET_LANE_0(CNT, IDX, KIND, BITS) \
  out = GREX_ISUFFIXED(vsetq_lane, KIND, BITS)(v##IDX, out, IDX);
#define GREX_SET_LANE_1(...)
#define GREX_SET_LANE(CNT, IDX, KIND, BITS) \
  GREX_CAT(GREX_SET_LANE_, GREX_EQUALS(IDX, 0))(CNT, IDX, KIND, BITS)

#define GREX_SET_BOOLLANE_0(CNT, IDX, BITS) \
  ext = GREX_ISUFFIXED(vsetq_lane, i, BITS)(i##BITS(v##IDX), ext, IDX);
#define GREX_SET_BOOLLANE_1(...)
#define GREX_SET_BOOLLANE(CNT, IDX, BITS) \
  GREX_CAT(GREX_SET_BOOLLANE_, GREX_EQUALS(IDX, 0))(CNT, IDX, BITS)

// 16×[iu]8
#define GREX_SET_I8(KIND, BITS, SIZE) \
  const u32 a0 = \
    isa::orr32(GREX_KINDCAST(u, KIND, BITS, v0), GREX_KINDCAST(u, KIND, BITS, v1), isa::lsl<8>); \
  const u32 a1 = \
    isa::orr32(GREX_KINDCAST(u, KIND, BITS, v2), GREX_KINDCAST(u, KIND, BITS, v3), isa::lsl<8>); \
  const u64 b0 = isa::orr64(u32{a0}, u32{a1}, isa::lsl<16>); \
\
  const u32 a2 = \
    isa::orr32(GREX_KINDCAST(u, KIND, BITS, v4), GREX_KINDCAST(u, KIND, BITS, v5), isa::lsl<8>); \
  const u32 a3 = \
    isa::orr32(GREX_KINDCAST(u, KIND, BITS, v6), GREX_KINDCAST(u, KIND, BITS, v7), isa::lsl<8>); \
  const u64 b1 = isa::orr64(u32{a2}, u32{a3}, isa::lsl<16>); \
\
  const u64 c0 = isa::orr64(b0, b1, isa::lsl<32>); \
  auto vec0 = expand_any(Scalar{c0}, grex::index_tag<2>); \
\
  const u32 a4 = \
    isa::orr32(GREX_KINDCAST(u, KIND, BITS, v8), GREX_KINDCAST(u, KIND, BITS, v9), isa::lsl<8>); \
  const u32 a5 = \
    isa::orr32(GREX_KINDCAST(u, KIND, BITS, v10), GREX_KINDCAST(u, KIND, BITS, v11), isa::lsl<8>); \
  const u64 b2 = isa::orr64(u32{a4}, u32{a5}, isa::lsl<16>); \
\
  const u32 a6 = \
    isa::orr32(GREX_KINDCAST(u, KIND, BITS, v12), GREX_KINDCAST(u, KIND, BITS, v13), isa::lsl<8>); \
  const u32 a7 = \
    isa::orr32(GREX_KINDCAST(u, KIND, BITS, v14), GREX_KINDCAST(u, KIND, BITS, v15), isa::lsl<8>); \
  const u64 b3 = isa::orr64(u32{a6}, u32{a7}, isa::lsl<16>); \
\
  const u64 c1 = isa::orr64(b2, b3, isa::lsl<32>); \
  auto vec1 = expand_any(Scalar{c1}, grex::index_tag<2>); \
\
  return as<KIND##BITS>(u64x2{.r = vzip1q_u64(vec0.r, vec1.r)});
// 8×[iu]16
#define GREX_SET_I16(KIND, BITS, SIZE) \
  const u64 a0 = isa::orr64(u32{GREX_KINDCAST(u, KIND, BITS, v0)}, \
                            u32{GREX_KINDCAST(u, KIND, BITS, v1)}, isa::lsl<16>); \
  const u64 a1 = isa::orr64(u32{GREX_KINDCAST(u, KIND, BITS, v2)}, \
                            u32{GREX_KINDCAST(u, KIND, BITS, v3)}, isa::lsl<16>); \
  const u64 b0 = isa::orr64(a0, a1, isa::lsl<32>); \
  auto vec0 = expand_any(Scalar{b0}, grex::index_tag<2>); \
\
  const u64 a2 = isa::orr64(u32{GREX_KINDCAST(u, KIND, BITS, v4)}, \
                            u32{GREX_KINDCAST(u, KIND, BITS, v5)}, isa::lsl<16>); \
  const u64 a3 = isa::orr64(u32{GREX_KINDCAST(u, KIND, BITS, v6)}, \
                            u32{GREX_KINDCAST(u, KIND, BITS, v7)}, isa::lsl<16>); \
  const u64 b1 = isa::orr64(a2, a3, isa::lsl<32>); \
  auto vec1 = expand_any(Scalar{b1}, grex::index_tag<2>); \
\
  return as<KIND##BITS>(u64x2{.r = vzip1q_u64(vec0.r, vec1.r)});
#if GREX_GCC
// 4×[iu]32: uses BFI
#define GREX_SET_I32(KIND, BITS, SIZE) \
  const u64 a0 = isa::bfi<32, 32>(expand_bits<u64>(GREX_KINDCAST(u, KIND, BITS, v0)), \
                                  expand_bits<u64>(GREX_KINDCAST(u, KIND, BITS, v1))); \
  auto vec0 = expand_any(Scalar{a0}, grex::index_tag<2>); \
\
  const u64 a1 = isa::bfi<32, 32>(expand_bits<u64>(GREX_KINDCAST(u, KIND, BITS, v2)), \
                                  expand_bits<u64>(GREX_KINDCAST(u, KIND, BITS, v3))); \
  auto vec1 = expand_any(Scalar{a1}, grex::index_tag<2>); \
\
  return as<KIND##BITS>(u64x2{.r = vzip1q_u64(vec0.r, vec1.r)});
#else
// 4×[iu]32: fallback packing
#define GREX_SET_I32(KIND, BITS, SIZE) \
  const u64 a0 = GREX_KINDCAST(u, KIND, BITS, v0) | (u64{GREX_KINDCAST(u, KIND, BITS, v1)} << 32); \
  auto vec0 = expand_any(Scalar{a0}, grex::index_tag<2>); \
\
  const u64 a1 = GREX_KINDCAST(u, KIND, BITS, v2) | (u64{GREX_KINDCAST(u, KIND, BITS, v3)} << 32); \
  auto vec1 = expand_any(Scalar{a1}, grex::index_tag<2>); \
\
  return as<KIND##BITS>(u64x2{.r = vzip1q_u64(vec0.r, vec1.r)});
#endif
// 2×[iu]64
#define GREX_SET_I64(KIND, BITS, SIZE) \
  auto vec0 = expand_any(Scalar{v0}, grex::index_tag<2>); \
  auto vec1 = expand_any(Scalar{v1}, grex::index_tag<2>); \
  return NativeVector<KIND##BITS, 2>{.r = GREX_ISUFFIXED(vzip1q, KIND, BITS)(vec0.r, vec1.r)};

// 4×f32
#define GREX_SET_F32(KIND, BITS, SIZE) \
  auto vec0 = expand_any(Scalar{v0}, grex::index_tag<4>); \
  auto vec1 = expand_any(Scalar{v1}, grex::index_tag<4>); \
  auto vec2 = expand_any(Scalar{v2}, grex::index_tag<4>); \
  auto vec3 = expand_any(Scalar{v3}, grex::index_tag<4>); \
  return f32x4{.r = vuzp1q_f32(vuzp1q_f32(vec0.r, vec1.r), vuzp1q_f32(vec2.r, vec3.r))};
// 2×f64
#define GREX_SET_F64(KIND, BITS, SIZE) \
  auto vec0 = expand_any(Scalar{v0}, grex::index_tag<2>); \
  auto vec1 = expand_any(Scalar{v1}, grex::index_tag<2>); \
  return f64x2{.r = vuzp1q_f64(vec0.r, vec1.r)};

#define GREX_SET_i(KIND, BITS, SIZE) GREX_SET_I##BITS(KIND, BITS, SIZE)
#define GREX_SET_u(KIND, BITS, SIZE) GREX_SET_I##BITS(KIND, BITS, SIZE)
#define GREX_SET_f(KIND, BITS, SIZE) GREX_SET_F##BITS(KIND, BITS, SIZE)

#define GREX_SET(KIND, BITS, SIZE) \
  inline NativeVector<KIND##BITS, SIZE> set(TypeTag<NativeVector<KIND##BITS, SIZE>>, \
                                            GREX_REPEAT(SIZE, GREX_SET_ARG, KIND##BITS)) { \
    GREX_SET_##KIND(KIND, BITS, SIZE) \
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
    GREX_REGISTER(KIND, BITS, SIZE) out = expand_register(Scalar{v0}); \
    GREX_RREPEAT(PART, GREX_SET_LANE, KIND, BITS) \
    return SubVector<KIND##BITS, PART, SIZE>{out}; \
  }
GREX_FOREACH_SUB(GREX_SET_SUB)
} // namespace grex::backend

#include "grex/backend/shared/operations/set.hpp"

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_SET_HPP
