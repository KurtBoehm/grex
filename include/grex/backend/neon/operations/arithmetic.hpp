// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_NEON_OPERATIONS_ARITHMETIC_HPP
#define INCLUDE_GREX_BACKEND_NEON_OPERATIONS_ARITHMETIC_HPP

#include <arm_neon.h>

#include "grex/backend/macros/types.hpp"
#include "grex/backend/neon/macros/types.hpp"
#include "grex/backend/neon/operations/reinterpret.hpp"
#include "grex/backend/neon/types.hpp"

namespace grex::backend {
#define GREX_NEGATE_f(BITS, SIZE) return {.r = vnegq_f##BITS(a.r)};
#define GREX_NEGATE_i(BITS, SIZE) return {.r = vnegq_s##BITS(a.r)};
#define GREX_NEGATE_u(BITS, SIZE) \
  const int##BITS##x##SIZE##_t reinterpreted = reinterpret<i##BITS>(a.r); \
  return {.r = reinterpret<u##BITS>(vnegq_s##BITS(reinterpreted))};

#define GREX_MUL_BASE(KIND, BITS) return {.r = GREX_ISUFFIXED(vmulq, KIND, BITS)(a.r, b.r)};
#define GREX_MUL_f64 GREX_MUL_BASE
#define GREX_MUL_u64(...) \
  /* Let `a32` and `b32` be `a` and `b` interpreted as u32x4 */ \
  /* Basic idea: ((u64(a32[1]) << 32 + a32[0]) * (u64(b32[1]) << 32 + b32[0])) */ \
  /*           = (a32[1] * b32[0] << 32) + (a32[0] * b32[1] << 32) + u64(a32[1]) * u64(b32[0]) */ \
  const uint32x4_t a32 = vreinterpretq_u32_u64(a.r); \
  const uint32x4_t b32 = vreinterpretq_u32_u64(b.r); \
  /* [a32[0], a32[2]] */ \
  const uint32x2_t alow = vmovn_u64(a.r); \
  /* [b32[0], b32[2]] */ \
  const uint32x2_t blow = vmovn_u64(b.r); \
  /* [a32[1], a32[0], a32[3], a32[2]] */ \
  const uint32x4_t arev = vrev64q_u32(a32); \
  /* [a32[1] * b32[0], a32[0] * b32[1], a32[3] * b32[2], a32[2] * b32[3]] */ \
  const uint32x4_t cross = vmulq_u32(arev, b32); \
  /* [u64(a32[1]) * u64(b32[0]) + u64(a32[0]) * u64(b32[1]), \
      u64(a32[3]) * u64(b32[2]) + u64(a32[2]) * u64(b32[3])] */ \
  const uint64x2_t hiext = vpaddlq_u32(cross); \
  /* [hiext[0] << 32, hiext[1] << 32] */ \
  const uint64x2_t hi = vshlq_n_u64(hiext, 32); \
  /* [a[0] * b[0], a[1] * b[1]] */ \
  return {.r = vmlal_u32(hi, alow, blow)};
// Signed and unsigned 64×64→64 bit multiplication are equivalent
#define GREX_MUL_i64(...) \
  const uint64x2_t una = vreinterpretq_u64_s64(a.r); \
  const uint64x2_t unb = vreinterpretq_u64_s64(b.r); \
  return {.r = vreinterpretq_s64_u64(multiply(u64x2{.r = una}, u64x2{.r = unb}).r)};
#define GREX_MUL_64(KIND, BITS) GREX_MUL_##KIND##BITS(KIND, BITS)
#define GREX_MUL_32 GREX_MUL_BASE
#define GREX_MUL_16 GREX_MUL_BASE
#define GREX_MUL_8 GREX_MUL_BASE

#define GREX_ARITH(KIND, BITS, SIZE) \
  inline Vector<KIND##BITS, SIZE> negate(Vector<KIND##BITS, SIZE> a) { \
    GREX_NEGATE_##KIND(BITS, SIZE) \
  } \
  inline Vector<KIND##BITS, SIZE> add(Vector<KIND##BITS, SIZE> a, Vector<KIND##BITS, SIZE> b) { \
    return {.r = GREX_ISUFFIXED(vaddq, KIND, BITS)(a.r, b.r)}; \
  } \
  inline Vector<KIND##BITS, SIZE> subtract(Vector<KIND##BITS, SIZE> a, \
                                           Vector<KIND##BITS, SIZE> b) { \
    return {.r = GREX_ISUFFIXED(vsubq, KIND, BITS)(a.r, b.r)}; \
  } \
  inline Vector<KIND##BITS, SIZE> multiply(Vector<KIND##BITS, SIZE> a, \
                                           Vector<KIND##BITS, SIZE> b) { \
    GREX_MUL_##BITS(KIND, BITS) \
  }

#define GREX_ARITH_DIV(KIND, BITS, SIZE) \
  inline Vector<KIND##BITS, SIZE> divide(Vector<KIND##BITS, SIZE> a, Vector<KIND##BITS, SIZE> b) { \
    return {.r = GREX_ISUFFIXED(vdivq, KIND, BITS)(a.r, b.r)}; \
  }

GREX_FOREACH_TYPE(GREX_ARITH, 128)
GREX_FOREACH_FP_TYPE(GREX_ARITH_DIV, 128)

GREX_SUBVECTOR_UNARY(negate)
GREX_SUBVECTOR_BINARY(add)
GREX_SUBVECTOR_BINARY(subtract)
GREX_SUBVECTOR_BINARY(multiply)
GREX_SUBVECTOR_BINARY(divide)

GREX_SUPERVECTOR_UNARY(negate)
GREX_SUPERVECTOR_BINARY(add)
GREX_SUPERVECTOR_BINARY(subtract)
GREX_SUPERVECTOR_BINARY(multiply)
GREX_SUPERVECTOR_BINARY(divide)
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_ARITHMETIC_HPP
