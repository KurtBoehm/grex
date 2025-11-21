// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_NEON_OPERATIONS_EXPAND_REGISTER_HPP
#define INCLUDE_GREX_BACKEND_NEON_OPERATIONS_EXPAND_REGISTER_HPP

#include <arm_neon.h>

#include "grex/backend/defs.hpp"
#include "grex/backend/macros/for-each.hpp"
#include "grex/backend/neon/macros/types.hpp"
#include "grex/backend/neon/operations/reinterpret.hpp"
#include "grex/base.hpp"

namespace grex::backend {
// Cast TSrc to TDst with arbitrary values in the upper bits
template<IntVectorizable TDst, IntVectorizable TSrc>
inline TDst expand_bits(TSrc src) {
  if (__builtin_constant_p(src)) {
    return TDst(src);
  }
  TDst dst;
  asm("" : "=r"(dst) : "0"(src)); // NOLINT
  return dst;
}

// Different ways of re-interpreting a floating-point variable as a SIMD register
#if GREX_GCC
#define GREX_EXPAND_REGISTER_f(KIND, BITS, SIZE) \
  float##BITS##x##SIZE##_t retval; \
  asm("" : "=w"(retval) : "0"(x.value)); \
  return retval;
#elif GREX_CLANG
#define GREX_EXPAND_REGISTER_f(KIND, BITS, SIZE) \
  f##BITS data[SIZE]; \
  data[0] = x.value; \
  return vld1q_f##BITS(static_cast<const f##BITS*>(data));
#endif

// Use a bit cast from integer to a floating-point value to generate `fmov` and go from there
// 8-bit and 16-bit integers are re-interpreted as a 32-bit integer (with garbage in the upper bits)
#define GREX_EXPAND_INT_BIG(KIND, BITS, SIZE) \
  const auto fp = expand_register(Scalar{std::bit_cast<f##BITS>(x.value)}); \
  return reinterpret<KIND##BITS>(fp);
#define GREX_EXPAND_INT_SMALL(KIND, BITS, SIZE) \
  const auto expanded = expand_bits<KIND##32>(x.value); \
  const auto fp = expand_register(Scalar{std::bit_cast<f32>(expanded)}); \
  return reinterpret<KIND##BITS>(fp);
#define GREX_EXPAND_INT64 GREX_EXPAND_INT_BIG
#define GREX_EXPAND_INT32 GREX_EXPAND_INT_BIG
#define GREX_EXPAND_INT16 GREX_EXPAND_INT_SMALL
#define GREX_EXPAND_INT8 GREX_EXPAND_INT_SMALL
#define GREX_EXPAND_REGISTER_i(KIND, BITS, SIZE) GREX_EXPAND_INT##BITS(KIND, BITS, SIZE)
#define GREX_EXPAND_REGISTER_u(KIND, BITS, SIZE) GREX_EXPAND_INT##BITS(KIND, BITS, SIZE)

#define GREX_EXPAND_REGISTER(KIND, BITS, SIZE) \
  inline GREX_REGISTER(KIND, BITS, SIZE) expand_register(Scalar<KIND##BITS> x) { \
    GREX_EXPAND_REGISTER_##KIND(KIND, BITS, SIZE) \
  }
GREX_FOREACH_TYPE(GREX_EXPAND_REGISTER, 128)
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_EXPAND_REGISTER_HPP
