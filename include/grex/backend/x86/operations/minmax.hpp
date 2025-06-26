// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_MINMAX_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_MINMAX_HPP

#include <immintrin.h>

#include "grex/backend/x86/helpers.hpp"
#include "grex/backend/x86/instruction-sets.hpp"
#include "grex/backend/x86/macros/base.hpp"
#include "grex/backend/x86/macros/decrement.hpp"
#include "grex/backend/x86/types.hpp"
#include "grex/base/defs.hpp" // IWYU pragma: keep

#if GREX_X86_64_LEVEL == 1
#include "grex/backend/x86/operations/bitwise.hpp"
#include "grex/backend/x86/operations/set.hpp"
#endif
#if GREX_X86_64_LEVEL < 4
#include "grex/backend/x86/operations/blend.hpp"
#include "grex/backend/x86/operations/compare.hpp"
#endif

namespace grex::backend {
#define GREX_MINMAX_INTRINSIC(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS, OP) \
  return {.r = GREX_CAT(BITPREFIX##_##OP##_, GREX_EPU_SUFFIX(KIND, BITS))(a.r, b.r)};
#define GREX_MINMAX_FLIP_IMPL(TOELEMENT, KIND, BITS, SIZE, BITPREFIX, REGISTERBITS, OP) \
  auto signbit = broadcast(KIND##BITS(1U << GREX_CAT(GREX_DECR(BITS), U)), \
                           type_tag<Vector<KIND##BITS, SIZE>>); \
  auto a1 = bitwise_xor(a, signbit); \
  auto b1 = bitwise_xor(b, signbit); \
  auto m1 = _mm_##OP##_ep##TOELEMENT(a1.r, b1.r); \
  return bitwise_xor({.r = m1}, signbit);
#define GREX_MINMAX_BLEND_min(...) blend(compare_lt(a, b), b, a)
#define GREX_MINMAX_BLEND_max(...) blend(compare_lt(a, b), a, b)
#define GREX_MINMAX_BLEND(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS, OP) \
  return GREX_MINMAX_BLEND_##OP(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS);
#if GREX_X86_64_LEVEL >= 4
#define GREX_MINMAX_BLEND64 GREX_MINMAX_INTRINSIC
#else
#define GREX_MINMAX_BLEND64 GREX_MINMAX_BLEND
#endif
#if GREX_X86_64_LEVEL >= 2
#define GREX_MINMAX_BLEND32 GREX_MINMAX_INTRINSIC
#define GREX_MINMAX_FLIP(TOELEMENT, ...) GREX_MINMAX_INTRINSIC(__VA_ARGS__)
#else
#define GREX_MINMAX_BLEND32 GREX_MINMAX_BLEND
#define GREX_MINMAX_FLIP GREX_MINMAX_FLIP_IMPL
#endif

#define GREX_MINMAX_IMPL_128_f32 GREX_MINMAX_INTRINSIC
#define GREX_MINMAX_IMPL_128_f64 GREX_MINMAX_INTRINSIC
#define GREX_MINMAX_IMPL_128_i8(...) GREX_MINMAX_FLIP(u8, __VA_ARGS__)
#define GREX_MINMAX_IMPL_128_i16 GREX_MINMAX_INTRINSIC
#define GREX_MINMAX_IMPL_128_i32 GREX_MINMAX_BLEND32
#define GREX_MINMAX_IMPL_128_i64 GREX_MINMAX_BLEND64
#define GREX_MINMAX_IMPL_128_u8 GREX_MINMAX_INTRINSIC
#define GREX_MINMAX_IMPL_128_u16(...) GREX_MINMAX_FLIP(i16, __VA_ARGS__)
#define GREX_MINMAX_IMPL_128_u32 GREX_MINMAX_BLEND32
#define GREX_MINMAX_IMPL_128_u64 GREX_MINMAX_BLEND64
#define GREX_MINMAX_IMPL_128(KIND, BITS, ...) \
  GREX_MINMAX_IMPL_128_##KIND##BITS(KIND, BITS, __VA_ARGS__)
#define GREX_MINMAX_IMPL_256_f32 GREX_MINMAX_INTRINSIC
#define GREX_MINMAX_IMPL_256_f64 GREX_MINMAX_INTRINSIC
#define GREX_MINMAX_IMPL_256_i8 GREX_MINMAX_INTRINSIC
#define GREX_MINMAX_IMPL_256_i16 GREX_MINMAX_INTRINSIC
#define GREX_MINMAX_IMPL_256_i32 GREX_MINMAX_INTRINSIC
#define GREX_MINMAX_IMPL_256_i64 GREX_MINMAX_BLEND64
#define GREX_MINMAX_IMPL_256_u8 GREX_MINMAX_INTRINSIC
#define GREX_MINMAX_IMPL_256_u16 GREX_MINMAX_INTRINSIC
#define GREX_MINMAX_IMPL_256_u32 GREX_MINMAX_INTRINSIC
#define GREX_MINMAX_IMPL_256_u64 GREX_MINMAX_BLEND64
#define GREX_MINMAX_IMPL_256(KIND, BITS, ...) \
  GREX_MINMAX_IMPL_256_##KIND##BITS(KIND, BITS, __VA_ARGS__)
#define GREX_MINMAX_IMPL_512 GREX_MINMAX_INTRINSIC
#define GREX_MINMAX_IMPL(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS, OP) \
  GREX_MINMAX_IMPL_##REGISTERBITS(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS, OP)

#define GREX_MINMAX(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS, OP) \
  inline Vector<KIND##BITS, SIZE> OP(Vector<KIND##BITS, SIZE> a, Vector<KIND##BITS, SIZE> b) { \
    GREX_MINMAX_IMPL(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS, OP) \
  }
#define GREX_MINMAX_ALL(REGISTERBITS, BITPREFIX) \
  GREX_FOREACH_TYPE(GREX_MINMAX, REGISTERBITS, BITPREFIX, REGISTERBITS, min) \
  GREX_FOREACH_TYPE(GREX_MINMAX, REGISTERBITS, BITPREFIX, REGISTERBITS, max)
GREX_FOREACH_X86_64_LEVEL(GREX_MINMAX_ALL)

GREX_SUBVECTOR_BINARY(min)
GREX_SUBVECTOR_BINARY(max)

GREX_SUPERVECTOR_BINARY(min)
GREX_SUPERVECTOR_BINARY(max)
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_MINMAX_HPP
