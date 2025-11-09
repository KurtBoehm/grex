// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_ARITHMETIC_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_ARITHMETIC_HPP

#include <cstddef> // IWYU pragma: keep

#include <immintrin.h>

#include "grex/backend/defs.hpp"
#include "grex/backend/macros/base.hpp"
#include "grex/backend/x86/instruction-sets.hpp"
#include "grex/backend/x86/macros/for-each.hpp"
#include "grex/backend/x86/macros/intrinsics.hpp"
#include "grex/backend/x86/macros/types.hpp"
#include "grex/backend/x86/types.hpp"
#include "grex/base/defs.hpp" // IWYU pragma: keep

#if GREX_X86_64_LEVEL < 4
#include "grex/backend/x86/operations/blend.hpp"
#endif

namespace grex::backend {
// Base case: Use intrinsics
#define GREX_ARITH_BASE(KIND, BITS, SIZE, NAME, OP) \
  inline Vector<KIND##BITS, SIZE> NAME(Vector<KIND##BITS, SIZE> a, Vector<KIND##BITS, SIZE> b) { \
    return {.r = GREX_CAT(OP##_, GREX_EPI_SUFFIX(KIND, BITS))(a.r, b.r)}; \
  }
#define GREX_ADDSUB_ALL(REGISTERBITS, BITPREFIX) \
  GREX_FOREACH_TYPE(GREX_ARITH_BASE, REGISTERBITS, add, BITPREFIX##_add) \
  GREX_FOREACH_TYPE(GREX_ARITH_BASE, REGISTERBITS, subtract, BITPREFIX##_sub)

// Negation: Flip sign bit for floating-point values, subtract from zero for integers
#define GREX_NEGATE_FP(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS, KINDSUFFIX) \
  BITPREFIX##_xor_##KINDSUFFIX(v.r, BITPREFIX##_set1_##KINDSUFFIX(KIND##BITS{-0.0}))
#define GREX_NEGATE_INT(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS, KINDSUFFIX) \
  BITPREFIX##_sub_##KINDSUFFIX(BITPREFIX##_setzero_si##REGISTERBITS(), v.r)
#define GREX_NEGATE_f(...) GREX_NEGATE_FP(__VA_ARGS__)
#define GREX_NEGATE_i(...) GREX_NEGATE_INT(__VA_ARGS__)
#define GREX_NEGATE_u(...) GREX_NEGATE_INT(__VA_ARGS__)
#define GREX_NEGATE(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  inline Vector<KIND##BITS, SIZE> negate(Vector<KIND##BITS, SIZE> v) { \
    return {.r = GREX_NEGATE_##KIND(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS, \
                                    GREX_EPI_SUFFIX(KIND, BITS))}; \
  }
#define GREX_NEGATE_ALL(REGISTERBITS, BITPREFIX) \
  GREX_FOREACH_TYPE(GREX_NEGATE, REGISTERBITS, BITPREFIX, REGISTERBITS)

// Integer multiplication
// 8 bit: No intrinsic, two 16 bit multiplications, shuffling and blending.
// Based on VCL.
#define GREX_MULLO_INT8_BASE(KIND, BITS, SIZE, BITPREFIX) \
  /* odd-numbered elements of a */ \
  const auto aodd = BITPREFIX##_srli_epi16(a.r, 8); \
  /* odd-numbered elements of b */ \
  const auto bodd = BITPREFIX##_srli_epi16(b.r, 8); \
  /* product of even-numbered elements */ \
  const auto muleven = BITPREFIX##_mullo_epi16(a.r, b.r); \
  /* product of odd-numbered elements */ \
  auto mulodd = BITPREFIX##_mullo_epi16(aodd, bodd); \
  /* put odd-numbered elements back in place */ \
  mulodd = BITPREFIX##_slli_epi16(mulodd, 8);
#if GREX_X86_64_LEVEL >= 4
#define GREX_MUL_INT8(KIND, BITS, SIZE, BITPREFIX) \
  GREX_MULLO_INT8_BASE(KIND, BITS, SIZE, BITPREFIX) \
  return {.r = \
            BITPREFIX##_mask_mov_epi8(mulodd, GREX_MMASK(SIZE)(0x5555555555555555), muleven)};
#else
#define GREX_MUL_INT8(KIND, BITS, SIZE, BITPREFIX) \
  GREX_MULLO_INT8_BASE(KIND, BITS, SIZE, BITPREFIX) \
  /* mask for even positions */ \
  auto mask = BITPREFIX##_set1_epi32(0x00FF00FF); \
  return blend(Mask<KIND##BITS, SIZE>{.r = mask}, {.r = mulodd}, {.r = muleven});
#endif
// 16 bit: Use the existing intrinsic
#define GREX_MUL_INT16(KIND, BITS, SIZE, BITPREFIX) return {.r = BITPREFIX##_mullo_epi16(a.r, b.r)};
// 32 bit: Use the existing intrinsic starting on level 2, otherwise use extended multiplication.
// Based on clang-generated assembly using GCC vector extensions.
#define GREX_MUL_INT32_SSE \
  /* [a0 * b0, a2 * b2] (32×32→64 bit) */ \
  const auto pl = _mm_mul_epu32(a.r, b.r); \
  /* [a1, 0, a3, 0] */ \
  const auto ah = _mm_srli_epi64(a.r, 32); \
  /* [b1, 0, b3, 0] */ \
  const auto bh = _mm_srli_epi64(b.r, 32); \
  /* [a1 * b1, a3 * b3] (32×32→64 bit) */ \
  const auto ph = _mm_mul_epu32(ah, bh); \
  /* [a0 * b0, a2 * b2, -, -] (lower 32 bits) */ \
  const auto pls = _mm_shuffle_epi32(pl, 8); \
  /* [a1 * b1, a3 * b3, -, -] (lower 32 bits) */ \
  const auto phs = _mm_shuffle_epi32(ph, 8); \
  /* [a0 * b0, a1 * b1, a2 * b2, a3 * b3] */ \
  return {.r = _mm_unpacklo_epi32(pls, phs)};
#if GREX_X86_64_LEVEL >= 2
#define GREX_MUL_INT32(KIND, BITS, SIZE, BITPREFIX) return {.r = BITPREFIX##_mullo_epi32(a.r, b.r)};
#else
#define GREX_MUL_INT32(...) GREX_MUL_INT32_SSE
#endif
// 64 bit: Without AVX-512, emulate using three 32×32→64 bit multiplications,
// some shifting and some additions.
// Based on clang-generated assembly using GCC vector extensions.
#define GREX_MUL_INT64_BASE(BITPREFIX) \
  /* c[0:64] = b[32:64] * a[0:32] */ \
  auto albu = BITPREFIX##_mul_epu32(a.r, BITPREFIX##_srli_epi64(b.r, 32)); \
  /* d[0:64] = a[32:64] * b[0:32] */ \
  auto aubl = BITPREFIX##_mul_epu32(BITPREFIX##_srli_epi64(a.r, 32), b.r); \
  /* d = upper 32 bit of the product without “overflow” from the product of the lower 32 bits */ \
  auto upmul = BITPREFIX##_slli_epi64(BITPREFIX##_add_epi64(aubl, albu), 32); \
  /* a[0:32] * b[0:32] + d */ \
  return {.r = BITPREFIX##_add_epi64(BITPREFIX##_mul_epu32(a.r, b.r), upmul)};
#if GREX_X86_64_LEVEL >= 4
#define GREX_MUL_INT64(KIND, BITS, SIZE, BITPREFIX) return {.r = BITPREFIX##_mullo_epi64(a.r, b.r)};
#else
#define GREX_MUL_INT64(KIND, BITS, SIZE, BITPREFIX) GREX_MUL_INT64_BASE(BITPREFIX)
#endif

#define GREX_MUL_f(KIND, BITS, SIZE, BITPREFIX) \
  return {.r = GREX_CAT(BITPREFIX##_mul_, GREX_FP_SUFFIX(BITS))(a.r, b.r)};
#define GREX_MUL_i(KIND, BITS, ...) GREX_MUL_INT##BITS(KIND, BITS, __VA_ARGS__)
#define GREX_MUL_u(KIND, BITS, ...) GREX_MUL_INT##BITS(KIND, BITS, __VA_ARGS__)
#define GREX_MUL(KIND, BITS, SIZE, BITPREFIX) \
  inline Vector<KIND##BITS, SIZE> multiply(Vector<KIND##BITS, SIZE> a, \
                                           Vector<KIND##BITS, SIZE> b) { \
    GREX_MUL_##KIND(KIND, BITS, SIZE, BITPREFIX) \
  }
#define GREX_MUL_ALL(REGISTERBITS, BITPREFIX) GREX_FOREACH_TYPE(GREX_MUL, REGISTERBITS, BITPREFIX)

// Floating-point division (integer division is not available because it is very slow)
#define GREX_DIV_ALL(REGISTERBITS, BITPREFIX) \
  GREX_FOREACH_FP_TYPE(GREX_ARITH_BASE, REGISTERBITS, divide, BITPREFIX##_div)

GREX_FOREACH_X86_64_LEVEL(GREX_NEGATE_ALL)
GREX_FOREACH_X86_64_LEVEL(GREX_ADDSUB_ALL)
GREX_FOREACH_X86_64_LEVEL(GREX_MUL_ALL)
GREX_FOREACH_X86_64_LEVEL(GREX_DIV_ALL)

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

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_ARITHMETIC_HPP
