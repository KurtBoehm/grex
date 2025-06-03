// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_COMPARE_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_COMPARE_HPP

#include <cstddef>

#include <boost/preprocessor.hpp>
#include <immintrin.h>

#include "grex/backend/defs.hpp"
#include "grex/backend/x86/helpers.hpp"
#include "grex/backend/x86/instruction-sets.hpp"
#include "grex/backend/x86/operations/bitwise.hpp" // IWYU pragma: keep
#include "grex/backend/x86/operations/set.hpp" // IWYU pragma: keep
#include "grex/backend/x86/types.hpp"
#include "grex/base/defs.hpp" // IWYU pragma: keep

namespace grex::backend {
#define GREX_CMP_AVX512(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS, OPNAME, CMPNAME, CMPIDX) \
  inline Mask<KIND##BITS, SIZE> compare_##OPNAME(Vector<KIND##BITS, SIZE> a, \
                                                 Vector<KIND##BITS, SIZE> b) { \
    return {.r = \
              GREX_CAT(BITPREFIX##_cmp_, GREX_EPU_SUFFIX(KIND, BITS), _mask)(a.r, b.r, CMPIDX)}; \
  }

////////////////////////////////////////////////////////////////////////////////////////////////////

// This is the base case (limited support for integers),
// which applies to the SSE family and integer AVX2

// Equality: All kinds have cmpeq apart from i64/u64 on level 1, unsigned fall back to signed
#define GREX_CMP_IMPL_BASE_CMPEQ_BASE(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  return {.r = \
            GREX_KINDCAST(KIND, i, BITS, REGISTERBITS, \
                          GREX_CAT(BITPREFIX##_cmpeq_, GREX_EPI_SUFFIX(KIND, BITS))(a.r, b.r))};
// i64/u64: 32 bit comparison, swap 32 bit pairs, and perform “and”
#define GREX_CMP_IMPL_BASE_CMPEQ_SSE64 \
  const __m128i eq32 = _mm_cmpeq_epi32(a.r, b.r); \
  const __m128i eq32s = _mm_shuffle_epi32(eq32, 0b10110001); \
  return {.r = _mm_and_si128(eq32, eq32s)};
// case distinctions
#if GREX_X86_64_LEVEL >= 2
#define GREX_CMP_IMPL_BASE_CMPEQ_INT64x2 GREX_CMP_IMPL_BASE_CMPEQ_BASE
#else
#define GREX_CMP_IMPL_BASE_CMPEQ_INT64x2(...) GREX_CMP_IMPL_BASE_CMPEQ_SSE64
#endif
#define GREX_CMP_IMPL_BASE_CMPEQ_INT64x4 GREX_CMP_IMPL_BASE_CMPEQ_BASE
#define GREX_CMP_IMPL_BASE_CMPEQ_INT64x8 GREX_CMP_IMPL_BASE_CMPEQ_BASE
#define GREX_CMP_IMPL_BASE_CMPEQ_INT64(KIND, BITS, SIZE, ...) \
  GREX_CMP_IMPL_BASE_CMPEQ_INT64x##SIZE(KIND, BITS, SIZE, __VA_ARGS__)
#define GREX_CMP_IMPL_BASE_CMPEQ_INT32 GREX_CMP_IMPL_BASE_CMPEQ_BASE
#define GREX_CMP_IMPL_BASE_CMPEQ_INT16 GREX_CMP_IMPL_BASE_CMPEQ_BASE
#define GREX_CMP_IMPL_BASE_CMPEQ_INT8 GREX_CMP_IMPL_BASE_CMPEQ_BASE
#define GREX_CMP_IMPL_BASE_CMPEQ_INT(KIND, BITS, SIZE, ...) \
  GREX_CMP_IMPL_BASE_CMPEQ_INT##BITS(KIND, BITS, SIZE, __VA_ARGS__)
#define GREX_CMP_IMPL_BASE_CMPEQ_f GREX_CMP_IMPL_BASE_CMPEQ_BASE
#define GREX_CMP_IMPL_BASE_CMPEQ_i GREX_CMP_IMPL_BASE_CMPEQ_INT
#define GREX_CMP_IMPL_BASE_CMPEQ_u GREX_CMP_IMPL_BASE_CMPEQ_INT
#define GREX_CMP_IMPL_BASE_cmpeq(KIND, ...) GREX_CMP_IMPL_BASE_CMPEQ_##KIND(KIND, __VA_ARGS__)

// Inequality: Separate cmpneq intrinsic only for f, negated equality for i and u
#define GREX_CMPNEQ_f(BITS, BITPREFIX, REGISTERBITS) \
  return {.r = GREX_KINDCAST(f, i, BITS, REGISTERBITS, \
                             GREX_CAT(BITPREFIX##_cmpneq_, GREX_FP_SUFFIX(f##BITS))(a.r, b.r))};
#define GREX_CMPNEQ_i(...) return logical_not(compare_eq(a, b));
#define GREX_CMPNEQ_u(...) return logical_not(compare_eq(a, b));
#define GREX_CMP_IMPL_BASE_cmpneq(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  GREX_CMPNEQ_##KIND(BITS, BITPREFIX, REGISTERBITS)

// Less
// f, i other than i64 on level 1: Separate cmpgt intrinsics
#define GREX_CMPLT_INTRINSIC(KIND, BITS, BITPREFIX, REGISTERBITS) \
  return {.r = \
            GREX_KINDCAST(KIND, i, BITS, REGISTERBITS, \
                          GREX_CAT(BITPREFIX##_cmpgt_, GREX_EPI_SUFFIX(KIND, BITS))(b.r, a.r))};
// u8/16/32 on level 1, u64 on level 2 and 3: Flip the “sign” bit and use the signed comparison.
#define GREX_CMPLT_UFLIP(KIND, BITS, SIZE, BITPREFIX) \
  const auto signbits = \
    broadcast(u##BITS{1} << u##BITS{BOOST_PP_DEC(BITS)}, type_tag<Vector<KIND##BITS, SIZE>>); \
  const auto a1 = bitwise_xor(a, signbits); \
  const auto b1 = bitwise_xor(b, signbits); \
  return {.r = GREX_CAT(BITPREFIX##_cmpgt_, GREX_EPI_SUFFIX(KIND, BITS))(b1.r, a1.r)};
// u8/16/32 on levels 2 and 3: Inequality with unsigned minimum for u.
#if GREX_X86_64_LEVEL >= 2
#define GREX_CMPLT_UMAX(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  return compare_neq(a, {.r = GREX_CAT(BITPREFIX##_max_, GREX_EPU_SUFFIX(KIND, BITS))(a.r, b.r)});
#else
#define GREX_CMPLT_UMAX(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  GREX_CMPLT_UFLIP(KIND, BITS, SIZE, BITPREFIX)
#endif
// u8/u16 on level 1: saturated difference different from zero
#define GREX_CMPLT_SUBS(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  return compare_neq({.r = GREX_CAT(BITPREFIX##_subs_, GREX_EPU_SUFFIX(KIND, BITS))(b.r, a.r)}, \
                     zeros(type_tag<Vector<KIND##BITS, SIZE>>));
// i64/u64 on level 1: Two 32 bit comparisons
#define GREX_CMPLT_U32X2_u (u64{1} << u64{31}) | (u64{1} << u64{63})
#define GREX_CMPLT_U32X2_i u64{1} << u64{31}
#define GREX_CMPLT_U32X2(KIND) \
  const auto s = _mm_set1_epi64x(GREX_CMPLT_U32X2_##KIND); \
  const auto as = _mm_xor_si128(a.r, s); \
  const auto bs = _mm_xor_si128(b.r, s); \
  const auto lt = _mm_cmplt_epi32(as, bs); \
  const auto eq = _mm_cmpeq_epi32(as, bs); \
  const auto ltlo = _mm_shuffle_epi32(lt, 160); \
  const auto lthi = _mm_shuffle_epi32(lt, 245); \
  const auto eqhi = _mm_shuffle_epi32(eq, 245); \
  return {.r = _mm_or_si128(lthi, _mm_and_si128(eqhi, ltlo))};
// f
#define GREX_CMPLT_f(BITS, SIZE, ...) GREX_CMPLT_INTRINSIC(f, BITS, __VA_ARGS__)
// i
#define GREX_CMPLT_i8(SIZE, ...) GREX_CMPLT_INTRINSIC(i, 8, __VA_ARGS__)
#define GREX_CMPLT_i16(SIZE, ...) GREX_CMPLT_INTRINSIC(i, 16, __VA_ARGS__)
#define GREX_CMPLT_i32(SIZE, ...) GREX_CMPLT_INTRINSIC(i, 32, __VA_ARGS__)
#if GREX_X86_64_LEVEL >= 2
#define GREX_CMPLT_i64(SIZE, ...) GREX_CMPLT_INTRINSIC(i, 64, __VA_ARGS__)
#else
#define GREX_CMPLT_i64(SIZE, ...) GREX_CMPLT_U32X2(i)
#endif
#define GREX_CMPLT_i(BITS, ...) GREX_CMPLT_i##BITS(__VA_ARGS__)
// u
#if GREX_X86_64_LEVEL >= 2
#define GREX_CMPLT_u8(...) GREX_CMPLT_UMAX(u, 8, __VA_ARGS__)
#define GREX_CMPLT_u16(...) GREX_CMPLT_UMAX(u, 16, __VA_ARGS__)
#else
#define GREX_CMPLT_u8(...) GREX_CMPLT_SUBS(u, 8, __VA_ARGS__)
#define GREX_CMPLT_u16(...) GREX_CMPLT_SUBS(u, 16, __VA_ARGS__)
#endif
#define GREX_CMPLT_u32(...) GREX_CMPLT_UMAX(u, 32, __VA_ARGS__)
#if GREX_X86_64_LEVEL >= 2
#define GREX_CMPLT_u64(SIZE, BITPREFIX, REGISTERBITS) GREX_CMPLT_UFLIP(u, 64, SIZE, BITPREFIX)
#else
#define GREX_CMPLT_u64(SIZE, BITPREFIX, REGISTERBITS) GREX_CMPLT_U32X2(u)
#endif
#define GREX_CMPLT_u(BITS, ...) GREX_CMPLT_u##BITS(__VA_ARGS__)
// base
#define GREX_CMP_IMPL_BASE_cmplt(KIND, ...) GREX_CMPLT_##KIND(__VA_ARGS__)

// Greater or equal
// f: Separate cmpgt intrinsics
#define GREX_CMPGE_INTRINSIC(KIND, BITS, BITPREFIX, REGISTERBITS) \
  return {.r = \
            GREX_KINDCAST(KIND, i, BITS, REGISTERBITS, \
                          GREX_CAT(BITPREFIX##_cmpge_, GREX_EPI_SUFFIX(KIND, BITS))(a.r, b.r))};
// Negated less than
#define GREX_CMPGE_NEGATED return logical_not(compare_lt(a, b));
// u8/16/32 on levels 2 and 3: Equality with unsigned minimum for u.
#if GREX_X86_64_LEVEL >= 2
#define GREX_CMPGE_UMAX(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  return compare_eq(a, {.r = GREX_CAT(BITPREFIX##_max_, GREX_EPU_SUFFIX(KIND, BITS))(a.r, b.r)});
#else
#define GREX_CMPGE_UMAX(...) GREX_CMPGE_NEGATED
#endif
// u8/u16 on level 1: saturated difference is from zero
#define GREX_CMPGE_SUBS(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  return compare_eq({.r = GREX_CAT(BITPREFIX##_subs_, GREX_EPU_SUFFIX(KIND, BITS))(b.r, a.r)}, \
                    zeros(type_tag<Vector<KIND##BITS, SIZE>>));
// f
#define GREX_CMPGE_f(BITS, SIZE, ...) GREX_CMPGE_INTRINSIC(f, BITS, __VA_ARGS__)
// i
#define GREX_CMPGE_i(...) GREX_CMPGE_NEGATED
// u
#if GREX_X86_64_LEVEL >= 2
#define GREX_CMPGE_u8(...) GREX_CMPGE_UMAX(u, 8, __VA_ARGS__)
#define GREX_CMPGE_u16(...) GREX_CMPGE_UMAX(u, 16, __VA_ARGS__)
#else
#define GREX_CMPGE_u8(...) GREX_CMPGE_SUBS(u, 8, __VA_ARGS__)
#define GREX_CMPGE_u16(...) GREX_CMPGE_SUBS(u, 16, __VA_ARGS__)
#endif
#define GREX_CMPGE_u32(...) GREX_CMPGE_UMAX(u, 32, __VA_ARGS__)
#define GREX_CMPGE_u64(...) GREX_CMPGE_NEGATED
#define GREX_CMPGE_u(BITS, ...) GREX_CMPGE_u##BITS(__VA_ARGS__)
// base
#define GREX_CMP_IMPL_BASE_cmpge(KIND, ...) GREX_CMPGE_##KIND(__VA_ARGS__)

// Base: Case distinction based on comparison type
#define GREX_CMP_IMPL_BASE(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS, CMPNAME, CMPIDX) \
  GREX_CMP_IMPL_BASE_##CMPNAME(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS)

////////////////////////////////////////////////////////////////////////////////////////////////////

// SSE family definitions
#define GREX_CMP_IMPL_128 GREX_CMP_IMPL_BASE
// AVX/AVX2 definitions
#define GREX_CMP_IMPL_256f(BITS, SIZE, BITPREFIX, REGISTERBITS, CMPNAME, CMPIDX) \
  return {.r = \
            GREX_KINDCAST(f, i, BITS, REGISTERBITS, \
                          GREX_CAT(BITPREFIX##_cmp_, GREX_FP_SUFFIX(f##BITS))(a.r, b.r, CMPIDX))};
#define GREX_CMP_IMPL_256i(...) GREX_CMP_IMPL_BASE(i, __VA_ARGS__)
#define GREX_CMP_IMPL_256u(...) GREX_CMP_IMPL_BASE(u, __VA_ARGS__)
#define GREX_CMP_IMPL_256(KIND, ...) GREX_CMP_IMPL_256##KIND(__VA_ARGS__)
// Base macro
#define GREX_CMP_IMPL(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS, CMPNAME, CMPIDX) \
  GREX_CMP_IMPL_##REGISTERBITS(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS, CMPNAME, CMPIDX)

////////////////////////////////////////////////////////////////////////////////////////////////////

#define GREX_CMP_BASE(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS, OPNAME, CMPNAME, CMPIDX) \
  inline Mask<KIND##BITS, SIZE> compare_##OPNAME(Vector<KIND##BITS, SIZE> a, \
                                                 Vector<KIND##BITS, SIZE> b) { \
    GREX_CMP_IMPL(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS, CMPNAME, CMPIDX) \
  }

#if GREX_X86_64_LEVEL >= 4
#define GREX_CMP GREX_CMP_AVX512
#else
#define GREX_CMP GREX_CMP_BASE
#endif

#define GREX_CMP_ALL(REGISTERBITS, BITPREFIX, OPNAME, CMPNAME, CMPIDX) \
  GREX_FOREACH_TYPE(GREX_CMP, REGISTERBITS, BITPREFIX, REGISTERBITS, OPNAME, CMPNAME, CMPIDX)
GREX_FOREACH_X86_64_LEVEL(GREX_CMP_ALL, eq, cmpeq, 0)
GREX_FOREACH_X86_64_LEVEL(GREX_CMP_ALL, neq, cmpneq, 4)
GREX_FOREACH_X86_64_LEVEL(GREX_CMP_ALL, lt, cmplt, 1)
GREX_FOREACH_X86_64_LEVEL(GREX_CMP_ALL, ge, cmpge, 5)

#define GREX_CMP_SUB(NAME) \
  template<Vectorizable T, std::size_t tPart, std::size_t tSize> \
  inline SubMask<T, tPart, tSize> NAME(SubVector<T, tPart, tSize> a, \
                                       SubVector<T, tPart, tSize> b) { \
    return {.full = NAME(a.full, b.full)}; \
  }
GREX_CMP_SUB(compare_eq)
GREX_CMP_SUB(compare_neq)
GREX_CMP_SUB(compare_lt)
GREX_CMP_SUB(compare_ge)

#define GREX_CMP_SUPER(NAME) \
  template<typename THalf> \
  inline auto NAME(SuperVector<THalf> a, SuperVector<THalf> b) { \
    return SuperMask{.lower = NAME(a.lower, b.lower), .upper = NAME(a.upper, b.upper)}; \
  }
GREX_CMP_SUPER(compare_eq)
GREX_CMP_SUPER(compare_neq)
GREX_CMP_SUPER(compare_lt)
GREX_CMP_SUPER(compare_ge)

////////////////////////////////////////////////////////////////////////////////////////////////////

// Mask equality
// Broad masks: Compare 8-bit chunks

#define GREX_MASKEQ_COMPACT(SIZE, BITPREFIX) GREX_CAT(_kxnor_mask, GREX_MMASKSIZE(SIZE))(a.r, b.r)
#define GREX_MASKEQ_BROAD(SIZE, BITPREFIX) BITPREFIX##_cmpeq_epi8(a.r, b.r)
#if GREX_X86_64_LEVEL >= 4
#define GREX_MASKEQ_IMPL GREX_MASKEQ_COMPACT
#else
#define GREX_MASKEQ_IMPL GREX_MASKEQ_BROAD
#endif

#define GREX_MASKEQ(KIND, BITS, SIZE, BITPREFIX) \
  inline Mask<KIND##BITS, SIZE> compare_eq(Mask<KIND##BITS, SIZE> a, Mask<KIND##BITS, SIZE> b) { \
    return {.r = GREX_MASKEQ_IMPL(SIZE, BITPREFIX)}; \
  }
#define GREX_MASKEQ_ALL(REGISTERBITS, BITPREFIX) \
  GREX_FOREACH_TYPE(GREX_MASKEQ, REGISTERBITS, BITPREFIX)
GREX_FOREACH_X86_64_LEVEL(GREX_MASKEQ_ALL)
GREX_SUBMASK_BINARY(compare_eq)
GREX_SUPERMASK_BINARY(compare_eq)

} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_COMPARE_HPP
