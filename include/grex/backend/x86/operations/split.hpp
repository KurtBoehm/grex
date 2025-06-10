// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_SPLIT_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_SPLIT_HPP

#include <immintrin.h>

#include "grex/backend/choosers.hpp"
#include "grex/backend/x86/helpers.hpp"
#include "grex/backend/x86/types.hpp" // IWYU pragma: keep
#include "grex/base/defs.hpp" // IWYU pragma: keep

namespace grex::backend {
// The lower half can always be extracted using a cast
#define GREX_SPLIT_LOWER(KIND, BITS, BITPREFIX, REGISTERBITS) \
  GREX_CAT(BITPREFIX##_cast, GREX_REGISTER_SUFFIX(KIND, BITS, REGISTERBITS), _, \
           GREX_REGISTER_SUFFIX(KIND, BITS, GREX_HALF(REGISTERBITS)))(v.r)

// 128 bit: No splitting
#define GREX_SPLIT_128_0(...)
#define GREX_SPLIT_128_1(...)
// 256 bit
#define GREX_SPLIT_256_0 GREX_SPLIT_LOWER
#define GREX_SPLIT_256_1(KIND, BITS, ...) \
  GREX_CAT(_mm256_extract, GREX_REGISTER_LETTER(KIND), 128_, GREX_SI_SUFFIX(KIND, BITS, 256))(v.r, \
                                                                                              1)
#define GREX_SPLIT_256(KIND, BITS, HALF, ...) GREX_SPLIT_256_##HALF(KIND, BITS, __VA_ARGS__)
// 512 bit
#define GREX_SPLIT_512_0 GREX_SPLIT_LOWER
#define GREX_SPLIT_512_1_f32(...) _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(v.r), 1))
#define GREX_SPLIT_512_1_f64(...) _mm512_extractf64x4_pd(v.r, 1)
#define GREX_SPLIT_512_1_f(KIND, BITS, ...) GREX_SPLIT_512_1_f##BITS(KIND, BITS, __VA_ARGS__)
#define GREX_SPLIT_512_1_i(...) _mm512_extracti64x4_epi64(v.r, 1)
#define GREX_SPLIT_512_1_u(...) _mm512_extracti64x4_epi64(v.r, 1)
#define GREX_SPLIT_512_1(KIND, ...) GREX_SPLIT_512_1_##KIND(KIND, __VA_ARGS__)

// Wrapper macros
#define GREX_SPLIT_HALF_IMPL(KIND, BITS, SIZE, HALF, IMPL) \
  BOOST_PP_REMOVE_PARENS( \
    BOOST_PP_IF(BOOST_PP_CHECK_EMPTY(IMPL), (), \
                (inline Vector<KIND##BITS, GREX_HALF(SIZE)> split( \
                  Vector<KIND##BITS, SIZE> v, IndexTag<HALF>) { return {.r = IMPL}; })))
#define GREX_SPLIT_HALF(KIND, BITS, SIZE, HALF, BITPREFIX, REGISTERBITS) \
  GREX_SPLIT_HALF_IMPL(KIND, BITS, SIZE, HALF, \
                       GREX_SPLIT_##REGISTERBITS##_##HALF(KIND, BITS, BITPREFIX, REGISTERBITS))
#define GREX_SPLIT(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  GREX_SPLIT_HALF(KIND, BITS, SIZE, 0, BITPREFIX, REGISTERBITS) \
  GREX_SPLIT_HALF(KIND, BITS, SIZE, 1, BITPREFIX, REGISTERBITS)
#define GREX_SPLIT_ALL(REGISTERBITS, BITPREFIX) \
  GREX_FOREACH_TYPE(GREX_SPLIT, REGISTERBITS, BITPREFIX, REGISTERBITS)

GREX_FOREACH_X86_64_LEVEL(GREX_SPLIT_ALL)

#define GREX_SPLIT_64x2_0(KIND, BITS, SIZE) v.registr()
#define GREX_SPLIT_64x2_1(KIND, BITS, SIZE) _mm_unpackhi_epi64(v.registr(), _mm_setzero_si128())
#define GREX_SPLIT_32x2_0(KIND, BITS, SIZE) v.registr()
#define GREX_SPLIT_32x2_1(KIND, BITS, SIZE) _mm_shuffle_epi32(v.registr(), 1)
#define GREX_SPLIT_16x2_0(KIND, BITS, SIZE) v.registr()
#define GREX_SPLIT_16x2_1(KIND, BITS, SIZE) _mm_shufflelo_epi16(v.registr(), 1)
#define GREX_SPLIT_SUB(KIND, BITS, SIZE, HALF, IMPL) \
  inline VectorFor<KIND##BITS, GREX_HALF(SIZE)> split(VectorFor<KIND##BITS, SIZE> v, \
                                                      IndexTag<HALF>) { \
    return VectorFor<KIND##BITS, GREX_HALF(SIZE)>{IMPL##_##HALF(KIND, BITS, SIZE)}; \
  }
#define GREX_SPLIT_SUB_ALL(KIND, BITS, SIZE, IMPL) \
  GREX_SPLIT_SUB(KIND, BITS, SIZE, 0, IMPL) \
  GREX_SPLIT_SUB(KIND, BITS, SIZE, 1, IMPL)

// 64×2
GREX_SPLIT_SUB_ALL(i, 32, 4, GREX_SPLIT_64x2)
GREX_SPLIT_SUB_ALL(u, 32, 4, GREX_SPLIT_64x2)
GREX_SPLIT_SUB_ALL(i, 16, 8, GREX_SPLIT_64x2)
GREX_SPLIT_SUB_ALL(u, 16, 8, GREX_SPLIT_64x2)
GREX_SPLIT_SUB_ALL(i, 8, 16, GREX_SPLIT_64x2)
GREX_SPLIT_SUB_ALL(u, 8, 16, GREX_SPLIT_64x2)
// 32×2
GREX_SPLIT_SUB_ALL(i, 16, 4, GREX_SPLIT_32x2)
GREX_SPLIT_SUB_ALL(u, 16, 4, GREX_SPLIT_32x2)
GREX_SPLIT_SUB_ALL(i, 8, 8, GREX_SPLIT_32x2)
GREX_SPLIT_SUB_ALL(u, 8, 8, GREX_SPLIT_32x2)
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_SPLIT_HPP
