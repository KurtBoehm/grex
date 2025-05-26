// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_MERGE_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_MERGE_HPP

#include "grex/backend/defs.hpp"
#include "grex/backend/x86/helpers.hpp"
#include "grex/backend/x86/types.hpp"
#include "grex/base/defs.hpp" // IWYU pragma: keep

namespace grex::backend {
// 128 bit: No merging
#define GREX_MERGE_128(...)
// 256 bit
#define GREX_MERGE_256(KIND, BITS, SIZE, REGISTERBITS, REGISTERBITS2) \
  GREX_CAT(_mm##REGISTERBITS##_set_, GREX_M_SUFFIX(KIND, BITS, REGISTERBITS2))(v1.r, v0.r)
// 512 bit
#define GREX_MERGE_512_f32 _mm512_insertf32x8(_mm512_castps256_ps512(v0.r), v1.r, 1)
#define GREX_MERGE_512_f64 _mm512_insertf64x4(_mm512_castpd256_pd512(v0.r), v1.r, 1)
#define GREX_MERGE_512_f(BITS) GREX_MERGE_512_f##BITS
#define GREX_MERGE_512_i(BITS) _mm512_inserti64x4(_mm512_castsi256_si512(v0.r), v1.r, 1)
#define GREX_MERGE_512_u(BITS) _mm512_inserti64x4(_mm512_castsi256_si512(v0.r), v1.r, 1)
#define GREX_MERGE_512(KIND, BITS, ...) GREX_MERGE_512_##KIND(BITS)

#define GREX_MERGE_IMPL(KIND, BITS, SIZE, IMPL) \
  BOOST_PP_REMOVE_PARENS( \
    BOOST_PP_IF(BOOST_PP_CHECK_EMPTY(IMPL), (), \
                (inline Vector<KIND##BITS, SIZE> merge(Vector<KIND##BITS, GREX_HALF(SIZE)> v0, \
                                                       Vector<KIND##BITS, GREX_HALF(SIZE)> v1) { \
                  return {.r = IMPL}; \
                })))
#define GREX_MERGE(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  GREX_MERGE_IMPL( \
    KIND, BITS, SIZE, \
    GREX_MERGE_##REGISTERBITS(KIND, BITS, BITPREFIX, REGISTERBITS, GREX_HALF(REGISTERBITS)))

#define GREX_MERGE_ALL(REGISTERBITS, BITPREFIX) \
  GREX_FOREACH_TYPE(GREX_MERGE, REGISTERBITS, BITPREFIX, REGISTERBITS)

GREX_FOREACH_X86_64_LEVEL(GREX_MERGE_ALL)
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_MERGE_HPP
