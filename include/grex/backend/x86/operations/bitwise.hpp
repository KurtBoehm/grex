// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_BITWISE_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_BITWISE_HPP

#include "grex/backend/x86/helpers.hpp"
#include "grex/backend/x86/instruction-sets.hpp"
#include "grex/backend/x86/types.hpp"
#include "grex/base/defs.hpp"

namespace grex::backend {
#define GREX_NEGATION_BASE_IMPL(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  BOOST_PP_CAT(BITPREFIX##_xor_, \
               GREX_SI_SUFFIX(i, BITS, REGISTERBITS))(m.r, BITPREFIX##_set1_epi32(-1))

#define GREX_NEGATION_VEC(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  inline Vector<KIND##BITS, SIZE> negate(Vector<KIND##BITS, SIZE> m) { \
    return {.r = GREX_NEGATION_BASE_IMPL(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS)}; \
  } \
  inline Vector<KIND##BITS, SIZE> bitwise_xor(Vector<KIND##BITS, SIZE> a, \
                                              Vector<KIND##BITS, SIZE> b) { \
    return {.r = \
              BOOST_PP_CAT(BITPREFIX##_xor_, GREX_SI_SUFFIX(KIND, BITS, REGISTERBITS))(a.r, b.r)}; \
  }

#if GREX_X86_64_LEVEL >= 4
#define GREX_NEGATION_MASK_AVX512_IMPL_2 __mmask8(m.r ^ 0x03U)
#define GREX_NEGATION_MASK_AVX512_IMPL_4 __mmask8(m.r ^ 0x0FU)
#define GREX_NEGATION_MASK_AVX512_IMPL_8 __mmask8(~m.r)
#define GREX_NEGATION_MASK_AVX512_IMPL_16 __mmask16(~m.r)
#define GREX_NEGATION_MASK_AVX512_IMPL_32 __mmask32(~m.r)
#define GREX_NEGATION_MASK_AVX512_IMPL_64 __mmask64(~m.r)
#define GREX_NEGATION_MASK_IMPL(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  GREX_NEGATION_MASK_AVX512_IMPL_##SIZE
#else
#define GREX_NEGATION_MASK_IMPL(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  GREX_NEGATION_BASE_IMPL(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS)
#endif
#define GREX_NEGATION_MASK(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  inline Mask<KIND##BITS, SIZE> negate(Mask<KIND##BITS, SIZE> m) { \
    return {.r = GREX_NEGATION_MASK_IMPL(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS)}; \
  }

#define GREX_NEGATION_VEC_ALL(REGISTERBITS, BITPREFIX) \
  GREX_FOREACH_INT_TYPE(GREX_NEGATION_VEC, REGISTERBITS, BITPREFIX, REGISTERBITS)
#define GREX_NEGATION_MASK_ALL(REGISTERBITS, BITPREFIX) \
  GREX_FOREACH_TYPE(GREX_NEGATION_MASK, REGISTERBITS, BITPREFIX, REGISTERBITS)

GREX_FOREACH_X86_64_LEVEL(GREX_NEGATION_VEC_ALL)
GREX_FOREACH_X86_64_LEVEL(GREX_NEGATION_MASK_ALL)
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_BITWISE_HPP
