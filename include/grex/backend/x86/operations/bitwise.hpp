// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_BITWISE_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_BITWISE_HPP

#include <cstddef> // IWYU pragma: keep

#include <immintrin.h>

#include "grex/backend/defs.hpp"
#include "grex/backend/macros/base.hpp"
#include "grex/backend/macros/for-each.hpp"
#include "grex/backend/macros/types.hpp"
#include "grex/backend/x86/instruction-sets.hpp"
#include "grex/backend/x86/macros/for-each.hpp"
#include "grex/backend/x86/macros/intrinsics.hpp"
#include "grex/backend/x86/types.hpp"
#include "grex/base/defs.hpp" // IWYU pragma: keep

#if GREX_X86_64_LEVEL >= 4
#include "grex/backend/macros/math.hpp"
#endif

namespace grex::backend {
#define GREX_NEGATION_BASE_IMPL(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  GREX_CAT(BITPREFIX##_xor_, GREX_SI_SUFFIX(i, BITS, REGISTERBITS))(m.r, BITPREFIX##_set1_epi32(-1))

#define GREX_NEGATION_VEC_IMPL(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS, SUFFIX) \
  inline Vector<KIND##BITS, SIZE> bitwise_not(Vector<KIND##BITS, SIZE> m) { \
    return {.r = GREX_NEGATION_BASE_IMPL(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS)}; \
  } \
  inline Vector<KIND##BITS, SIZE> bitwise_and(Vector<KIND##BITS, SIZE> a, \
                                              Vector<KIND##BITS, SIZE> b) { \
    return {.r = GREX_CAT(BITPREFIX##_and_, SUFFIX)(a.r, b.r)}; \
  } \
  inline Vector<KIND##BITS, SIZE> bitwise_or(Vector<KIND##BITS, SIZE> a, \
                                             Vector<KIND##BITS, SIZE> b) { \
    return {.r = GREX_CAT(BITPREFIX##_or_, SUFFIX)(a.r, b.r)}; \
  } \
  inline Vector<KIND##BITS, SIZE> bitwise_xor(Vector<KIND##BITS, SIZE> a, \
                                              Vector<KIND##BITS, SIZE> b) { \
    return {.r = GREX_CAT(BITPREFIX##_xor_, SUFFIX)(a.r, b.r)}; \
  }
#define GREX_NEGATION_VEC(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  GREX_NEGATION_VEC_IMPL(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS, \
                         GREX_SI_SUFFIX(KIND, BITS, REGISTERBITS))

#if GREX_X86_64_LEVEL >= 4
#define GREX_NEGATION_MASK_AVX512_IMPL_2 __mmask8(m.r ^ 0x03U)
#define GREX_NEGATION_MASK_AVX512_IMPL_4 __mmask8(m.r ^ 0x0FU)
#define GREX_NEGATION_MASK_AVX512_IMPL_8 __mmask8(~m.r)
#define GREX_NEGATION_MASK_AVX512_IMPL_16 __mmask16(~m.r)
#define GREX_NEGATION_MASK_AVX512_IMPL_32 ~m.r
#define GREX_NEGATION_MASK_AVX512_IMPL_64 ~m.r
#define GREX_NEGATION_MASK_IMPL(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  GREX_NEGATION_MASK_AVX512_IMPL_##SIZE
#define GREX_BITWISE_MASK_IMPL(NAME, KNAME, KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  GREX_CAT(_k##KNAME##_mask, GREX_MAX(SIZE, 8))(a.r, b.r)
#else
#define GREX_NEGATION_MASK_IMPL GREX_NEGATION_BASE_IMPL
#define GREX_BITWISE_MASK_IMPL(NAME, KNAME, KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  GREX_CAT(BITPREFIX##_##NAME##_, GREX_SI_SUFFIX(i, BITS, REGISTERBITS))(a.r, b.r)
#endif
#define GREX_BITWISE_MASK(NAME, KNAME, KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  inline Mask<KIND##BITS, SIZE> logical_##NAME(Mask<KIND##BITS, SIZE> a, \
                                               Mask<KIND##BITS, SIZE> b) { \
    return {.r = GREX_BITWISE_MASK_IMPL(NAME, KNAME, KIND, BITS, SIZE, BITPREFIX, REGISTERBITS)}; \
  }
#define GREX_NEGATION_MASK(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  inline Mask<KIND##BITS, SIZE> logical_not(Mask<KIND##BITS, SIZE> m) { \
    return {.r = GREX_NEGATION_MASK_IMPL(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS)}; \
  } \
  GREX_BITWISE_MASK(and, and, KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  GREX_BITWISE_MASK(andnot, andn, KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  GREX_BITWISE_MASK(or, or, KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  GREX_BITWISE_MASK(xor, xor, KIND, BITS, SIZE, BITPREFIX, REGISTERBITS)

#define GREX_NEGATION_ALL(REGISTERBITS, BITPREFIX) \
  GREX_FOREACH_INT_TYPE(GREX_NEGATION_VEC, REGISTERBITS, BITPREFIX, REGISTERBITS) \
  GREX_FOREACH_TYPE(GREX_NEGATION_MASK, REGISTERBITS, BITPREFIX, REGISTERBITS)
GREX_FOREACH_X86_64_LEVEL(GREX_NEGATION_ALL)

GREX_SUBVECTOR_UNARY(bitwise_not)
GREX_SUBVECTOR_BINARY(bitwise_and)
GREX_SUBVECTOR_BINARY(bitwise_or)
GREX_SUBVECTOR_BINARY(bitwise_xor)
GREX_SUBMASK_UNARY(logical_not)
GREX_SUBMASK_BINARY(logical_and)
GREX_SUBMASK_BINARY(logical_andnot)
GREX_SUBMASK_BINARY(logical_or)
GREX_SUBMASK_BINARY(logical_xor)

GREX_SUPERVECTOR_UNARY(bitwise_not)
GREX_SUPERVECTOR_BINARY(bitwise_and)
GREX_SUPERVECTOR_BINARY(bitwise_or)
GREX_SUPERVECTOR_BINARY(bitwise_xor)
GREX_SUPERMASK_UNARY(logical_not)
GREX_SUPERMASK_BINARY(logical_and)
GREX_SUPERMASK_BINARY(logical_andnot)
GREX_SUPERMASK_BINARY(logical_or)
GREX_SUPERMASK_BINARY(logical_xor)
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_BITWISE_HPP
