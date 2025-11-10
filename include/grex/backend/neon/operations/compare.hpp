// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_NEON_OPERATIONS_COMPARE_HPP
#define INCLUDE_GREX_BACKEND_NEON_OPERATIONS_COMPARE_HPP

#include <arm_neon.h>

#include "grex/backend/macros/base.hpp"
#include "grex/backend/neon/macros/types.hpp"
#include "grex/backend/neon/operations/bitwise.hpp"
#include "grex/backend/neon/types.hpp"

namespace grex::backend {
#define GREX_CMP_VEC(KIND, BITS, SIZE) \
  inline Mask<KIND##BITS, SIZE> compare_eq(Vector<KIND##BITS, SIZE> a, \
                                           Vector<KIND##BITS, SIZE> b) { \
    return {.r = GREX_CAT(vceqq_, GREX_ISUFFIX(KIND, BITS))(a.r, b.r)}; \
  } \
  inline Mask<KIND##BITS, SIZE> compare_lt(Vector<KIND##BITS, SIZE> a, \
                                           Vector<KIND##BITS, SIZE> b) { \
    return {.r = GREX_CAT(vcltq_, GREX_ISUFFIX(KIND, BITS))(a.r, b.r)}; \
  } \
  inline Mask<KIND##BITS, SIZE> compare_ge(Vector<KIND##BITS, SIZE> a, \
                                           Vector<KIND##BITS, SIZE> b) { \
    return {.r = GREX_CAT(vcgeq_, GREX_ISUFFIX(KIND, BITS))(a.r, b.r)}; \
  } \
  inline Mask<KIND##BITS, SIZE> compare_eq(Mask<KIND##BITS, SIZE> a, Mask<KIND##BITS, SIZE> b) { \
    return {.r = vceqq_u##BITS(a.r, b.r)}; \
  }
GREX_FOREACH_TYPE(GREX_CMP_VEC, 128)
#define GREX_CMP_SUPER(NAME) \
  template<typename THalf> \
  inline auto NAME(SuperVector<THalf> a, SuperVector<THalf> b) { \
    return SuperMask{.lower = NAME(a.lower, b.lower), .upper = NAME(a.upper, b.upper)}; \
  }
GREX_CMP_SUPER(compare_eq)
GREX_CMP_SUPER(compare_neq)
GREX_CMP_SUPER(compare_lt)
GREX_CMP_SUPER(compare_ge)

template<Vectorizable T, std::size_t tSize>
inline Mask<T, tSize> compare_neq(Vector<T, tSize> a, Vector<T, tSize> b) {
  return logical_not(compare_eq(a, b));
}
GREX_SUBMASK_BINARY(compare_eq)
GREX_SUPERMASK_BINARY(compare_eq)
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_COMPARE_HPP
