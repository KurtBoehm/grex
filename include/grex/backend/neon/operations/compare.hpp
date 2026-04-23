// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_NEON_OPERATIONS_COMPARE_HPP
#define INCLUDE_GREX_BACKEND_NEON_OPERATIONS_COMPARE_HPP

#include <cstddef>

#include <arm_neon.h>

#include "grex/backend/base.hpp"
#include "grex/backend/defs.hpp" // IWYU pragma: keep
#include "grex/backend/macros/for-each.hpp"
#include "grex/backend/macros/types.hpp"
#include "grex/backend/neon/macros/types.hpp"
#include "grex/backend/neon/operations/bitwise.hpp"
#include "grex/backend/neon/types.hpp"
#include "grex/backend/shared/operations/compare.hpp" // IWYU pragma: export
#include "grex/base.hpp"

namespace grex::backend {
#define GREX_CMP_VEC(KIND, BITS, SIZE) \
  inline NativeMask<KIND##BITS, SIZE> compare_eq(NativeVector<KIND##BITS, SIZE> a, \
                                                 NativeVector<KIND##BITS, SIZE> b) { \
    return {.r = GREX_ISUFFIXED(vceqq, KIND, BITS)(a.r, b.r)}; \
  } \
  inline NativeMask<KIND##BITS, SIZE> compare_lt(NativeVector<KIND##BITS, SIZE> a, \
                                                 NativeVector<KIND##BITS, SIZE> b) { \
    return {.r = GREX_ISUFFIXED(vcltq, KIND, BITS)(a.r, b.r)}; \
  } \
  inline NativeMask<KIND##BITS, SIZE> compare_ge(NativeVector<KIND##BITS, SIZE> a, \
                                                 NativeVector<KIND##BITS, SIZE> b) { \
    return {.r = GREX_ISUFFIXED(vcgeq, KIND, BITS)(a.r, b.r)}; \
  } \
  inline NativeMask<KIND##BITS, SIZE> compare_eq(NativeMask<KIND##BITS, SIZE> a, \
                                                 NativeMask<KIND##BITS, SIZE> b) { \
    return {.r = vceqq_u##BITS(a.r, b.r)}; \
  }
GREX_FOREACH_TYPE(GREX_CMP_VEC, 128)
template<Vectorizable T, std::size_t tSize>
inline NativeMask<T, tSize> compare_neq(NativeVector<T, tSize> a, NativeVector<T, tSize> b) {
  return logical_not(compare_eq(a, b));
}

GREX_SUBMASK_BINARY(compare_eq)
GREX_SUPERMASK_BINARY(compare_eq)
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_COMPARE_HPP
