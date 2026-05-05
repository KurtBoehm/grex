// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_NEON_OPERATIONS_BITWISE_HPP
#define INCLUDE_GREX_BACKEND_NEON_OPERATIONS_BITWISE_HPP

#include <arm_neon.h>

#include "grex/backend/macros/for-each.hpp"
#include "grex/backend/macros/types.hpp"
#include "grex/backend/neon/macros/types.hpp"
#include "grex/backend/neon/operations/reinterpret.hpp"
#include "grex/backend/neon/types.hpp"

namespace grex::backend {
#define GREX_BITWISE_NOT_BASE(KIND, BITS, SIZE) \
  return {.r = GREX_ISUFFIXED(vmvnq, KIND, BITS)(a.r)};
#define GREX_BITWISE_NOT_8 GREX_BITWISE_NOT_BASE
#define GREX_BITWISE_NOT_16 GREX_BITWISE_NOT_BASE
#define GREX_BITWISE_NOT_32 GREX_BITWISE_NOT_BASE
#define GREX_BITWISE_NOT_64(KIND, BITS, SIZE) \
  const auto neg = GREX_ISUFFIXED(vmvnq, KIND, 32)(as<KIND##32>(a.r)); \
  return {.r = as<KIND##64>(neg)};

#define GREX_BITWISE(KIND, BITS, SIZE) \
  inline NativeVector<KIND##BITS, SIZE> bitwise_not(NativeVector<KIND##BITS, SIZE> a) { \
    GREX_BITWISE_NOT_##BITS(KIND, BITS, SIZE) \
  } \
  inline NativeVector<KIND##BITS, SIZE> bitwise_and(NativeVector<KIND##BITS, SIZE> a, \
                                                    NativeVector<KIND##BITS, SIZE> b) { \
    return {.r = GREX_ISUFFIXED(vandq, KIND, BITS)(a.r, b.r)}; \
  } \
  inline NativeVector<KIND##BITS, SIZE> bitwise_or(NativeVector<KIND##BITS, SIZE> a, \
                                                   NativeVector<KIND##BITS, SIZE> b) { \
    return {.r = GREX_ISUFFIXED(vorrq, KIND, BITS)(a.r, b.r)}; \
  } \
  inline NativeVector<KIND##BITS, SIZE> bitwise_xor(NativeVector<KIND##BITS, SIZE> a, \
                                                    NativeVector<KIND##BITS, SIZE> b) { \
    return {.r = GREX_ISUFFIXED(veorq, KIND, BITS)(a.r, b.r)}; \
  }
#define GREX_LOGICAL(KIND, BITS, SIZE) \
  inline NativeMask<KIND##BITS, SIZE> logical_not(NativeMask<KIND##BITS, SIZE> a) { \
    GREX_BITWISE_NOT_##BITS(u, BITS, SIZE) \
  } \
  inline NativeMask<KIND##BITS, SIZE> logical_and(NativeMask<KIND##BITS, SIZE> a, \
                                                  NativeMask<KIND##BITS, SIZE> b) { \
    return {.r = GREX_ISUFFIXED(vandq, u, BITS)(a.r, b.r)}; \
  } \
  inline NativeMask<KIND##BITS, SIZE> logical_andnot(NativeMask<KIND##BITS, SIZE> a, \
                                                     NativeMask<KIND##BITS, SIZE> b) { \
    return {.r = GREX_ISUFFIXED(vbicq, u, BITS)(b.r, a.r)}; \
  } \
  inline NativeMask<KIND##BITS, SIZE> logical_or(NativeMask<KIND##BITS, SIZE> a, \
                                                 NativeMask<KIND##BITS, SIZE> b) { \
    return {.r = GREX_ISUFFIXED(vorrq, u, BITS)(a.r, b.r)}; \
  } \
  inline NativeMask<KIND##BITS, SIZE> logical_xor(NativeMask<KIND##BITS, SIZE> a, \
                                                  NativeMask<KIND##BITS, SIZE> b) { \
    return {.r = GREX_ISUFFIXED(veorq, u, BITS)(a.r, b.r)}; \
  }

GREX_FOREACH_INT_TYPE(GREX_BITWISE, 128)
GREX_FOREACH_TYPE(GREX_LOGICAL, 128)

GREX_NNVECTOR_UNARY(bitwise_not)
GREX_NNVECTOR_BINARY(bitwise_and)
GREX_NNVECTOR_BINARY(bitwise_or)
GREX_NNVECTOR_BINARY(bitwise_xor)
GREX_NNMASK_UNARY(logical_not)
GREX_NNMASK_BINARY(logical_and)
GREX_NNMASK_BINARY(logical_andnot)
GREX_NNMASK_BINARY(logical_or)
GREX_NNMASK_BINARY(logical_xor)
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_BITWISE_HPP
