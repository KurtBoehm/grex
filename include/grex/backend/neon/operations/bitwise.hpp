// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_NEON_OPERATIONS_BITWISE_HPP
#define INCLUDE_GREX_BACKEND_NEON_OPERATIONS_BITWISE_HPP

#include <arm_neon.h>

#include "grex/backend/macros/base.hpp"
#include "grex/backend/macros/types.hpp"
#include "grex/backend/neon/macros/types.hpp"
#include "grex/backend/neon/types.hpp"

namespace grex::backend {
#define GREX_BITWISE_NOT_BASE(KIND, BITS, SIZE) \
  return {.r = GREX_CAT(vmvnq_, GREX_ISUFFIX(KIND, BITS))(a.r)};
#define GREX_BITWISE_NOT_8 GREX_BITWISE_NOT_BASE
#define GREX_BITWISE_NOT_16 GREX_BITWISE_NOT_BASE
#define GREX_BITWISE_NOT_32 GREX_BITWISE_NOT_BASE
#define GREX_BITWISE_NOT_64(KIND, BITS, SIZE) \
  const auto a32 = \
    GREX_CAT(vreinterpretq_, GREX_ISUFFIX(KIND, 32), _, GREX_ISUFFIX(KIND, 64))(a.r); \
  const auto neg = GREX_CAT(vmvnq_, GREX_ISUFFIX(KIND, 32))(a32); \
  return {.r = GREX_CAT(vreinterpretq_, GREX_ISUFFIX(KIND, 64), _, GREX_ISUFFIX(KIND, 32))(neg)};

#define GREX_BITWISE(KIND, BITS, SIZE) \
  inline Vector<KIND##BITS, SIZE> bitwise_not(Vector<KIND##BITS, SIZE> a) { \
    GREX_BITWISE_NOT_##BITS(KIND, BITS, SIZE) \
  } \
  inline Vector<KIND##BITS, SIZE> bitwise_and(Vector<KIND##BITS, SIZE> a, \
                                              Vector<KIND##BITS, SIZE> b) { \
    return {.r = GREX_CAT(vandq_, GREX_ISUFFIX(KIND, BITS))(a.r, b.r)}; \
  } \
  inline Vector<KIND##BITS, SIZE> bitwise_or(Vector<KIND##BITS, SIZE> a, \
                                             Vector<KIND##BITS, SIZE> b) { \
    return {.r = GREX_CAT(vorrq_, GREX_ISUFFIX(KIND, BITS))(a.r, b.r)}; \
  } \
  inline Vector<KIND##BITS, SIZE> bitwise_xor(Vector<KIND##BITS, SIZE> a, \
                                              Vector<KIND##BITS, SIZE> b) { \
    return {.r = GREX_CAT(veorq_, GREX_ISUFFIX(KIND, BITS))(a.r, b.r)}; \
  }
#define GREX_LOGICAL(KIND, BITS, SIZE) \
  inline Mask<KIND##BITS, SIZE> logical_not(Mask<KIND##BITS, SIZE> a) { \
    GREX_BITWISE_NOT_##BITS(u, BITS, SIZE) \
  } \
  inline Mask<KIND##BITS, SIZE> logical_and(Mask<KIND##BITS, SIZE> a, Mask<KIND##BITS, SIZE> b) { \
    return {.r = GREX_CAT(vandq_, GREX_ISUFFIX(u, BITS))(a.r, b.r)}; \
  } \
  inline Mask<KIND##BITS, SIZE> logical_andnot(Mask<KIND##BITS, SIZE> a, \
                                               Mask<KIND##BITS, SIZE> b) { \
    return {.r = GREX_CAT(vbicq_, GREX_ISUFFIX(u, BITS))(a.r, b.r)}; \
  } \
  inline Mask<KIND##BITS, SIZE> logical_or(Mask<KIND##BITS, SIZE> a, Mask<KIND##BITS, SIZE> b) { \
    return {.r = GREX_CAT(vorrq_, GREX_ISUFFIX(u, BITS))(a.r, b.r)}; \
  } \
  inline Mask<KIND##BITS, SIZE> logical_xor(Mask<KIND##BITS, SIZE> a, Mask<KIND##BITS, SIZE> b) { \
    return {.r = GREX_CAT(veorq_, GREX_ISUFFIX(u, BITS))(a.r, b.r)}; \
  }

GREX_FOREACH_INT_TYPE(GREX_BITWISE, 128)
GREX_FOREACH_TYPE(GREX_LOGICAL, 128)

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

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_BITWISE_HPP
