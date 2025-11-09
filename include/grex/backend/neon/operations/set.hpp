// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_NEON_OPERATIONS_SET_HPP
#define INCLUDE_GREX_BACKEND_NEON_OPERATIONS_SET_HPP

#include <arm_neon.h>

#include "grex/backend/macros/base.hpp"
#include "grex/backend/macros/conditional.hpp"
#include "grex/backend/macros/repeat.hpp"
#include "grex/backend/neon/macros/types.hpp"
#include "grex/backend/neon/types.hpp"
#include "grex/base/defs.hpp"

namespace grex::backend {
#define GREX_DIAGUNDEF_PUSH() \
  _Pragma("GCC diagnostic push") _Pragma("GCC diagnostic ignored \"-Wuninitialized\"")
#define GREX_DIAGUNDEF_POP() _Pragma("GCC diagnostic pop")

template<typename T>
requires(std::is_trivial_v<T>)
inline T make_undefined() {
  GREX_DIAGUNDEF_PUSH()
  T undefined;
  return undefined;
  GREX_DIAGUNDEF_POP()
}

#define GREX_SET_ARG(CNT, IDX, TYPE) GREX_COMMA_IF(IDX) TYPE v##IDX
#define GREX_SET_VAL(CNT, IDX, KIND, BITS) v##IDX GREX_COMMA_IF(IDX)
#define GREX_SET_LANE(CNT, IDX, KIND, BITS) \
  out = GREX_CAT(vsetq_lane_, GREX_ISUFFIX(KIND, BITS))(v##IDX, out, IDX);

#define GREX_SET_VEC(KIND, BITS, SIZE) \
  inline Vector<KIND##BITS, SIZE> zeros(TypeTag<Vector<KIND##BITS, SIZE>>) { \
    return {.r = GREX_CAT(vdupq_n_, GREX_ISUFFIX(KIND, BITS))(0)}; \
  } \
  inline Vector<KIND##BITS, SIZE> undefined(TypeTag<Vector<KIND##BITS, SIZE>>) { \
    return {.r = make_undefined<GREX_REGISTER(KIND, BITS, SIZE)>()}; \
  } \
  inline Vector<KIND##BITS, SIZE> broadcast(KIND##BITS value, TypeTag<Vector<KIND##BITS, SIZE>>) { \
    return {.r = GREX_CAT(vdupq_n_, GREX_ISUFFIX(KIND, BITS))(value)}; \
  } \
  inline Vector<KIND##BITS, SIZE> set(TypeTag<Vector<KIND##BITS, SIZE>>, \
                                      GREX_REPEAT(SIZE, GREX_SET_ARG, KIND##BITS)) { \
    GREX_REGISTER(KIND, BITS, SIZE) out = make_undefined<GREX_REGISTER(KIND, BITS, SIZE)>(); \
    GREX_RREPEAT(SIZE, GREX_SET_LANE, KIND, BITS) \
    return {.r = out}; \
  }
GREX_FOREACH_TYPE(GREX_SET_VEC, 128)
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_SET_HPP
