// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_NEON_OPERATIONS_SET_HPP
#define INCLUDE_GREX_BACKEND_NEON_OPERATIONS_SET_HPP

#include <array>

#include <arm_neon.h>

#include "grex/backend/macros/cast.hpp"
#include "grex/backend/macros/conditional.hpp"
#include "grex/backend/macros/equals.hpp"
#include "grex/backend/macros/repeat.hpp"
#include "grex/backend/neon/macros/types.hpp"
#include "grex/backend/neon/operations/expand-register.hpp"
#include "grex/backend/neon/types.hpp"
#include "grex/base/defs.hpp"

namespace grex::backend {
template<typename T>
requires(std::is_trivial_v<T>)
inline T make_undefined() {
  GREX_DIAGNOSTIC_UNINIT_PUSH()
  T undefined;
  return undefined;
  GREX_DIAGNOSTIC_UNINIT_POP()
}

#define GREX_SET_ARG(CNT, IDX, TYPE) GREX_COMMA_IF(IDX) TYPE v##IDX
#define GREX_SET_VAL(CNT, IDX) GREX_COMMA_IF(IDX) v##IDX

#define GREX_SET_LANE_0(CNT, IDX, KIND, BITS) \
  out = GREX_ISUFFIXED(vsetq_lane, KIND, BITS)(v##IDX, out, IDX);
#define GREX_SET_LANE_1(...)
#define GREX_SET_LANE(CNT, IDX, KIND, BITS) \
  GREX_CAT(GREX_SET_LANE_, GREX_EQUALS(IDX, 0))(CNT, IDX, KIND, BITS)

#define GREX_SET_BOOLLANE_0(CNT, IDX, BITS) \
  ext = GREX_ISUFFIXED(vsetq_lane, i, BITS)(i##BITS(v##IDX), ext, IDX);
#define GREX_SET_BOOLLANE_1(...)
#define GREX_SET_BOOLLANE(CNT, IDX, BITS) \
  GREX_CAT(GREX_SET_BOOLLANE_, GREX_EQUALS(IDX, 0))(CNT, IDX, BITS)

#if GREX_GCC || true
#define GREX_SET(KIND, BITS, SIZE) \
  inline Vector<KIND##BITS, SIZE> set(TypeTag<Vector<KIND##BITS, SIZE>>, \
                                      GREX_REPEAT(SIZE, GREX_SET_ARG, KIND##BITS)) { \
    std::array<KIND##BITS, SIZE> data{GREX_REPEAT(SIZE, GREX_SET_VAL)}; \
    return {.r = GREX_ISUFFIXED(vld1q, KIND, BITS)(data.data())}; \
  }
#elif GREX_CLANG
#define GREX_SET(KIND, BITS, SIZE) \
  inline Vector<KIND##BITS, SIZE> set(TypeTag<Vector<KIND##BITS, SIZE>>, \
                                      GREX_REPEAT(SIZE, GREX_SET_ARG, KIND##BITS)) { \
    GREX_REGISTER(KIND, BITS, SIZE) out = expand_register(Scalar{v0}); \
    GREX_RREPEAT(SIZE, GREX_SET_LANE, KIND, BITS) \
    return {.r = out}; \
  }
#endif

GREX_FOREACH_TYPE(GREX_SET, 128)

#define GREX_CREATE(KIND, BITS, SIZE) \
  inline Vector<KIND##BITS, SIZE> zeros(TypeTag<Vector<KIND##BITS, SIZE>>) { \
    return {.r = GREX_ISUFFIXED(vdupq_n, KIND, BITS)(0)}; \
  } \
  inline Vector<KIND##BITS, SIZE> undefined(TypeTag<Vector<KIND##BITS, SIZE>>) { \
    return {.r = make_undefined<GREX_REGISTER(KIND, BITS, SIZE)>()}; \
  } \
  inline Vector<KIND##BITS, SIZE> broadcast(KIND##BITS value, TypeTag<Vector<KIND##BITS, SIZE>>) { \
    return {.r = GREX_ISUFFIXED(vdupq_n, KIND, BITS)(value)}; \
  } \
\
  inline Mask<KIND##BITS, SIZE> zeros(TypeTag<Mask<KIND##BITS, SIZE>>) { \
    return {.r = GREX_ISUFFIXED(vdupq_n, u, BITS)(0)}; \
  } \
  inline Mask<KIND##BITS, SIZE> ones(TypeTag<Mask<KIND##BITS, SIZE>>) { \
    return {.r = GREX_ISUFFIXED(vdupq_n, u, BITS)(u##BITS(-1))}; \
  } \
  inline Mask<KIND##BITS, SIZE> broadcast(bool value, TypeTag<Mask<KIND##BITS, SIZE>>) { \
    const u##BITS entry = GREX_OPCAST(u, BITS, -u##BITS(value)); \
    return {.r = GREX_ISUFFIXED(vdupq_n, u, BITS)(entry)}; \
  } \
  /* Idea: Set to the Boolean values and use a greater-than comparison with zero */ \
  inline Mask<KIND##BITS, SIZE> set(TypeTag<Mask<KIND##BITS, SIZE>>, \
                                    GREX_REPEAT(SIZE, GREX_SET_ARG, bool)) { \
    GREX_REGISTER(i, BITS, SIZE) ext = expand_register(Scalar{i##BITS(v0)}); \
    GREX_RREPEAT(SIZE, GREX_SET_BOOLLANE, BITS) \
    GREX_REGISTER(u, BITS, SIZE) cmp = GREX_ISUFFIXED(vcgtzq, i, BITS)(ext); \
    return {.r = cmp}; \
  }

GREX_FOREACH_TYPE(GREX_CREATE, 128)

#define GREX_SET_SUB(KIND, BITS, PART, SIZE) \
  inline SubVector<KIND##BITS, PART, SIZE> set(TypeTag<SubVector<KIND##BITS, PART, SIZE>>, \
                                               GREX_REPEAT(PART, GREX_SET_ARG, KIND##BITS)) { \
    GREX_REGISTER(KIND, BITS, SIZE) out = expand_register(Scalar{v0}); \
    GREX_RREPEAT(PART, GREX_SET_LANE, KIND, BITS) \
    return SubVector<KIND##BITS, PART, SIZE>{out}; \
  }
GREX_FOREACH_SUB(GREX_SET_SUB)
} // namespace grex::backend

#include "grex/backend/shared/operations/set.hpp"

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_SET_HPP
