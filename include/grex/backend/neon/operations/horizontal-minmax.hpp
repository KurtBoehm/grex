// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_NEON_OPERATIONS_HORIZONTAL_MINMAX_HPP
#define INCLUDE_GREX_BACKEND_NEON_OPERATIONS_HORIZONTAL_MINMAX_HPP

#include <algorithm>

#include <arm_neon.h>

#include "grex/backend/neon/macros/types.hpp"
#include "grex/backend/neon/types.hpp"

namespace grex::backend {
#define GREX_HMINMAX_IMPL_BASE(NAME, KIND, BITS, INFIX, VEC) \
  return GREX_ISUFFIXED(v##NAME##v##INFIX, KIND, BITS)(VEC);
#define GREX_HMINMAX_IMPL_INT8 GREX_HMINMAX_IMPL_BASE
#define GREX_HMINMAX_IMPL_INT16 GREX_HMINMAX_IMPL_BASE
#define GREX_HMINMAX_IMPL_INT32 GREX_HMINMAX_IMPL_BASE
#define GREX_HMINMAX_IMPL_INT64(NAME, KIND, BITS, INFIX, VEC) \
  const auto a = GREX_ISUFFIXED(vgetq_lane, KIND, BITS)(VEC, 0); \
  const auto b = GREX_ISUFFIXED(vgetq_lane, KIND, BITS)(VEC, 1); \
  return std::NAME(a, b);

#define GREX_HMINMAX_IMPL_f(NAME, KIND, BITS, INFIX, VEC) return v##NAME##nmv##INFIX##_f##BITS(VEC);
#define GREX_HMINMAX_IMPL_i(NAME, KIND, BITS, INFIX, VEC) \
  GREX_HMINMAX_IMPL_INT##BITS(NAME, KIND, BITS, INFIX, VEC)
#define GREX_HMINMAX_IMPL_u(NAME, KIND, BITS, INFIX, VEC) \
  GREX_HMINMAX_IMPL_INT##BITS(NAME, KIND, BITS, INFIX, VEC)

#define GREX_HMINMAX(KIND, BITS, SIZE) \
  inline KIND##BITS horizontal_min(Vector<KIND##BITS, SIZE> v) { \
    GREX_HMINMAX_IMPL_##KIND(min, KIND, BITS, q, v.r) \
  } \
  inline KIND##BITS horizontal_max(Vector<KIND##BITS, SIZE> v) { \
    GREX_HMINMAX_IMPL_##KIND(max, KIND, BITS, q, v.r) \
  }
GREX_FOREACH_TYPE(GREX_HMINMAX, 128)

// 64 bits: Use 64-bit instructions
#define GREX_HMINMAX_64(KIND, BITS, PART, SIZE) \
  inline KIND##BITS horizontal_min(SubVector<KIND##BITS, PART, SIZE> v) { \
    const auto lo64 = GREX_ISUFFIXED(vget_low, KIND, BITS)(v.registr()); \
    GREX_HMINMAX_IMPL_##KIND(min, KIND, BITS, , lo64) \
  } \
  inline KIND##BITS horizontal_max(SubVector<KIND##BITS, PART, SIZE> v) { \
    const auto lo64 = GREX_ISUFFIXED(vget_low, KIND, BITS)(v.registr()); \
    GREX_HMINMAX_IMPL_##KIND(max, KIND, BITS, , lo64) \
  }

// <64 bits: Use up to two pairwise min/max operations and extract
#define GREX_HMINMAX_PW(NAME, KIND, BITS, PART, SIZE) \
  auto r = GREX_ISUFFIXED(vget_low, KIND, BITS)(v.registr()); \
  r = GREX_ISUFFIXED(vp##NAME, KIND, BITS)(r, r); \
  return GREX_ISUFFIXED(vget_lane, KIND, BITS)(r, 0);
#define GREX_HMINMAX_32_16 GREX_HMINMAX_PW
#define GREX_HMINMAX_32_8(NAME, KIND, BITS, PART, SIZE) \
  auto r = GREX_ISUFFIXED(vget_low, KIND, BITS)(v.registr()); \
  r = GREX_ISUFFIXED(vp##NAME, KIND, 8)(r, r); \
  r = GREX_ISUFFIXED(vp##NAME, KIND, 8)(r, r); \
  return GREX_ISUFFIXED(vget_lane, KIND, 8)(r, 0);
#define GREX_HMINMAX_16_8 GREX_HMINMAX_PW

#define GREX_HMINMAX_TINY(KIND, BITS, PART, SIZE) \
  inline KIND##BITS horizontal_min(SubVector<KIND##BITS, PART, SIZE> v) { \
    GREX_CAT(GREX_HMINMAX_, GREX_MULTIPLY(BITS, PART), _##BITS)(min, KIND, BITS, PART, SIZE) \
  } \
  inline KIND##BITS horizontal_max(SubVector<KIND##BITS, PART, SIZE> v) { \
    GREX_CAT(GREX_HMINMAX_, GREX_MULTIPLY(BITS, PART), _##BITS)(max, KIND, BITS, PART, SIZE) \
  }

// 64 bits
GREX_HMINMAX_64(f, 32, 2, 4)
GREX_HMINMAX_64(i, 32, 2, 4)
GREX_HMINMAX_64(u, 32, 2, 4)
GREX_HMINMAX_64(i, 16, 4, 8)
GREX_HMINMAX_64(u, 16, 4, 8)
GREX_HMINMAX_64(i, 8, 8, 16)
GREX_HMINMAX_64(u, 8, 8, 16)
// 32 bits
GREX_HMINMAX_TINY(i, 16, 2, 8)
GREX_HMINMAX_TINY(u, 16, 2, 8)
GREX_HMINMAX_TINY(i, 8, 4, 16)
GREX_HMINMAX_TINY(u, 8, 4, 16)
// 16 bits
GREX_HMINMAX_TINY(i, 8, 2, 16)
GREX_HMINMAX_TINY(u, 8, 2, 16)

template<typename THalf>
inline THalf::Value horizontal_min(SuperVector<THalf> v) {
  return horizontal_min(min(v.lower, v.upper));
}
template<typename THalf>
inline THalf::Value horizontal_max(SuperVector<THalf> v) {
  return horizontal_max(max(v.lower, v.upper));
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_HORIZONTAL_MINMAX_HPP
