// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_NEON_OPERATIONS_HORIZONTAL_ADD_HPP
#define INCLUDE_GREX_BACKEND_NEON_OPERATIONS_HORIZONTAL_ADD_HPP

#include <arm_neon.h>

#include "grex/backend/neon/macros/types.hpp"
#include "grex/backend/neon/operations/arithmetic.hpp"
#include "grex/backend/neon/types.hpp"

namespace grex::backend {
#define GREX_HADD(KIND, BITS, SIZE) \
  inline KIND##BITS horizontal_add(Vector<KIND##BITS, SIZE> v) { \
    return GREX_ISUFFIXED(vaddvq, KIND, BITS)(v.r); \
  }
GREX_FOREACH_TYPE(GREX_HADD, 128)

// 64 bits: Use the 64-bit instructions
#define GREX_HADD_64(KIND, BITS, PART, SIZE) \
  inline KIND##BITS horizontal_add(SubVector<KIND##BITS, PART, SIZE> v) { \
    const auto lo64 = GREX_ISUFFIXED(vget_low, KIND, BITS)(v.registr()); \
    return GREX_ISUFFIXED(vaddv, KIND, BITS)(lo64); \
  }

// <64 bits: Use one or two pair-wise additions and extract
#define GREX_HADD_PW(NAME, KIND, BITS, PART, SIZE) \
  auto r = GREX_ISUFFIXED(vget_low, KIND, BITS)(v.registr()); \
  r = GREX_ISUFFIXED(vpadd, KIND, BITS)(r, r); \
  return GREX_ISUFFIXED(vget_lane, KIND, BITS)(r, 0);
#define GREX_HADD_32_16 GREX_HADD_PW
#define GREX_HADD_32_8(NAME, KIND, BITS, PART, SIZE) \
  auto r = GREX_ISUFFIXED(vget_low, KIND, BITS)(v.registr()); \
  r = GREX_ISUFFIXED(vpadd, KIND, 8)(r, r); \
  r = GREX_ISUFFIXED(vpadd, KIND, 8)(r, r); \
  return GREX_ISUFFIXED(vget_lane, KIND, 8)(r, 0);
#define GREX_HADD_16_8 GREX_HADD_PW

#define GREX_HADD_TINY(KIND, BITS, PART, SIZE) \
  inline KIND##BITS horizontal_add(SubVector<KIND##BITS, PART, SIZE> v) { \
    GREX_CAT(GREX_HADD_, GREX_MULTIPLY(BITS, PART), _##BITS)(min, KIND, BITS, PART, SIZE) \
  }

// 64 bits
GREX_HADD_64(f, 32, 2, 4)
GREX_HADD_64(i, 32, 2, 4)
GREX_HADD_64(u, 32, 2, 4)
GREX_HADD_64(i, 16, 4, 8)
GREX_HADD_64(u, 16, 4, 8)
GREX_HADD_64(i, 8, 8, 16)
GREX_HADD_64(u, 8, 8, 16)
// 32 bits
GREX_HADD_TINY(i, 16, 2, 8)
GREX_HADD_TINY(u, 16, 2, 8)
GREX_HADD_TINY(i, 8, 4, 16)
GREX_HADD_TINY(u, 8, 4, 16)
// 16 bits
GREX_HADD_TINY(i, 8, 2, 16)
GREX_HADD_TINY(u, 8, 2, 16)

// super-native: Compute the horizontal sum of the sum of the two halves
template<typename THalf>
inline THalf::Value horizontal_add(SuperVector<THalf> v) {
  return horizontal_add(add(v.lower, v.upper));
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_HORIZONTAL_ADD_HPP
