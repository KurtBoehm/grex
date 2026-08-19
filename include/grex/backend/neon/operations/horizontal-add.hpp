// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_NEON_OPERATIONS_HORIZONTAL_ADD_HPP
#define INCLUDE_GREX_BACKEND_NEON_OPERATIONS_HORIZONTAL_ADD_HPP

#include <arm_neon.h>

#include "grex/backend/base.hpp"
#include "grex/backend/defs.hpp"
#include "grex/backend/macros/base.hpp"
#include "grex/backend/macros/for-each.hpp"
#include "grex/backend/macros/math.hpp"
#include "grex/backend/neon/macros/types.hpp"
#include "grex/backend/neon/types.hpp"
#include "grex/f16.hpp"

#if GREX_F16_NATIVE_ARITHMETIC
#include "grex/backend/neon/operations/f16.hpp"
#endif

namespace grex::backend {
#define GREX_HADD(KIND, BITS, SIZE) \
  inline KIND##BITS horizontal_add(NativeVector<KIND##BITS, SIZE> v) { \
    return GREX_ISUFFIXED(vaddvq, KIND, BITS)(v.r); \
  }
GREX_FOREACH_TYPE(GREX_HADD, 128)

// 64 bits: Use the 64-bit instructions
// <64 bits: Use one or two pair-wise additions and extract

#define GREX_HADD_64(KIND, BITS, PART, SIZE) \
  const auto lo64 = GREX_ISUFFIXED(vget_low, KIND, BITS)(v.registr()); \
  return GREX_ISUFFIXED(vaddv, KIND, BITS)(lo64);
#define GREX_HADD_PW(KIND, BITS, PART, SIZE) \
  auto r = GREX_ISUFFIXED(vget_low, KIND, BITS)(v.registr()); \
  r = GREX_ISUFFIXED(vpadd, KIND, BITS)(r, r); \
  return GREX_ISUFFIXED(vget_lane, KIND, BITS)(r, 0);

#define GREX_HADD_64_32 GREX_HADD_64
#define GREX_HADD_64_16 GREX_HADD_64
#define GREX_HADD_64_8 GREX_HADD_64
#define GREX_HADD_32_16 GREX_HADD_PW
#define GREX_HADD_32_8(KIND, BITS, PART, SIZE) \
  auto r = GREX_ISUFFIXED(vget_low, KIND, BITS)(v.registr()); \
  r = GREX_ISUFFIXED(vpadd, KIND, 8)(r, r); \
  r = GREX_ISUFFIXED(vpadd, KIND, 8)(r, r); \
  return GREX_ISUFFIXED(vget_lane, KIND, 8)(r, 0);
#define GREX_HADD_16_8 GREX_HADD_PW

#define GREX_HADD_SUB(KIND, BITS, PART, SIZE) \
  inline KIND##BITS horizontal_add(SubVector<KIND##BITS, PART> v) { \
    GREX_CAT(GREX_HADD_, GREX_MULTIPLY(BITS, PART), _##BITS)(KIND, BITS, PART, SIZE) \
  }
GREX_FOREACH_SUB(GREX_HADD_SUB)

// Binary16 with FP16: always use a sequence of pairwise additions, as FP16 does not provide
// binary16 horizontal addition.
#if GREX_F16_NATIVE_ARITHMETIC
inline f16 horizontal_add(f16x8 v) {
  const auto vr = as_f16(v.r);
  const auto s4 = vget_low_f16(vpaddq_f16(vr, vr));
  const auto s2 = vpadd_f16(s4, s4);
  return vget_lane_f16(vpadd_f16(s2, s2), 0);
}
inline f16 horizontal_add(SubVector<f16, 4> v) {
  const auto vr = vget_low_f16(as_f16(v.full.r));
  const auto s2 = vpadd_f16(vr, vr);
  return vget_lane_f16(vpadd_f16(s2, s2), 0);
}
inline f16 horizontal_add(SubVector<f16, 2> v) {
  const auto vr = vget_low_f16(as_f16(v.full.r));
  return vget_lane_f16(vpadd_f16(vr, vr), 0);
}
#endif
} // namespace grex::backend

#include "grex/backend/shared/operations/horizontal-add.hpp" // IWYU pragma: export

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_HORIZONTAL_ADD_HPP
