// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_NEON_OPERATIONS_SHINGLE_HPP
#define INCLUDE_GREX_BACKEND_NEON_OPERATIONS_SHINGLE_HPP

#include <cstddef>

#include <arm_neon.h>

#include "grex/backend/choosers.hpp"
#include "grex/backend/defs.hpp"
#include "grex/backend/macros/base.hpp"
#include "grex/backend/macros/for-each.hpp"
#include "grex/backend/macros/math.hpp"
#include "grex/backend/neon/macros/types.hpp"
#include "grex/backend/neon/operations/expand.hpp"
#include "grex/backend/neon/types.hpp"
#include "grex/base/defs.hpp"

namespace grex::backend {
#define GREX_VUSHINGLE_f(KIND, BITS, SIZE) \
  const auto vfront = expand_any(front, index_tag<SIZE>); \
  const auto xfront = GREX_ISUFFIXED(vextq, KIND, BITS)(vfront.r, vfront.r, 1); \
  return {.r = GREX_ISUFFIXED(vextq, KIND, BITS)(xfront, v.r, GREX_DECR(SIZE))};
#define GREX_VUSHINGLE_i(KIND, BITS, SIZE) \
  const auto ext = GREX_ISUFFIXED(vextq, KIND, BITS)(v.r, v.r, GREX_DECR(SIZE)); \
  return {.r = GREX_ISUFFIXED(vsetq_lane, KIND, BITS)(front.value, ext, 0)};
#define GREX_VUSHINGLE_u GREX_VUSHINGLE_i

#define GREX_SHINGLE(KIND, BITS, SIZE) \
  inline Vector<KIND##BITS, SIZE> shingle_up(Vector<KIND##BITS, SIZE> v) { \
    return {.r = GREX_ISUFFIXED(vextq, KIND, BITS)(GREX_ISUFFIXED(vdupq_n, KIND, BITS)(0), v.r, \
                                                   GREX_DECR(SIZE))}; \
  } \
  inline Vector<KIND##BITS, SIZE> shingle_up(Scalar<KIND##BITS> front, \
                                             Vector<KIND##BITS, SIZE> v) { \
    GREX_VUSHINGLE_##KIND(KIND, BITS, SIZE) \
  } \
  inline Vector<KIND##BITS, SIZE> shingle_down(Vector<KIND##BITS, SIZE> v) { \
    return {.r = \
              GREX_ISUFFIXED(vextq, KIND, BITS)(v.r, GREX_ISUFFIXED(vdupq_n, KIND, BITS)(0), 1)}; \
  } \
  inline Vector<KIND##BITS, SIZE> shingle_down(Vector<KIND##BITS, SIZE> v, \
                                               Scalar<KIND##BITS> back) { \
    const auto vback = expand_any(Scalar{back}, index_tag<SIZE>).r; \
    return {.r = GREX_ISUFFIXED(vextq, KIND, BITS)(v.r, vback, 1)}; \
  }
GREX_FOREACH_TYPE(GREX_SHINGLE, 128)

#define GREX_ZDSHINGLE_64(KIND, BITS) \
  const auto dst = GREX_ISUFFIXED(vext, KIND, BITS)(low64, GREX_ISUFFIXED(vdup_n, KIND, BITS)(0), 1)
#define GREX_ZDSHINGLE_x2(KIND, BITS) \
  const auto dst = GREX_ISUFFIXED(vtrn2, KIND, BITS)(low64, GREX_ISUFFIXED(vdup_n, KIND, BITS)(0))

#define GREX_ZDSHINGLE_32x2 GREX_ZDSHINGLE_64
#define GREX_ZDSHINGLE_16x4 GREX_ZDSHINGLE_64
#define GREX_ZDSHINGLE_16x2 GREX_ZDSHINGLE_x2
#define GREX_ZDSHINGLE_8x8 GREX_ZDSHINGLE_64
#define GREX_ZDSHINGLE_8x4(KIND, BITS) \
  const auto zero = GREX_ISUFFIXED(vdup_n, KIND, 8)(0); \
  /* [0, 0, 0, 0, 0, v[0], v[1], v[2], v[3]] */ \
  const auto tmp = GREX_ISUFFIXED(vext, KIND, 8)(zero, low64, 4); \
  /* [v[1], v[2], v[3], 0, 0, 0, 0, 0] */ \
  const auto dst = GREX_ISUFFIXED(vext, KIND, 8)(tmp, zero, 5)
#define GREX_ZDSHINGLE_8x2 GREX_ZDSHINGLE_x2

#define GREX_VDSHINGLE_64(KIND, BITS, SIZE) \
  const auto back64 = GREX_ISUFFIXED(vget_low, KIND, BITS)(expand_any(back, index_tag<SIZE>).r); \
  const auto dst = GREX_ISUFFIXED(vext, KIND, BITS)(low64, back64, 1)
#define GREX_VDSHINGLE_x2(KIND, BITS, SIZE) \
  const auto down = GREX_ISUFFIXED(vext, KIND, BITS)(low64, low64, 1); \
  const auto dst = GREX_ISUFFIXED(vset_lane, KIND, BITS)(back.value, down, 1)

#define GREX_VDSHINGLE_32x2 GREX_VDSHINGLE_64
#define GREX_VDSHINGLE_16x4 GREX_VDSHINGLE_64
#define GREX_VDSHINGLE_16x2 GREX_VDSHINGLE_x2
#define GREX_VDSHINGLE_8x8 GREX_VDSHINGLE_64
#define GREX_VDSHINGLE_8x4(KIND, BITS, SIZE) \
  const auto back64 = GREX_ISUFFIXED(vget_low, KIND, BITS)(expand_any(back, index_tag<SIZE>).r); \
  /* [0, 0, 0, 0, 0, v[0], v[1], v[2], v[3]] */ \
  const auto tmp = GREX_ISUFFIXED(vext, KIND, 8)(GREX_ISUFFIXED(vdup_n, KIND, 8)(0), low64, 4); \
  /* [v[1], v[2], v[3], back, -, -, -, -] */ \
  const auto dst = GREX_ISUFFIXED(vext, KIND, 8)(tmp, back64, 5)
#define GREX_VDSHINGLE_8x2 GREX_VDSHINGLE_x2

#define GREX_SHINGLE_SUB(KIND, BITS, PART, SIZE) \
  inline SubVector<KIND##BITS, PART, SIZE> shingle_up(SubVector<KIND##BITS, PART, SIZE> v) { \
    const auto low64 = GREX_ISUFFIXED(vget_low, KIND, BITS)(v.full.r); \
    const auto ext = GREX_ISUFFIXED(vext, KIND, BITS)(GREX_ISUFFIXED(vdup_n, KIND, BITS)(0), \
                                                      low64, GREX_DECR(GREX_DIVIDE(SIZE, 2))); \
    return SubVector<KIND##BITS, PART, SIZE>{expand64(ext)}; \
  } \
  inline SubVector<KIND##BITS, PART, SIZE> shingle_up(Scalar<KIND##BITS> front, \
                                                      SubVector<KIND##BITS, PART, SIZE> v) { \
    const auto low64 = GREX_ISUFFIXED(vget_low, KIND, BITS)(v.full.r); \
    const auto ext = \
      GREX_ISUFFIXED(vext, KIND, BITS)(low64, low64, GREX_DECR(GREX_DIVIDE(SIZE, 2))); \
    const auto set = GREX_ISUFFIXED(vset_lane, KIND, BITS)(front.value, ext, 0); \
    return SubVector<KIND##BITS, PART, SIZE>{expand64(set)}; \
  } \
  inline SubVector<KIND##BITS, PART, SIZE> shingle_down(SubVector<KIND##BITS, PART, SIZE> v) { \
    const auto low64 = GREX_ISUFFIXED(vget_low, KIND, BITS)(v.full.r); \
    GREX_CAT(GREX_ZDSHINGLE_, BITS, x, PART)(KIND, BITS); \
    return SubVector<KIND##BITS, PART, SIZE>{expand64(dst)}; \
  } \
  inline SubVector<KIND##BITS, PART, SIZE> shingle_down(SubVector<KIND##BITS, PART, SIZE> v, \
                                                        Scalar<KIND##BITS> back) { \
    const auto low64 = GREX_ISUFFIXED(vget_low, KIND, BITS)(v.full.r); \
    GREX_CAT(GREX_VDSHINGLE_, BITS, x, PART)(KIND, BITS, SIZE); \
    return SubVector<KIND##BITS, PART, SIZE>{expand64(dst)}; \
  }
GREX_FOREACH_SUB(GREX_SHINGLE_SUB)
} // namespace grex::backend

#include "grex/backend/shared/operations/shingle.hpp" // IWYU pragma: export

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_SHINGLE_HPP
