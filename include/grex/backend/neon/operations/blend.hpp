// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_NEON_OPERATIONS_BLEND_HPP
#define INCLUDE_GREX_BACKEND_NEON_OPERATIONS_BLEND_HPP

#include <cstddef>

#include <arm_neon.h>

#include "grex/backend/neon/types.hpp"

namespace grex::backend {
#define GREX_BLENDZ_f(BITS, SIZE) \
  const auto iv1 = vreinterpretq_u##BITS##_f##BITS(v1.r); \
  const auto iret = vandq_u##BITS(m.r, iv1); \
  return {.r = vreinterpretq_f##BITS##_u##BITS(iret)};
#define GREX_BLENDZ_i(BITS, SIZE) \
  const auto im = vreinterpretq_s##BITS##_u##BITS(m.r); \
  return {.r = vandq_s##BITS(im, v1.r)};
#define GREX_BLENDZ_u(BITS, SIZE) return {.r = vandq_s##BITS(m.r, v1.r)};

#define GREX_BLEND(KIND, BITS, SIZE) \
  inline Vector<KIND##BITS, SIZE> blend(Mask<KIND##BITS, SIZE> m, Vector<KIND##BITS, SIZE> v0, \
                                        Vector<KIND##BITS, SIZE> v1) { \
    return {.r = GREX_ISUFFIXED(vbslq, KIND, BITS)(m.r, v1.r, v0.r)}; \
  } \
  inline Vector<KIND##BITS, SIZE> blend_zero(Mask<KIND##BITS, SIZE> m, \
                                             Vector<KIND##BITS, SIZE> v1) { \
    GREX_BLENDZ_##KIND(BITS, SIZE) \
  }

GREX_FOREACH_TYPE(GREX_BLEND, 128)
} // namespace grex::backend

#include "grex/backend/shared/operations/blend.hpp"

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_BLEND_HPP
