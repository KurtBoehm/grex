// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_NEON_OPERATIONS_BLEND_HPP
#define INCLUDE_GREX_BACKEND_NEON_OPERATIONS_BLEND_HPP

#include <cstddef>

#include <arm_neon.h>

#include "grex/backend/base.hpp"
#include "grex/backend/macros/for-each.hpp"
#include "grex/backend/neon/macros/types.hpp"
#include "grex/backend/neon/operations/f16.hpp"
#include "grex/backend/neon/operations/reinterpret.hpp"
#include "grex/backend/neon/types.hpp"

namespace grex::backend {
#define GREX_BLENDZ_f(BITS, SIZE) return {.r = as<f##BITS>(vandq_u##BITS(m.r, as<u##BITS>(v1.r)))};
#define GREX_BLENDZ_i(BITS, SIZE) return {.r = vandq_s##BITS(as<i##BITS>(m.r), v1.r)};
#define GREX_BLENDZ_u(BITS, SIZE) return {.r = vandq_u##BITS(m.r, v1.r)};

#define GREX_BLEND(KIND, BITS, SIZE) \
  inline NativeVector<KIND##BITS, SIZE> blend(NativeMask<KIND##BITS, SIZE> m, \
                                              NativeVector<KIND##BITS, SIZE> v0, \
                                              NativeVector<KIND##BITS, SIZE> v1) { \
    const auto v0r = from_stored<KIND##BITS>(v0.r); \
    const auto v1r = from_stored<KIND##BITS>(v1.r); \
    return {.r = to_stored<KIND##BITS>(GREX_ISUFFIXED(vbslq, KIND, BITS)(m.r, v1r, v0r))}; \
  } \
  inline NativeVector<KIND##BITS, SIZE> blend_zero(NativeMask<KIND##BITS, SIZE> m, \
                                                   NativeVector<KIND##BITS, SIZE> v1) { \
    GREX_BLENDZ_##KIND(BITS, SIZE) \
  }

GREX_FOREACH_TYPE_EXT(GREX_BLEND, 128)
} // namespace grex::backend

#include "grex/backend/shared/operations/blend.hpp" // IWYU pragma: export

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_BLEND_HPP
