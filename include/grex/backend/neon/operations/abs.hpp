// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_NEON_OPERATIONS_ABS_HPP
#define INCLUDE_GREX_BACKEND_NEON_OPERATIONS_ABS_HPP

#include <arm_neon.h>

#include "grex/backend/defs.hpp"
#include "grex/backend/macros/for-each.hpp"
#include "grex/backend/macros/types.hpp"
#include "grex/backend/neon/macros/types.hpp"
#include "grex/backend/neon/operations/f16.hpp"
#include "grex/backend/neon/types.hpp"

namespace grex::backend {
#define GREX_ABS(KIND, BITS, SIZE) \
  inline NativeVector<KIND##BITS, SIZE> abs(NativeVector<KIND##BITS, SIZE> a) { \
    const auto r = GREX_ISUFFIXED(vabsq, KIND, BITS)(from_stored<KIND##BITS>(a.r)); \
    return {.r = to_stored<KIND##BITS>(r)}; \
  }

GREX_FOREACH_FP_TYPE_OPT_EXT(GREX_ABS, 128)
GREX_FOREACH_SINT_TYPE(GREX_ABS, 128)

GREX_NNVECTOR_UNARY(abs)

#if !GREX_F16_NATIVE_ARITHMETIC
// Binary16 emulation: clear the sign bit.
inline f16x8 abs(f16x8 a) {
  return {.r = vandq_u16(a.r, vdupq_n_u16(0x7FFF))};
}
#endif
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_ABS_HPP
