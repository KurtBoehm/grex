// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_NEON_OPERATIONS_CLASSIFICATION_HPP
#define INCLUDE_GREX_BACKEND_NEON_OPERATIONS_CLASSIFICATION_HPP

#include <arm_neon.h>

#include "grex/backend/base.hpp"
#include "grex/backend/defs.hpp" // IWYU pragma: keep
#include "grex/backend/macros/for-each.hpp"
#include "grex/backend/neon/operations/f16.hpp"
#include "grex/backend/neon/types.hpp"
#include "grex/base.hpp"

#if GREX_F16_NATIVE_ARITHMETIC
#include "grex/backend/neon/operations/f16.hpp"
#endif

namespace grex::backend {
#define GREX_ISFIN(KIND, BITS, SIZE, ...) \
  inline NativeMask<KIND##BITS, SIZE> is_finite(NativeVector<KIND##BITS, SIZE> v) { \
    /* The largest finite value. */ \
    const auto inf = vdupq_n_f##BITS(NumericTrait<f##BITS>::infinity()); \
    /* Compare the absolute value with the largest finite value. */ \
    return {.r = vcagtq_f##BITS(inf, from_stored<f##BITS>(v.r))}; \
  }
GREX_FOREACH_FP_TYPE_OPT_EXT(GREX_ISFIN, 128)

#if !GREX_F16_NATIVE_ARITHMETIC
// Binary16: a value is finite exactly if its exponent bits are not all ones.
inline bf16x8 is_finite(f16x8 v) {
  const auto expo = vandq_u16(v.r, vdupq_n_u16(0x7C00));
  return {.r = vmvnq_u16(vceqq_u16(expo, vdupq_n_u16(0x7C00)))};
}
#endif
} // namespace grex::backend

#include "grex/backend/shared/operations/classification.hpp" // IWYU pragma: export

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_CLASSIFICATION_HPP
