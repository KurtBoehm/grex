// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_NEON_OPERATIONS_SQRT_HPP
#define INCLUDE_GREX_BACKEND_NEON_OPERATIONS_SQRT_HPP

#include <cmath>

#include <arm_neon.h>

#include "grex/backend/defs.hpp"
#include "grex/backend/macros/for-each.hpp"
#include "grex/backend/macros/types.hpp"
#include "grex/backend/neon/macros/types.hpp"
#include "grex/backend/neon/types.hpp"
#include "grex/base.hpp"

namespace grex::backend {
#define GREX_SQRT(KIND, BITS, SIZE) \
  inline Vector<KIND##BITS, SIZE> sqrt(Vector<KIND##BITS, SIZE> v) { \
    return {.r = GREX_ISUFFIXED(vsqrtq, KIND, BITS)(v.r)}; \
  }
GREX_FOREACH_FP_TYPE(GREX_SQRT, 128)
GREX_SUBVECTOR_UNARY(sqrt)
GREX_SUPERVECTOR_UNARY(sqrt)

// scalar implementations
inline Scalar<f32> sqrt(Scalar<f32> v) {
  return {.value = std::sqrt(v.value)};
}
inline Scalar<f64> sqrt(Scalar<f64> v) {
  return {.value = std::sqrt(v.value)};
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_SQRT_HPP
