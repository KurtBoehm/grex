// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_NEON_OPERATIONS_CLASSIFICATION_HPP
#define INCLUDE_GREX_BACKEND_NEON_OPERATIONS_CLASSIFICATION_HPP

#include <limits>

#include <arm_neon.h>

#include "grex/backend/base.hpp"
#include "grex/backend/defs.hpp" // IWYU pragma: keep
#include "grex/backend/macros/for-each.hpp"
#include "grex/backend/neon/types.hpp"
#include "grex/base.hpp"

namespace grex::backend {
#define GREX_ISFIN(KIND, BITS, SIZE, ...) \
  inline NativeMask<KIND##BITS, SIZE> is_finite(NativeVector<KIND##BITS, SIZE> v) { \
    /* the largest finite value */ \
    const auto maxvec = vdupq_n_f##BITS(std::numeric_limits<f##BITS>::max()); \
    /* compare the absolute value with the largest finite value */ \
    return {.r = vcleq_f##BITS(vabsq_f##BITS(v.r), maxvec)}; \
  }
GREX_FOREACH_FP_TYPE(GREX_ISFIN, 128)
} // namespace grex::backend

#include "grex/backend/shared/operations/classification.hpp" // IWYU pragma: export

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_CLASSIFICATION_HPP
