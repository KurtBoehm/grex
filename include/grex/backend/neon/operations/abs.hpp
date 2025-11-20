// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_NEON_OPERATIONS_ABS_HPP
#define INCLUDE_GREX_BACKEND_NEON_OPERATIONS_ABS_HPP

#include <arm_neon.h>

#include "grex/backend/macros/for-each.hpp"
#include "grex/backend/macros/types.hpp"
#include "grex/backend/neon/macros/types.hpp"
#include "grex/backend/neon/types.hpp"

namespace grex::backend {
#define GREX_ABS(KIND, BITS, SIZE) \
  inline Vector<KIND##BITS, SIZE> abs(Vector<KIND##BITS, SIZE> a) { \
    return {.r = GREX_ISUFFIXED(vabsq, KIND, BITS)(a.r)}; \
  }

GREX_FOREACH_FP_TYPE(GREX_ABS, 128)
GREX_FOREACH_SINT_TYPE(GREX_ABS, 128)

GREX_SUBVECTOR_UNARY(abs)
GREX_SUPERVECTOR_UNARY(abs)
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_ABS_HPP
