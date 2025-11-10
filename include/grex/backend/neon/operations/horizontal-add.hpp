// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_NEON_OPERATIONS_HORIZONTAL_ADD_HPP
#define INCLUDE_GREX_BACKEND_NEON_OPERATIONS_HORIZONTAL_ADD_HPP

#include <arm_neon.h>

#include "grex/backend/macros/base.hpp"
#include "grex/backend/neon/macros/types.hpp"
#include "grex/backend/neon/operations/arithmetic.hpp"
#include "grex/backend/neon/types.hpp"

namespace grex::backend {
#define GREX_HADD(KIND, BITS, SIZE) \
  inline KIND##BITS horizontal_add(Vector<KIND##BITS, SIZE> v) { \
    return GREX_CAT(vaddvq_, GREX_ISUFFIX(KIND, BITS))(v.r); \
  }

GREX_FOREACH_TYPE(GREX_HADD, 128)

// SuperVector
template<typename THalf>
inline THalf::Value horizontal_add(SuperVector<THalf> v) {
  return horizontal_add(add(v.lower, v.upper));
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_HORIZONTAL_ADD_HPP
