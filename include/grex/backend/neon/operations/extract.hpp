// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_NEON_OPERATIONS_EXTRACT_HPP
#define INCLUDE_GREX_BACKEND_NEON_OPERATIONS_EXTRACT_HPP

#include <cstddef>
#include <utility>

#include <arm_neon.h>

#include "grex/backend/macros/base.hpp"
#include "grex/backend/macros/repeat.hpp"
#include "grex/backend/neon/macros/types.hpp"
#include "grex/backend/neon/types.hpp"

namespace grex::backend {
#define GREX_EXTRACT_SWITCH(SIZE, INDEX, INTRINSIC) \
  case INDEX: return {.value = INTRINSIC(v.r, INDEX)};

#define GREX_EXTRACT(KIND, BITS, SIZE) \
  inline Scalar<KIND##BITS> extract(Vector<KIND##BITS, SIZE> v, std::size_t i) { \
    switch (i) { \
      GREX_REPEAT(SIZE, GREX_EXTRACT_SWITCH, GREX_CAT(vgetq_lane_, GREX_ISUFFIX(KIND, BITS))) \
      default: std::unreachable(); \
    } \
  }

GREX_FOREACH_TYPE(GREX_EXTRACT, 128)
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_EXTRACT_HPP
