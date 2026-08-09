// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_NEON_OPERATIONS_INSERT_HPP
#define INCLUDE_GREX_BACKEND_NEON_OPERATIONS_INSERT_HPP

#include <cstddef>
#include <utility>

#include <arm_neon.h>

#include "grex/backend/base.hpp"
#include "grex/backend/macros/cast.hpp"
#include "grex/backend/macros/for-each.hpp"
#include "grex/backend/macros/repeat.hpp"
#include "grex/backend/neon/macros/types.hpp"
#include "grex/backend/neon/operations/f16.hpp"
#include "grex/backend/neon/types.hpp"

// vsetq_lane_f16 is always available irrespective of FP16 availability.

namespace grex::backend {
#define GREX_INSERT_SWITCH(SIZE, INDEX, KIND, BITS) \
  case INDEX: \
    return {.r = to_stored<KIND##BITS>(GREX_ISUFFIXED(vsetq_lane, KIND, BITS)(value, vr, INDEX))};

#define GREX_INSERT_VEC(KIND, BITS, SIZE) \
  inline NativeVector<KIND##BITS, SIZE> insert(NativeVector<KIND##BITS, SIZE> v, \
                                               std::size_t index, KIND##BITS value) { \
    const auto vr = from_stored<KIND##BITS>(v.r); \
    switch (index) { \
      GREX_REPEAT(SIZE, GREX_INSERT_SWITCH, KIND, BITS) \
      default: std::unreachable(); \
    } \
  }

#define GREX_INSERT_MASK(KIND, BITS, SIZE) \
  inline NativeMask<KIND##BITS, SIZE> insert(NativeMask<KIND##BITS, SIZE> m, std::size_t index, \
                                             bool value) { \
    const u##BITS entry = GREX_OPCAST(u, BITS, -u##BITS(value)); \
    return {.r = insert(NativeVector<u##BITS, SIZE>{m.r}, index, entry).r}; \
  }

GREX_FOREACH_TYPE_EXT(GREX_INSERT_VEC, 128)
GREX_FOREACH_TYPE_EXT(GREX_INSERT_MASK, 128)
} // namespace grex::backend

#include "grex/backend/shared/operations/insert.hpp" // IWYU pragma: export

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_INSERT_HPP
