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

#include "grex/backend/macros/cast.hpp"
#include "grex/backend/macros/for-each.hpp"
#include "grex/backend/macros/repeat.hpp"
#include "grex/backend/neon/macros/types.hpp"
#include "grex/backend/neon/types.hpp"

namespace grex::backend {
#define GREX_INSERT_SWITCH(SIZE, INDEX, INTRINSIC) \
  case INDEX: return {.r = INTRINSIC(value, v.r, INDEX)};

#define GREX_INSERT_VEC(KIND, BITS, SIZE) \
  inline Vector<KIND##BITS, SIZE> insert(Vector<KIND##BITS, SIZE> v, std::size_t index, \
                                         KIND##BITS value) { \
    switch (index) { \
      GREX_REPEAT(SIZE, GREX_INSERT_SWITCH, GREX_ISUFFIXED(vsetq_lane, KIND, BITS)) \
      default: std::unreachable(); \
    } \
  }

#define GREX_INSERT_MASK(KIND, BITS, SIZE) \
  inline Mask<KIND##BITS, SIZE> insert(Mask<KIND##BITS, SIZE> m, std::size_t index, bool value) { \
    const u##BITS entry = GREX_OPCAST(u, BITS, -u##BITS(value)); \
    return {.r = insert(Vector<u##BITS, SIZE>{m.r}, index, entry).r}; \
  }

GREX_FOREACH_TYPE(GREX_INSERT_VEC, 128)
GREX_FOREACH_TYPE(GREX_INSERT_MASK, 128)
} // namespace grex::backend

#include "grex/backend/shared/operations/insert.hpp" // IWYU pragma: export

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_INSERT_HPP
