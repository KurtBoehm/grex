// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_NEON_OPERATIONS_INSERT_STATIC_HPP
#define INCLUDE_GREX_BACKEND_NEON_OPERATIONS_INSERT_STATIC_HPP

#include <arm_neon.h>

#include "grex/backend/defs.hpp"
#include "grex/backend/macros/cast.hpp"
#include "grex/backend/macros/for-each.hpp"
#include "grex/backend/neon/macros/types.hpp"
#include "grex/backend/neon/types.hpp"
#include "grex/base/defs.hpp"

namespace grex::backend {
#define GREX_VEC_SINSERT(KIND, BITS, SIZE) \
  inline Vector<KIND##BITS, SIZE> insert(Vector<KIND##BITS, SIZE> v, AnyIndexTag auto index, \
                                         KIND##BITS value) { \
    const auto ret = GREX_ISUFFIXED(vsetq_lane, KIND, BITS)(value, v.r, index.value); \
    return {.r = ret}; \
  } \
  inline Mask<KIND##BITS, SIZE> insert(Mask<KIND##BITS, SIZE> v, AnyIndexTag auto index, \
                                       bool value) { \
    const auto ret = GREX_ISUFFIXED(vsetq_lane, u, BITS)(GREX_OPCAST(u, BITS, -u##BITS(value)), \
                                                         v.r, index.value); \
    return {.r = ret}; \
  }
GREX_FOREACH_TYPE(GREX_VEC_SINSERT, 128)
} // namespace grex::backend

#include "grex/backend/shared/operations/insert-static.hpp" // IWYU pragma: export

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_INSERT_STATIC_HPP
