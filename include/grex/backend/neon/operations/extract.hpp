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

#include "grex/backend/macros/for-each.hpp"
#include "grex/backend/macros/repeat.hpp"
#include "grex/backend/neon/macros/types.hpp"
#include "grex/backend/neon/types.hpp"
#include "grex/base.hpp"

namespace grex::backend {
#define GREX_EXTRACT_SWITCH(SIZE, INDEX, INTRINSIC) \
  case INDEX: return INTRINSIC(v.r, INDEX);

#define GREX_EXTRACT_VEC(KIND, BITS, SIZE) \
  inline KIND##BITS extract(Vector<KIND##BITS, SIZE> v, std::size_t index) { \
    switch (index) { \
      GREX_REPEAT(SIZE, GREX_EXTRACT_SWITCH, GREX_ISUFFIXED(vgetq_lane, KIND, BITS)) \
      default: std::unreachable(); \
    } \
  } \
  inline KIND##BITS extract(Vector<KIND##BITS, SIZE> v, AnyIndexTag auto index) { \
    return GREX_ISUFFIXED(vgetq_lane, KIND, BITS)(v.r, index.value); \
  }

#define GREX_EXTRACT_MASK(KIND, BITS, SIZE) \
  inline bool extract(Mask<KIND##BITS, SIZE> m, std::size_t i) { \
    return extract(Vector<u##BITS, SIZE>{m.r}, i) != 0; \
  } \
  inline bool extract(Mask<KIND##BITS, SIZE> m, AnyIndexTag auto i) { \
    return extract(Vector<u##BITS, SIZE>{m.r}, i) != 0; \
  }

GREX_FOREACH_TYPE(GREX_EXTRACT_VEC, 128)
GREX_FOREACH_TYPE(GREX_EXTRACT_MASK, 128)
} // namespace grex::backend

#include "grex/backend/shared/operations/extract.hpp"

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_EXTRACT_HPP
