// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_NEON_OPERATIONS_EXTRACT_SINGLE_HPP
#define INCLUDE_GREX_BACKEND_NEON_OPERATIONS_EXTRACT_SINGLE_HPP

#include <cstddef>

#include <arm_neon.h>

#include "grex/backend/defs.hpp"
#include "grex/backend/macros/for-each.hpp"
#include "grex/backend/neon/macros/types.hpp"
#include "grex/backend/neon/types.hpp"
#include "grex/base/defs.hpp"

namespace grex::backend {
#define GREX_EXTRINGLE(KIND, BITS, SIZE) \
  inline Scalar<KIND##BITS> extract_single(Vector<KIND##BITS, SIZE> v) { \
    return {.value = GREX_ISUFFIXED(vgetq_lane, KIND, BITS)(v.r, 0)}; \
  }
GREX_FOREACH_TYPE(GREX_EXTRINGLE, 128)

template<Vectorizable T, std::size_t tPart, std::size_t tSize>
inline Scalar<T> extract_single(SubVector<T, tPart, tSize> v) {
  return extract_single(v.full);
}
template<typename THalf>
inline Scalar<typename THalf::Value> extract_single(SuperVector<THalf> v) {
  return extract_single(v.lower);
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_EXTRACT_SINGLE_HPP
