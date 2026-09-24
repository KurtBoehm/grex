// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_NEON_OPERATIONS_EXTRACT_SINGLE_HPP
#define INCLUDE_GREX_BACKEND_NEON_OPERATIONS_EXTRACT_SINGLE_HPP

#include <cstddef>

#include <arm_neon.h>

#include "grex/backend/base.hpp"
#include "grex/backend/defs.hpp" // IWYU pragma: keep
#include "grex/backend/macros/for-each.hpp"
#include "grex/backend/neon/macros/types.hpp"
#include "grex/backend/neon/operations/f16.hpp"
#include "grex/backend/neon/types.hpp"
#include "grex/base.hpp"

// vgetq_lane_f16 is always available irrespective of FP16 availability.

namespace grex::backend {
#define GREX_EXTRINGLE(KIND, BITS, SIZE) \
  inline KIND##BITS extract_single(NativeVector<KIND##BITS, SIZE> v) { \
    return GREX_ISUFFIXED(vgetq_lane, KIND, BITS)(from_stored<KIND##BITS>(v.r), 0); \
  }
GREX_FOREACH_TYPE_EXT(GREX_EXTRINGLE, 128)

template<Vectorizable T, std::size_t N>
inline T extract_single(SubVector<T, N> v) {
  return extract_single(v.full);
}
template<typename Half>
inline ValueOf<Half> extract_single(SuperVector<Half> v) {
  return extract_single(v.lower);
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_EXTRACT_SINGLE_HPP
