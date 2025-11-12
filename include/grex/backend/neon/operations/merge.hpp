// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_NEON_OPERATIONS_MERGE_HPP
#define INCLUDE_GREX_BACKEND_NEON_OPERATIONS_MERGE_HPP

#include <cstddef>

#include <arm_neon.h>

#include "grex/backend/choosers.hpp"
#include "grex/backend/defs.hpp"
#include "grex/backend/neon/types.hpp" // IWYU pragma: keep
#include "grex/base/defs.hpp"

namespace grex::backend {
// Merging sub-native vectors
#define GREX_MERGE_BASE(KIND, BITS, SIZE, HALF) \
  const auto zipped = GREX_ISUFFIXED(vzip1q, KIND, HALF)(v0.registr(), v1.registr()); \
  return VectorFor<KIND##BITS, SIZE>{zipped};
#define GREX_MERGE_64x2(KIND, BITS, SIZE) GREX_MERGE_BASE(KIND, BITS, SIZE, 64)
#define GREX_MERGE_32x2(KIND, BITS, SIZE) GREX_MERGE_BASE(KIND, BITS, SIZE, 32)
#define GREX_MERGE_16x2(KIND, BITS, SIZE) GREX_MERGE_BASE(KIND, BITS, SIZE, 16)

#define GREX_MERGE_SUB(KIND, BITS, SIZE, IMPL) \
  inline VectorFor<KIND##BITS, SIZE> merge(VectorFor<KIND##BITS, GREX_DIVIDE(SIZE, 2)> v0, \
                                           VectorFor<KIND##BITS, GREX_DIVIDE(SIZE, 2)> v1) { \
    IMPL(KIND, BITS, SIZE) \
  }

// 2×64
GREX_MERGE_SUB(f, 32, 4, GREX_MERGE_64x2)
GREX_MERGE_SUB(i, 32, 4, GREX_MERGE_64x2)
GREX_MERGE_SUB(u, 32, 4, GREX_MERGE_64x2)
GREX_MERGE_SUB(i, 16, 8, GREX_MERGE_64x2)
GREX_MERGE_SUB(u, 16, 8, GREX_MERGE_64x2)
GREX_MERGE_SUB(i, 8, 16, GREX_MERGE_64x2)
GREX_MERGE_SUB(u, 8, 16, GREX_MERGE_64x2)
// 2×32
GREX_MERGE_SUB(i, 16, 4, GREX_MERGE_32x2)
GREX_MERGE_SUB(u, 16, 4, GREX_MERGE_32x2)
GREX_MERGE_SUB(i, 8, 8, GREX_MERGE_32x2)
GREX_MERGE_SUB(u, 8, 8, GREX_MERGE_32x2)
// 2×16
GREX_MERGE_SUB(i, 8, 4, GREX_MERGE_16x2)
GREX_MERGE_SUB(u, 8, 4, GREX_MERGE_16x2)

// Merge to super-native vector
template<Vectorizable T, std::size_t tSize>
requires(is_supernative<T, 2 * tSize>)
inline SuperVector<Vector<T, tSize>> merge(Vector<T, tSize> a, Vector<T, tSize> b) {
  return {.lower = a, .upper = b};
}
template<typename THalf>
inline SuperVector<SuperVector<THalf>> merge(SuperVector<THalf> a, SuperVector<THalf> b) {
  return {.lower = a, .upper = b};
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_MERGE_HPP
