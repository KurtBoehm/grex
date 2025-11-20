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
#include "grex/backend/macros/for-each.hpp"
#include "grex/backend/macros/math.hpp"
#include "grex/backend/neon/operations/mask-convert.hpp"
#include "grex/backend/neon/operations/reinterpret.hpp"
#include "grex/backend/neon/types.hpp" // IWYU pragma: keep
#include "grex/base/defs.hpp"
#include "grex/types.hpp"

namespace grex::backend {
// Merging sub-native vectors
#define GREX_MERGE_SUB_II(KIND, BITS, PART, PARTSIZE) \
  const auto r0 = reinterpret<u##PARTSIZE>(v0.full.r); \
  const auto r1 = reinterpret<u##PARTSIZE>(v1.full.r); \
  const auto zipped = vzip1q_u##PARTSIZE(r0, r1); \
  return VectorFor<KIND##BITS, PART>{reinterpret<KIND##BITS>(zipped)};
#define GREX_MERGE_SUB_I(KIND, BITS, PART, PARTSIZE) GREX_MERGE_SUB_II(KIND, BITS, PART, PARTSIZE)
#define GREX_MERGE_SUB(KIND, BITS, PART, SIZE) \
  inline VectorFor<KIND##BITS, GREX_MULTIPLY(PART, 2)> merge(VectorFor<KIND##BITS, PART> v0, \
                                                             VectorFor<KIND##BITS, PART> v1) { \
    GREX_MERGE_SUB_I(KIND, BITS, GREX_MULTIPLY(PART, 2), GREX_MULTIPLY(BITS, PART)) \
  }
GREX_FOREACH_SUB(GREX_MERGE_SUB)

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

template<AnyMask TMask>
inline MaskFor<typename TMask::VectorValue, TMask::size * 2> merge(TMask a, TMask b) {
  return vector2mask(merge(mask2vector(a), mask2vector(b)), type_tag<typename TMask::VectorValue>);
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_MERGE_HPP
