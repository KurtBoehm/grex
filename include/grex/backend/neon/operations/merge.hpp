// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_NEON_OPERATIONS_MERGE_HPP
#define INCLUDE_GREX_BACKEND_NEON_OPERATIONS_MERGE_HPP

#include <cstddef>

#include <arm_neon.h>

#include "grex/backend/base.hpp"
#include "grex/backend/choosers.hpp"
#include "grex/backend/defs.hpp" // IWYU pragma: keep
#include "grex/backend/macros/for-each.hpp"
#include "grex/backend/macros/math.hpp"
#include "grex/backend/neon/operations/mask-convert.hpp"
#include "grex/backend/neon/operations/reinterpret.hpp"
#include "grex/backend/neon/types.hpp" // IWYU pragma: keep
#include "grex/base.hpp"

namespace grex::backend {
// Merging sub-native vectors
#define GREX_MERGE_SUB_II(KIND, BITS, PART, PARTSIZE) \
  const auto r0 = as<u##PARTSIZE>(v0.full.r); \
  const auto r1 = as<u##PARTSIZE>(v1.full.r); \
  const auto zipped = vzip1q_u##PARTSIZE(r0, r1); \
  return VectorFor<KIND##BITS, PART>{as<KIND##BITS>(zipped)};
#define GREX_MERGE_SUB_I(KIND, BITS, PART, PARTSIZE) GREX_MERGE_SUB_II(KIND, BITS, PART, PARTSIZE)
#define GREX_MERGE_SUB(KIND, BITS, PART, SIZE) \
  inline VectorFor<KIND##BITS, GREX_MULTIPLY(PART, 2)> merge(VectorFor<KIND##BITS, PART> v0, \
                                                             VectorFor<KIND##BITS, PART> v1) { \
    GREX_MERGE_SUB_I(KIND, BITS, GREX_MULTIPLY(PART, 2), GREX_MULTIPLY(BITS, PART)) \
  }
GREX_FOREACH_SUB_EXT(GREX_MERGE_SUB)

// Merge to super-native vector
template<Vectorizable T, std::size_t N>
requires(is_supernative<T, 2 * N>)
inline SuperVector<NativeVector<T, N>> merge(NativeVector<T, N> a, NativeVector<T, N> b) {
  return {.lower = a, .upper = b};
}
template<typename Half>
inline SuperVector<SuperVector<Half>> merge(SuperVector<Half> a, SuperVector<Half> b) {
  return {.lower = a, .upper = b};
}

template<AnyMask Mask>
inline MaskFor<typename Mask::VectorValue, Mask::size * 2> merge(Mask a, Mask b) {
  return vector2mask(merge(mask2vector(a), mask2vector(b)), type_tag<typename Mask::VectorValue>);
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_MERGE_HPP
