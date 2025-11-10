// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_NEON_OPERATIONS_MASK_INDEX_HPP
#define INCLUDE_GREX_BACKEND_NEON_OPERATIONS_MASK_INDEX_HPP

#include <cstddef>

#include <arm_neon.h>

#include "grex/backend/neon/operations/compare.hpp"
#include "grex/backend/neon/operations/set.hpp"
#include "grex/backend/neon/types.hpp"

namespace grex::backend {
#define GREX_INDEX_MASK(KIND, BITS, SIZE) \
  inline Mask<KIND##BITS, SIZE> cutoff_mask(std::size_t i, TypeTag<Mask<KIND##BITS, SIZE>>) { \
    const auto idxs = indices(type_tag<Vector<u##BITS, SIZE>>); \
    const auto ref = broadcast(u##BITS(i), type_tag<Vector<u##BITS, SIZE>>); \
    return {.r = compare_lt(idxs, ref).r}; \
  }

GREX_FOREACH_TYPE(GREX_INDEX_MASK, 128)
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_MASK_INDEX_HPP
