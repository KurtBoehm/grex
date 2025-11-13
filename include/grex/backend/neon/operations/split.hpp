// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_NEON_OPERATIONS_SPLIT_HPP
#define INCLUDE_GREX_BACKEND_NEON_OPERATIONS_SPLIT_HPP

#include <cstddef>

#include <arm_neon.h>

#include "grex/backend/choosers.hpp"
#include "grex/backend/macros/math.hpp"
#include "grex/backend/neon/types.hpp" // IWYU pragma: keep

namespace grex::backend {
#define GREX_SPLIT(KIND, BITS, SIZE) \
  inline VectorFor<KIND##BITS, GREX_DIVIDE(SIZE, 2)> get_low(VectorFor<KIND##BITS, SIZE> v) { \
    return VectorFor<KIND##BITS, GREX_DIVIDE(SIZE, 2)>{v.registr()}; \
  } \
  inline VectorFor<KIND##BITS, GREX_DIVIDE(SIZE, 2)> get_high(VectorFor<KIND##BITS, SIZE> v) { \
    const auto r = v.registr(); \
    const auto out = GREX_ISUFFIXED(vextq, KIND, BITS)(r, r, GREX_DIVIDE(SIZE, 2)); \
    return VectorFor<KIND##BITS, GREX_DIVIDE(SIZE, 2)>{out}; \
  }

// 64×2
GREX_SPLIT(f, 32, 4)
GREX_SPLIT(i, 32, 4)
GREX_SPLIT(u, 32, 4)
GREX_SPLIT(i, 16, 8)
GREX_SPLIT(u, 16, 8)
GREX_SPLIT(i, 8, 16)
GREX_SPLIT(u, 8, 16)
// 32×2
GREX_SPLIT(i, 16, 4)
GREX_SPLIT(u, 16, 4)
GREX_SPLIT(i, 8, 8)
GREX_SPLIT(u, 8, 8)
// 16×2
GREX_SPLIT(i, 8, 4)
GREX_SPLIT(u, 8, 4)
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_SPLIT_HPP
