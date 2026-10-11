// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_NEON_OPERATIONS_BROADCAST_LANE_HPP
#define INCLUDE_GREX_BACKEND_NEON_OPERATIONS_BROADCAST_LANE_HPP

#include <cassert>
#include <cstddef>

#include "grex/backend/macros/for-each.hpp"
#include "grex/backend/macros/types.hpp"
#include "grex/backend/neon/macros/types.hpp"
#include "grex/backend/neon/types.hpp"
#include "grex/backend/shared/defs.hpp"
#include "grex/base.hpp"

namespace grex::backend {
#define GREX_DUP_I(KIND, BITS, SIZE, REGKIND) \
  template<std::size_t I> \
  inline KIND##BITS##x##SIZE broadcast_lane(KIND##BITS##x##SIZE v, IndexTag<I> /*i*/) { \
    return {.r = GREX_ISUFFIXED(vdupq_laneq, REGKIND, BITS)(v.r, I)}; \
  }
#define GREX_DUP(KIND, BITS, SIZE) GREX_DUP_I(KIND, BITS, SIZE, GREX_REGKIND(KIND, BITS))
GREX_FOREACH_TYPE_EXT(GREX_DUP, 128)

/**
 * One `DUP` (element), which has a latency of 2 and is present on each of the four SIMD execution
 * units on each performance core on Apple Silicon, according to the Apple Silicon CPU Optimization
 * Guide.
 */
template<std::size_t ValueBytes, std::size_t N, std::size_t I>
consteval Cost broadcast_lane_cost() {
  return {.inv_throughput = 0.25, .latency = 2};
}
} // namespace grex::backend

#include "grex/backend/shared/operations/broadcast-lane.hpp" // IWYU pragma: keep

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_BROADCAST_LANE_HPP
