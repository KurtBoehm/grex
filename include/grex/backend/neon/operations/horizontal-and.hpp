// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_NEON_OPERATIONS_HORIZONTAL_AND_HPP
#define INCLUDE_GREX_BACKEND_NEON_OPERATIONS_HORIZONTAL_AND_HPP

#include <limits>

#include <arm_neon.h>

#include "grex/backend/defs.hpp"
#include "grex/backend/macros/for-each.hpp"
#include "grex/backend/macros/math.hpp"
#include "grex/backend/neon/types.hpp"

namespace grex::backend {
#define GREX_HAND_8 vminvq_u8(m.r)
#define GREX_HAND_16 vminvq_u16(m.r)
#define GREX_HAND_32 vminvq_u32(m.r)
#define GREX_HAND_64 vminvq_u32(vreinterpretq_u32_u64(m.r))

#define GREX_HAND(KIND, BITS, SIZE) \
  inline bool horizontal_and(Mask<KIND##BITS, SIZE> m) { \
    return GREX_HAND_##BITS != 0; \
  }
GREX_FOREACH_TYPE(GREX_HAND, 128)

// At most 64 bits: Extract as integer and compare with all ones
#define GREX_HAND_SUB_II(KIND, BITS, PART, SIZE, TOTAL) \
  inline bool horizontal_and(SubMask<KIND##BITS, PART, SIZE> m) { \
    const auto lo64 = vget_low_u##BITS(m.registr()); \
    const auto reindeer = vreinterpret_u##TOTAL##_u##BITS(lo64); \
    const auto value = vget_lane_u##TOTAL(reindeer, 0); \
    return value == std::numeric_limits<u##TOTAL>::max(); \
  }
#define GREX_HAND_SUB_I(KIND, BITS, PART, SIZE, TOTAL) \
  GREX_HAND_SUB_II(KIND, BITS, PART, SIZE, TOTAL)
#define GREX_HAND_SUB(KIND, BITS, PART, SIZE) \
  GREX_HAND_SUB_I(KIND, BITS, PART, SIZE, GREX_MULTIPLY(BITS, PART))

GREX_FOREACH_SUB(GREX_HAND_SUB)

template<typename THalf>
inline bool horizontal_and(SuperMask<THalf> m) {
  return horizontal_and(m.lower) && horizontal_and(m.upper);
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_HORIZONTAL_AND_HPP
