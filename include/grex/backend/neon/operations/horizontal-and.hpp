// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_NEON_OPERATIONS_HORIZONTAL_AND_HPP
#define INCLUDE_GREX_BACKEND_NEON_OPERATIONS_HORIZONTAL_AND_HPP

#include <arm_neon.h>

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
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_HORIZONTAL_AND_HPP
