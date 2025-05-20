// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_HORIZONTAL_AND_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_HORIZONTAL_AND_HPP

#include "grex/backend/x86/helpers.hpp"
#include "grex/backend/x86/instruction-sets.hpp"
#include "grex/backend/x86/types.hpp"
#include "grex/base/defs.hpp" // IWYU pragma: keep

namespace grex::backend {
#define GREX_HAND_BROAD_128 0xFFFF
#define GREX_HAND_BROAD_256 -1
#define GREX_HAND_BROAD(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  BITPREFIX##_movemask_epi8(m.r) == GREX_HAND_BROAD_##REGISTERBITS

#define GREX_HAND_COMPACT_64 i64(m.r) == -i64{1}
#define GREX_HAND_COMPACT_32 m.r == 0xFFFFFFFF
#define GREX_HAND_COMPACT_16 m.r == 0xFFFF
#define GREX_HAND_COMPACT_8 u8(m.r) == 0xFFU
#define GREX_HAND_COMPACT_4 (m.r & 0xFU) == 0xF
#define GREX_HAND_COMPACT_2 (m.r & 0x3U) == 0x3
#define GREX_HAND_COMPACT(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) GREX_HAND_COMPACT_##SIZE

#if GREX_X86_64_LEVEL >= 4
#define GREX_HAND_IMPL GREX_HAND_COMPACT
#else
#define GREX_HAND_IMPL GREX_HAND_BROAD
#endif

#define GREX_HAND(KIND, BITS, SIZE, ...) \
  inline bool horizontal_and(Mask<KIND##BITS, SIZE> m) { \
    return GREX_HAND_IMPL(KIND, BITS, SIZE, __VA_ARGS__); \
  }

#define GREX_HAND_ALL(REGISTERBITS, BITPREFIX) \
  GREX_FOREACH_TYPE(GREX_HAND, REGISTERBITS, BITPREFIX, REGISTERBITS)

GREX_FOREACH_X86_64_LEVEL(GREX_HAND_ALL)
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_HORIZONTAL_AND_HPP
