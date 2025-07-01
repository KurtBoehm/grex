// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_HORIZONTAL_AND_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_HORIZONTAL_AND_HPP

#include <immintrin.h>

#include "grex/backend/defs.hpp"
#include "grex/backend/x86/instruction-sets.hpp"
#include "grex/backend/x86/macros/for-each.hpp"
#include "grex/backend/x86/macros/math.hpp"
#include "grex/backend/x86/types.hpp"
#include "grex/base/defs.hpp"

namespace grex::backend {
// A mask for the given number of bits
#define GREX_HAND_BITMASK_2 0x3U
#define GREX_HAND_BITMASK_4 0xFU
#define GREX_HAND_BITMASK_8 0xFFU
#define GREX_HAND_BITMASK_I(BITS) GREX_HAND_BITMASK_##BITS
#define GREX_HAND_BITMASK(BITS) GREX_HAND_BITMASK_I(BITS)
// movemask treats the mask as consisting of byte blocks → compute part bytes
#define GREX_HANDB_MASK(BITS, PART) GREX_HAND_BITMASK(GREX_DIVIDE(GREX_MULTIPLY(BITS, PART), 8))
// a compact mask is just a bit mask
#define GREX_HANDC_MASK GREX_HAND_BITMASK

#define GREX_HAND_TYPE_128 u16
#define GREX_HAND_TYPE_256 u32
#define GREX_HAND_BROAD(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  GREX_HAND_TYPE_##REGISTERBITS(BITPREFIX##_movemask_epi8(m.r)) == GREX_HAND_TYPE_##REGISTERBITS(-1)
#define GREX_HAND_BROAD_SUB(KIND, BITS, PART, SIZE) \
  (u32(_mm_movemask_epi8(m.full.r)) & GREX_HANDB_MASK(BITS, PART)) == GREX_HANDB_MASK(BITS, PART)

#define GREX_HAND_COMPACT_64 u64(m.r) == u64(-1)
#define GREX_HAND_COMPACT_32 u32(m.r) == u32(-1)
#define GREX_HAND_COMPACT_16 u16(m.r) == u16(-1)
#define GREX_HAND_COMPACT_8 u8(m.r) == u8(-1)
#define GREX_HAND_COMPACT_4 (m.r & 0xFU) == 0xF
#define GREX_HAND_COMPACT_2 (m.r & 0x3U) == 0x3
#define GREX_HAND_COMPACT(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) GREX_HAND_COMPACT_##SIZE
#define GREX_HAND_COMPACT_SUB(KIND, BITS, PART, SIZE) \
  u16((m.full.r & GREX_HANDC_MASK(PART))) == GREX_HANDC_MASK(PART)

#if GREX_X86_64_LEVEL >= 4
#define GREX_HAND_IMPL GREX_HAND_COMPACT
#define GREX_HAND_SUB_IMPL GREX_HAND_COMPACT_SUB
#else
#define GREX_HAND_IMPL GREX_HAND_BROAD
#define GREX_HAND_SUB_IMPL GREX_HAND_BROAD_SUB
#endif

#define GREX_HAND(KIND, BITS, SIZE, ...) \
  inline bool horizontal_and(Mask<KIND##BITS, SIZE> m) { \
    return GREX_HAND_IMPL(KIND, BITS, SIZE, __VA_ARGS__); \
  }
#define GREX_HAND_ALL(REGISTERBITS, BITPREFIX) \
  GREX_FOREACH_TYPE(GREX_HAND, REGISTERBITS, BITPREFIX, REGISTERBITS)
GREX_FOREACH_X86_64_LEVEL(GREX_HAND_ALL)

#define GREX_HAND_SUB(KIND, BITS, PART, SIZE) \
  inline bool horizontal_and(SubMask<KIND##BITS, PART, SIZE> m) { \
    return GREX_HAND_SUB_IMPL(KIND, BITS, PART, SIZE); \
  }
GREX_FOREACH_SUB(GREX_HAND_SUB)

template<typename THalf>
inline bool horizontal_and(SuperMask<THalf> m) {
  const bool lower = horizontal_and(m.lower);
  const bool upper = horizontal_and(m.upper);
  return lower && upper;
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_HORIZONTAL_AND_HPP
