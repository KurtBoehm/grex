// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_SHUFFLE_AUX_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_SHUFFLE_AUX_HPP

#include <array>
#include <cstddef>

#include "thesauros/ranges.hpp"

#include "grex/base/defs.hpp"

namespace grex::backend {
// compute a compact bit mask mask for zeroing using an AVX-512 mask
template<std::size_t tSize>
constexpr auto zero_mask(const std::array<ShuffleIndex, tSize>& a) {
  u64 mask = 0;

  for (const std::size_t i : thes::range(tSize)) {
    if (is_index(a[i])) {
      mask |= u64{1} << i;
    }
  }
  if constexpr (tSize <= 8) {
    return u8(mask);
  } else if constexpr (tSize <= 16) {
    return u16(mask);
  } else if constexpr (tSize <= 32) {
    return u32(mask);
  } else {
    return mask;
  }
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_SHUFFLE_AUX_HPP
