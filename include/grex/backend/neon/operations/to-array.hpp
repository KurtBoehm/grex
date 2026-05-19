// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_NEON_OPERATIONS_TO_ARRAY_HPP
#define INCLUDE_GREX_BACKEND_NEON_OPERATIONS_TO_ARRAY_HPP

#include <array>
#include <concepts>
#include <cstddef>
#include <utility>

#include <arm_neon.h>

#include "grex/backend/base.hpp"
#include "grex/backend/defs.hpp" // IWYU pragma: keep
#include "grex/backend/neon/operations/expand.hpp"
#include "grex/backend/neon/operations/store.hpp"
#include "grex/backend/neon/types.hpp"
#include "grex/base.hpp"

namespace grex::backend {
inline void to_array(bool* dst, NativeMask<u8, 16> m) {
  const auto masked = vandq_u8(m.r, vdupq_n_u8(1));
  vst1q_u8(reinterpret_cast<u8*>(dst), masked);
}
inline void to_array(bool* dst, SubMask<u8, 8, 16> m) {
  const auto masked = vand_u8(vget_low_u8(m.full.r), vdup_n_u8(1));
  vst1_u8(reinterpret_cast<u8*>(dst), masked);
}
inline void to_array(bool* dst, SubMask<u8, 4, 16> m) {
  const auto masked = vand_u8(vget_low_u8(m.full.r), vdup_n_u8(1));
  store(reinterpret_cast<u8*>(dst), SubVector<u8, 4, 16>{expand64(masked)});
}
inline void to_array(bool* dst, SubMask<u8, 2, 16> m) {
  const auto masked = vand_u8(vget_low_u8(m.full.r), vdup_n_u8(1));
  store(reinterpret_cast<u8*>(dst), SubVector<u8, 2, 16>{expand64(masked)});
}
} // namespace grex::backend

#include "grex/backend/shared/operations/to-array.hpp" // IWYU pragma: export

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_TO_ARRAY_HPP
