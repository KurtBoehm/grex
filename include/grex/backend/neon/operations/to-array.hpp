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

#include "grex/backend/defs.hpp" // IWYU pragma: keep
#include "grex/backend/neon/operations/convert.hpp"
#include "grex/backend/neon/operations/expand.hpp"
#include "grex/backend/neon/operations/store.hpp"
#include "grex/backend/neon/types.hpp"
#include "grex/base.hpp"

namespace grex::backend {
template<AnyVector TVec>
inline void to_array(typename TVec::Value* dst, TVec v) {
  store(dst, v);
}

inline void to_array(bool* dst, Mask<u8, 16> m) {
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

template<AnyMask TMask>
inline void to_array(bool* dst, TMask m) {
  using VectorValue = TMask::VectorValue;
  constexpr std::size_t size = TMask::size;

  if constexpr (!std::same_as<VectorValue, u8>) {
    // elements are bigger than 1 byte → convert to 1-byte mask
    to_array(dst, convert<u8>(m));
  } else if constexpr (is_supernative<VectorValue, size>) {
    // super-native → store in halves
    to_array(dst, m.lower);
    to_array(dst + size / 2, m.upper);
  } else {
    static_assert(false, "Unsupported argument!");
    std::unreachable();
  }
}

template<AnyVector TVec>
inline std::array<typename TVec::Value, TVec::size> to_array(TVec v) {
  std::array<typename TVec::Value, TVec::size> buf{};
  to_array(buf.data(), v);
  return buf;
}
template<AnyMask TMask>
inline std::array<bool, TMask::size> to_array(TMask m) {
  std::array<bool, TMask::size> buf{};
  to_array(buf.data(), m);
  return buf;
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_TO_ARRAY_HPP
