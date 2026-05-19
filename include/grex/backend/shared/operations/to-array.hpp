// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_TO_ARRAY_HPP
#define INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_TO_ARRAY_HPP

#include <array>
#include <concepts>
#include <cstddef>
#include <utility>

#include "grex/backend/active/operations/convert.hpp"
#include "grex/backend/active/sizes.hpp"
#include "grex/backend/base.hpp"
#include "grex/base.hpp"

namespace grex::backend {
template<AnyVector TVec>
inline void to_array(typename TVec::Value* dst, TVec v) {
  store(dst, v);
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

#endif // INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_TO_ARRAY_HPP
