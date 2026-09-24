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
template<AnyVector Vec>
inline void to_array(typename Vec::Value* dst, Vec v) {
  store(dst, v);
}

template<AnyMask Mask>
inline void to_array(bool* dst, Mask m) {
  using VectorValue = Mask::VectorValue;
  constexpr std::size_t size = Mask::size;

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

template<AnyVector Vec>
inline std::array<typename Vec::Value, Vec::size> to_array(Vec v) {
  std::array<typename Vec::Value, Vec::size> buf{};
  to_array(buf.data(), v);
  return buf;
}
template<AnyMask Mask>
inline std::array<bool, Mask::size> to_array(Mask m) {
  std::array<bool, Mask::size> buf{};
  to_array(buf.data(), m);
  return buf;
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_TO_ARRAY_HPP
