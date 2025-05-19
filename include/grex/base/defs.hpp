// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BASE_DEFS_HPP
#define INCLUDE_GREX_BASE_DEFS_HPP

#include <concepts>
#include <stdexcept>

#include "thesauros/types.hpp"

namespace grex {
using thes::f32;
using thes::f64;
using thes::i16;
using thes::i32;
using thes::i64;
using thes::i8;
using thes::u16;
using thes::u32;
using thes::u64;
using thes::u8;

template<typename T>
concept Vectorizable =
  std::same_as<T, u8> || std::same_as<T, i8> || std::same_as<T, u16> || std::same_as<T, i16> ||
  std::same_as<T, u32> || std::same_as<T, i32> || std::same_as<T, u64> || std::same_as<T, i64> ||
  std::same_as<T, f32> || std::same_as<T, f64>;

enum struct ShuffleIndex : u8 { any = 254, zero = 255 };
inline constexpr ShuffleIndex any_sh = ShuffleIndex::any;
inline constexpr ShuffleIndex zero_sh = ShuffleIndex::zero;
constexpr bool is_index(ShuffleIndex sh) {
  return u8(sh) < u8(any_sh);
}

namespace literals {
consteval ShuffleIndex operator""_sh(unsigned long long int v) {
  if (v < 254) {
    return ShuffleIndex(v);
  }
  throw std::invalid_argument{"Unsupported value!"};
}
} // namespace literals
} // namespace grex

#endif // INCLUDE_GREX_BASE_DEFS_HPP
