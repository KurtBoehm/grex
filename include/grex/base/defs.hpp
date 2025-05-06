// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BASE_DEFS_HPP
#define INCLUDE_GREX_BASE_DEFS_HPP

#include <concepts>

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
} // namespace grex

#endif // INCLUDE_GREX_BASE_DEFS_HPP
