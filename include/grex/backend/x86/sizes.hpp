// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_SIZES_HPP
#define INCLUDE_GREX_BACKEND_X86_SIZES_HPP

#include <array>
#include <bit>
#include <climits>
#include <cstddef>

#include "grex/backend/x86/instruction-sets.hpp"
#include "grex/base.hpp"

namespace grex::backend {
#if GREX_X86_64_LEVEL >= 4
static constexpr std::array<std::size_t, 3> register_bits{128, 256, 512};
#elif GREX_X86_64_LEVEL >= 3
static constexpr std::array<std::size_t, 2> register_bits{128, 256};
#else
static constexpr std::array<std::size_t, 1> register_bits{128};
#endif
static constexpr std::array<std::size_t, register_bits.size()> register_bytes =
  static_apply<register_bits.size()>(
    []<std::size_t... tIdxs> { return std::array{register_bits[tIdxs] / CHAR_BIT...}; });

template<Vectorizable T>
static constexpr std::array native_sizes =
  static_apply<register_bits.size()>([]<std::size_t... tIdxs>() {
    return std::array{(std::get<tIdxs>(register_bits) / (sizeof(T) * CHAR_BIT))...};
  });
template<Vectorizable T>
static constexpr std::size_t min_native_size = native_sizes<T>.front();
template<Vectorizable T>
static constexpr std::size_t max_native_size = native_sizes<T>.back();

template<Vectorizable T, std::size_t tSize>
static constexpr bool is_native = static_apply<native_sizes<T>.size()>(
  []<std::size_t... tIdxs>() { return (... || (tSize == std::get<tIdxs>(native_sizes<T>))); });
template<Vectorizable T, std::size_t tSize>
static constexpr bool is_subnative = tSize > 1 &&
                                     std::has_single_bit(tSize) && tSize < min_native_size<T>;
template<Vectorizable T, std::size_t tSize>
static constexpr bool is_supernative = std::has_single_bit(tSize) && tSize > max_native_size<T>;
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_SIZES_HPP
