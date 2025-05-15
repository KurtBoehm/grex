// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_INSTRUCTION_SETS_HPP
#define INCLUDE_GREX_BACKEND_X86_INSTRUCTION_SETS_HPP

#include <array>
#include <bit>
#include <climits>
#include <cstddef>

#include "thesauros/static-ranges.hpp"

#include "grex/base/defs.hpp"

namespace grex::backend {
#ifndef GREX_X86_64_LEVEL
#if defined(__AVX512VL__) && defined(__AVX512DQ__) && defined(__AVX512BW__)
#define GREX_X86_64_LEVEL 4
#elif defined(__AVX2__)
#define GREX_X86_64_LEVEL 3
#elif defined(__SSE4_2__)
#define GREX_X86_64_LEVEL 2
#elif defined(__SSE2__) || defined(__x86_64__)
#define GREX_X86_64_LEVEL 1
#else
#error "Only x86-64 is supported, which implies SSE2!"
#endif
#endif

#if __AVX512VBMI2__
#define GREX_HAS_AVX512VBMI2 true
#else
#define GREX_HAS_AVX512VBMI2 false
#endif

#if GREX_X86_64_LEVEL >= 4
inline constexpr std::array<std::size_t, 3> register_bits{128, 256, 512};
#elif GREX_X86_64_LEVEL >= 3
inline constexpr std::array<std::size_t, 3> register_bits{128, 256};
#else
inline constexpr std::array<std::size_t, 3> register_bits{128};
#endif
template<Vectorizable T>
inline constexpr std::array native_sizes =
  register_bits | thes::star::transform([](std::size_t s) { return s / (sizeof(T) * CHAR_BIT); }) |
  thes::star::to_array;

template<Vectorizable T, std::size_t tSize>
inline constexpr bool is_native = native_sizes<T> | thes::star::contains(tSize);
template<Vectorizable T, std::size_t tSize>
inline constexpr bool is_subnative = tSize > 1 &&
                                     std::has_single_bit(tSize) && tSize < native_sizes<T>.front();
template<Vectorizable T, std::size_t tSize>
inline constexpr bool is_supernative = std::has_single_bit(tSize) && tSize > native_sizes<T>.back();
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_INSTRUCTION_SETS_HPP
