// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_MULTIBYTE_HPP
#define INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_MULTIBYTE_HPP

#include <array>
#include <cstddef>

#include "grex/backend/base.hpp"
#include "grex/base.hpp"

namespace grex::backend {
namespace mb {
template<std::size_t Src, std::size_t Dst, std::size_t N, std::size_t Part = N>
requires((Dst * N) == 16)
inline constexpr auto shuffle_indices_128 = static_apply<N * Dst>([]<std::size_t... I> {
  auto op = []<std::size_t J>(IndexTag<J> /*idx*/) {
    constexpr std::size_t j = J % Dst;
    constexpr std::size_t k = J / Dst;
    return (j < Src && k < Part) ? static_cast<i8>(j + Src * k) : i8{-1};
  };
  return std::array{op(index_tag<I>)...};
});
} // namespace mb

// super-native
template<std::size_t Src, typename Half>
inline SuperVector<Half> load_multibyte(const u8* ptr, IndexTag<Src> /*src*/,
                                        TypeTag<SuperVector<Half>> /*dst*/) {
  return {
    .lower = load_multibyte(ptr, index_tag<Src>, type_tag<Half>),
    .upper = load_multibyte(ptr + Src * Half::size, index_tag<Src>, type_tag<Half>),
  };
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_MULTIBYTE_HPP
