// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_MULTIBYTE_HPP
#define INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_MULTIBYTE_HPP

#include <array>
#include <cstddef>

#include "grex/backend/defs.hpp"
#include "grex/base.hpp"

namespace grex::backend {
namespace mb {
template<std::size_t tSrc, std::size_t tDst, std::size_t tSize, std::size_t tPart = tSize>
requires((tDst * tSize) == 16)
inline constexpr auto shuffle_indices_128 = static_apply<tSize * tDst>([]<std::size_t... tIdxs>() {
  auto op = []<std::size_t tIdx>(IndexTag<tIdx> /*idx*/) {
    constexpr std::size_t j = tIdx % tDst;
    constexpr std::size_t k = tIdx / tDst;
    return (j < tSrc && k < tPart) ? i8(j + tSrc * k) : i8(-1);
  };
  return std::array{op(index_tag<tIdxs>)...};
});
} // namespace mb

// super-native
template<std::size_t tSrc, typename THalf>
inline SuperVector<THalf> load_multibyte(const u8* ptr, IndexTag<tSrc> /*src*/,
                                         TypeTag<SuperVector<THalf>> /*dst*/) {
  return {
    .lower = load_multibyte(ptr, index_tag<tSrc>, type_tag<THalf>),
    .upper = load_multibyte(ptr + tSrc * THalf::size, index_tag<tSrc>, type_tag<THalf>),
  };
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_MULTIBYTE_HPP
