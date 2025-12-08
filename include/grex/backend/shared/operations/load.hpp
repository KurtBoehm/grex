// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_LOAD_HPP
#define INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_LOAD_HPP

#include <cstddef>

#include "grex/backend/base.hpp"

namespace grex::backend {
// Super-native vectors: Split into halves
template<typename THalf>
inline SuperVector<THalf> load(const typename THalf::Value* ptr,
                               TypeTag<SuperVector<THalf>> /*tag*/) {
  return {
    .lower = load(ptr, type_tag<THalf>),
    .upper = load(ptr + THalf::size, type_tag<THalf>),
  };
}
template<typename THalf>
inline SuperVector<THalf> load_aligned(const typename THalf::Value* ptr,
                                       TypeTag<SuperVector<THalf>> /*tag*/) {
  return {
    .lower = load_aligned(ptr, type_tag<THalf>),
    .upper = load_aligned(ptr + THalf::size, type_tag<THalf>),
  };
}
template<typename THalf>
inline SuperVector<THalf> load_part(const typename THalf::Value* ptr, std::size_t size,
                                    TypeTag<SuperVector<THalf>> /*tag*/) {
  if (size <= THalf::size) {
    return {
      .lower = load_part(ptr, size, type_tag<THalf>),
      .upper = zeros(type_tag<THalf>),
    };
  }
  return {
    .lower = load(ptr, type_tag<THalf>),
    .upper = load_part(ptr + THalf::size, size - THalf::size, type_tag<THalf>),
  };
}
} // namespace grex::backend
#endif // INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_LOAD_HPP
