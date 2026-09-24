// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_LOAD_HPP
#define INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_LOAD_HPP

#include <cstddef>

#include "grex/backend/active/operations/set.hpp"
#include "grex/backend/base.hpp"
#include "grex/base.hpp"

namespace grex::backend {
// Super-native vectors: Split into halves
template<typename Half>
inline SuperVector<Half> load(const typename Half::Value* ptr, TypeTag<SuperVector<Half>> /*tag*/) {
  return {
    .lower = load(ptr, type_tag<Half>),
    .upper = load(ptr + Half::size, type_tag<Half>),
  };
}
template<typename Half>
inline SuperVector<Half> load_aligned(const typename Half::Value* ptr,
                                      TypeTag<SuperVector<Half>> /*tag*/) {
  return {
    .lower = load_aligned(ptr, type_tag<Half>),
    .upper = load_aligned(ptr + Half::size, type_tag<Half>),
  };
}
template<typename Half>
inline SuperVector<Half> load_part(const typename Half::Value* ptr, std::size_t size,
                                   TypeTag<SuperVector<Half>> /*tag*/) {
  if (size <= Half::size) {
    return {
      .lower = load_part(ptr, size, type_tag<Half>),
      .upper = undefined(type_tag<Half>),
    };
  }
  return {
    .lower = load(ptr, type_tag<Half>),
    .upper = load_part(ptr + Half::size, size - Half::size, type_tag<Half>),
  };
}
template<typename Half>
inline SuperVector<Half> load_part(const typename Half::Value* ptr, AnyIndexTag auto size,
                                   TypeTag<SuperVector<Half>> /*tag*/) {
  if constexpr (size <= Half::size) {
    return {
      .lower = load_part(ptr, size, type_tag<Half>),
      .upper = undefined(type_tag<Half>),
    };
  } else {
    return {
      .lower = load(ptr, type_tag<Half>),
      .upper = load_part(ptr + Half::size, index_tag<size - Half::size>, type_tag<Half>),
    };
  }
}
} // namespace grex::backend
#endif // INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_LOAD_HPP
