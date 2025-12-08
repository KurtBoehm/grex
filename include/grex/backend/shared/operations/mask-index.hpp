// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_MASK_INDEX_HPP
#define INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_MASK_INDEX_HPP

#include <cstddef>

#include "grex/backend/base.hpp"
#include "grex/base.hpp"

namespace grex::backend {
template<Vectorizable T, std::size_t tPart, std::size_t tSize>
inline SubMask<T, tPart, tSize> cutoff_mask(std::size_t i,
                                            TypeTag<SubMask<T, tPart, tSize>> /*tag*/) {
  return SubMask<T, tPart, tSize>{cutoff_mask(i, type_tag<Mask<T, tSize>>)};
}
template<Vectorizable T, std::size_t tPart, std::size_t tSize>
inline SubVector<T, tPart, tSize> cutoff(std::size_t i, SubVector<T, tPart, tSize> v) {
  return SubVector<T, tPart, tSize>{cutoff(i, v.full)};
}

template<typename THalf>
inline SuperMask<THalf> cutoff_mask(std::size_t i, TypeTag<SuperMask<THalf>> /*tag*/) {
  if (i <= THalf::size) {
    return {.lower = cutoff_mask(i, type_tag<THalf>), .upper = zeros(type_tag<THalf>)};
  }
  return {
    .lower = ones(type_tag<THalf>),
    .upper = cutoff_mask(i - THalf::size, type_tag<THalf>),
  };
}
template<typename THalf>
inline SuperVector<THalf> cutoff(std::size_t i, SuperVector<THalf> v) {
  if (i <= THalf::size) {
    return {.lower = cutoff(i, v.lower), .upper = zeros(type_tag<THalf>)};
  }
  return {.lower = v.lower, .upper = cutoff(i - THalf::size, v.upper)};
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_MASK_INDEX_HPP
