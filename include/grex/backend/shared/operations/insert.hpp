// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_INSERT_HPP
#define INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_INSERT_HPP

#include <cstddef>

#include "grex/backend/defs.hpp"
#include "grex/base/defs.hpp"

namespace grex::backend {
// SubVector/SubMask
template<Vectorizable T, std::size_t tPart, std::size_t tSize>
inline SubVector<T, tPart, tSize> insert(SubVector<T, tPart, tSize> v, std::size_t index, T value) {
  return SubVector<T, tPart, tSize>{insert(v.full, index, value)};
}
template<Vectorizable T, std::size_t tPart, std::size_t tSize>
inline SubMask<T, tPart, tSize> insert(SubMask<T, tPart, tSize> v, std::size_t index, bool value) {
  return SubMask<T, tPart, tSize>{insert(v.full, index, value)};
}

// SuperVector/SuperMask
template<typename THalf>
inline SuperVector<THalf> insert(SuperVector<THalf> v, std::size_t index,
                                 typename THalf::Value value) {
  if (index < THalf::size) {
    return {.lower = insert(v.lower, index, value), .upper = v.upper};
  }
  return {.lower = v.lower, .upper = insert(v.upper, index - THalf::size, value)};
}
template<typename THalf>
inline SuperMask<THalf> insert(SuperMask<THalf> m, std::size_t index, bool value) {
  if (index < THalf::size) {
    return {.lower = insert(m.lower, index, value), .upper = m.upper};
  }
  return {.lower = m.lower, .upper = insert(m.upper, index - THalf::size, value)};
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_INSERT_HPP
