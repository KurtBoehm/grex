// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_INSERT_HPP
#define INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_INSERT_HPP

#include <cstddef>

#include "grex/backend/base.hpp"
#include "grex/base.hpp"

namespace grex::backend {
// SubVector/SubMask
template<Vectorizable T, std::size_t N>
inline SubVector<T, N> insert(SubVector<T, N> v, std::size_t index, T value) {
  return SubVector<T, N>{insert(v.full, index, value)};
}
template<Vectorizable T, std::size_t N>
inline SubMask<T, N> insert(SubMask<T, N> v, std::size_t index, bool value) {
  return SubMask<T, N>{insert(v.full, index, value)};
}

// SuperVector/SuperMask
template<typename Half>
inline SuperVector<Half> insert(SuperVector<Half> v, std::size_t index,
                                typename Half::Value value) {
  if (index < Half::size) {
    return {.lower = insert(v.lower, index, value), .upper = v.upper};
  }
  return {.lower = v.lower, .upper = insert(v.upper, index - Half::size, value)};
}
template<typename Half>
inline SuperMask<Half> insert(SuperMask<Half> m, std::size_t index, bool value) {
  if (index < Half::size) {
    return {.lower = insert(m.lower, index, value), .upper = m.upper};
  }
  return {.lower = m.lower, .upper = insert(m.upper, index - Half::size, value)};
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_INSERT_HPP
