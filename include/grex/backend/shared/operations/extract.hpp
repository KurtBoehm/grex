// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_EXTRACT_HPP
#define INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_EXTRACT_HPP

#include <cstddef>

#include "grex/backend/base.hpp"
#include "grex/base.hpp"

namespace grex::backend {
// SubVector/SubMask
template<Vectorizable T, std::size_t N>
inline T extract(SubVector<T, N> v, std::size_t index) {
  return extract(v.full, index);
}
template<Vectorizable T, std::size_t N>
inline bool extract(SubMask<T, N> v, std::size_t index) {
  return extract(v.full, index);
}

// SuperVector/SuperMask
template<typename Half>
inline Half::Value extract(SuperVector<Half> v, std::size_t i) {
  if (i < Half::size) {
    return extract(v.lower, i);
  }
  return extract(v.upper, i - Half::size);
}
template<typename Half>
inline bool extract(SuperMask<Half> m, std::size_t i) {
  if (i < Half::size) {
    return extract(m.lower, i);
  }
  return extract(m.upper, i - Half::size);
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_EXTRACT_HPP
