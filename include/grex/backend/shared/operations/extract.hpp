// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_EXTRACT_HPP
#define INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_EXTRACT_HPP

#include <cstddef>

#include "grex/backend/base.hpp"

namespace grex::backend {
// SubVector/SubMask
template<Vectorizable T, std::size_t tPart, std::size_t tSize>
inline T extract(SubVector<T, tPart, tSize> v, std::size_t index) {
  return extract(v.full, index);
}
template<Vectorizable T, std::size_t tPart, std::size_t tSize>
inline bool extract(SubMask<T, tPart, tSize> v, std::size_t index) {
  return extract(v.full, index);
}

// SuperVector/SuperMask
template<typename THalf>
inline THalf::Value extract(SuperVector<THalf> v, std::size_t i) {
  if (i < THalf::size) {
    return extract(v.lower, i);
  }
  return extract(v.upper, i - THalf::size);
}
template<typename THalf>
inline bool extract(SuperMask<THalf> m, std::size_t i) {
  if (i < THalf::size) {
    return extract(m.lower, i);
  }
  return extract(m.upper, i - THalf::size);
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_EXTRACT_HPP
