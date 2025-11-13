// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_STORE_HPP
#define INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_STORE_HPP

#include <cstddef>

#include "grex/backend/defs.hpp"

namespace grex::backend {
// SuperVector
template<typename THalf>
inline void store(typename THalf::Value* dst, SuperVector<THalf> src) {
  store(dst, src.lower);
  store(dst + THalf::size, src.upper);
}
template<typename THalf>
inline void store_aligned(typename THalf::Value* dst, SuperVector<THalf> src) {
  store_aligned(dst, src.lower);
  store_aligned(dst + THalf::size, src.upper);
}
template<typename THalf>
inline void store_part(typename THalf::Value* dst, SuperVector<THalf> src, std::size_t size) {
  if (size <= THalf::size) {
    store_part(dst, src.lower, size);
    return;
  }
  store(dst, src.lower);
  store_part(dst + THalf::size, src.upper, size - THalf::size);
}
} // namespace grex::backend
#endif // INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_STORE_HPP
