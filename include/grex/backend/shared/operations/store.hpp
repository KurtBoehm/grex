// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_STORE_HPP
#define INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_STORE_HPP

#include <cstddef>

#include "grex/backend/base.hpp"
#include "grex/base.hpp"

namespace grex::backend {
// SuperVector
template<typename Half>
inline void store(typename Half::Value* dst, SuperVector<Half> src) {
  store(dst, src.lower);
  store(dst + Half::size, src.upper);
}
template<typename Half>
inline void store_aligned(typename Half::Value* dst, SuperVector<Half> src) {
  store_aligned(dst, src.lower);
  store_aligned(dst + Half::size, src.upper);
}
template<typename Half>
inline void store_part(typename Half::Value* dst, SuperVector<Half> src, std::size_t size) {
  if (size <= Half::size) {
    store_part(dst, src.lower, size);
    return;
  }
  store(dst, src.lower);
  store_part(dst + Half::size, src.upper, size - Half::size);
}
template<typename Half>
inline void store_part(typename Half::Value* dst, SuperVector<Half> src, AnyIndexTag auto size) {
  if constexpr (size <= Half::size) {
    store_part(dst, src.lower, size);
    return;
  } else {
    store(dst, src.lower);
    store_part(dst + Half::size, src.upper, index_tag<size - Half::size>);
  }
}
} // namespace grex::backend
#endif // INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_STORE_HPP
