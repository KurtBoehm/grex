// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_SHIFT_HPP
#define INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_SHIFT_HPP

#include <cstddef>

#include "grex/backend/base.hpp"
#include "grex/base.hpp"

namespace grex::backend {
#define GREX_SUBSUPER(NAME) \
  template<typename Half> \
  inline SuperVector<Half> NAME(SuperVector<Half> v, AnyIndexTag auto offset) { \
    return {.lower = NAME(v.lower, offset), .upper = NAME(v.upper, offset)}; \
  } \
  template<IntVectorizable T, std::size_t N> \
  inline SubVector<T, N> NAME(SubVector<T, N> v, AnyIndexTag auto offset) { \
    return SubVector<T, N>{NAME(v.full, offset)}; \
  }

GREX_SUBSUPER(shift_left)
GREX_SUBSUPER(shift_right)
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_SHIFT_HPP
