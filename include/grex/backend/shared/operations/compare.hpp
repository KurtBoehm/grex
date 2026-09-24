// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_COMPARE_HPP
#define INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_COMPARE_HPP

#include <cstddef>

#include "grex/backend/base.hpp"
#include "grex/base.hpp"

namespace grex::backend {
#define GREX_NN_CMP(NAME) \
  template<Vectorizable T, std::size_t N> \
  inline SubMask<T, N> NAME(SubVector<T, N> a, SubVector<T, N> b) { \
    return SubMask<T, N>{NAME(a.full, b.full)}; \
  } \
  template<typename Half> \
  inline auto NAME(SuperVector<Half> a, SuperVector<Half> b) { \
    return SuperMask{.lower = NAME(a.lower, b.lower), .upper = NAME(a.upper, b.upper)}; \
  }

GREX_NN_CMP(compare_eq)
GREX_NN_CMP(compare_neq)
GREX_NN_CMP(compare_lt)
GREX_NN_CMP(compare_ge)
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_COMPARE_HPP
