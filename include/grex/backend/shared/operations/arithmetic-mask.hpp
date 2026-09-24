// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_ARITHMETIC_MASK_HPP
#define INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_ARITHMETIC_MASK_HPP

#include <cstddef>

#include "grex/backend/base.hpp"
#include "grex/base.hpp"

namespace grex::backend {
#define GREX_MASKARITH_SUB(NAME) \
  template<Vectorizable T, std::size_t N> \
  inline SubVector<T, N> NAME(SubMask<T, N> m, SubVector<T, N> a, SubVector<T, N> b) { \
    return SubVector<T, N>{NAME(m.full, a.full, b.full)}; \
  }
GREX_MASKARITH_SUB(mask_add)
GREX_MASKARITH_SUB(mask_subtract)
GREX_MASKARITH_SUB(mask_multiply)
GREX_MASKARITH_SUB(mask_divide)

#define GREX_MASKARITH_SUPER(NAME) \
  template<typename VecHalf, typename MaskHalf> \
  inline SuperVector<VecHalf> NAME(SuperMask<MaskHalf> m, SuperVector<VecHalf> a, \
                                   SuperVector<VecHalf> b) { \
    return { \
      .lower = NAME(m.lower, a.lower, b.lower), \
      .upper = NAME(m.upper, a.upper, b.upper), \
    }; \
  }
GREX_MASKARITH_SUPER(mask_add)
GREX_MASKARITH_SUPER(mask_subtract)
GREX_MASKARITH_SUPER(mask_multiply)
GREX_MASKARITH_SUPER(mask_divide)
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_ARITHMETIC_MASK_HPP
