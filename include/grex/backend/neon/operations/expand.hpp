// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_NEON_OPERATIONS_EXPAND_HPP
#define INCLUDE_GREX_BACKEND_NEON_OPERATIONS_EXPAND_HPP

#include <concepts>
#include <cstddef>

#include <arm_neon.h>

#include "grex/backend/base.hpp"
#include "grex/backend/choosers.hpp"
#include "grex/backend/defs.hpp" // IWYU pragma: keep
#include "grex/backend/macros/for-each.hpp"
#include "grex/backend/neon/operations/expand-register.hpp"
#include "grex/backend/neon/operations/f16.hpp"
#include "grex/backend/neon/operations/insert-static.hpp"
#include "grex/backend/neon/sizes.hpp"
#include "grex/base.hpp"

namespace grex::backend {
#define GREX_EXPAND_ANY(KIND, BITS, SIZE) \
  template<std::same_as<KIND##BITS> T> \
  inline NativeVector<T, SIZE> expand(T x, IndexTag<SIZE> /*size*/, BoolTag<false> /*tag*/) { \
    return {.r = to_stored<T>(expand_register(x))}; \
  }
GREX_FOREACH_TYPE_EXT(GREX_EXPAND_ANY, 128)

template<Vectorizable T, std::size_t tSize>
inline VectorFor<T, tSize> expand(T x, IndexTag<tSize> /*size*/, BoolTag<true> /*tag*/) {
  return insert(zeros(type_tag<VectorFor<T, tSize>>), index_tag<0>, x);
}

// native/super-native → super-native
template<AnyVector TVec, std::size_t tDstSize, bool tZero>
requires(tDstSize > TVec::size && is_supernative<typename TVec::Value, tDstSize> &&
         (AnyNativeVector<TVec> || AnySuperNativeVector<TVec>))
inline VectorFor<typename TVec::Value, tDstSize> expand(TVec v, IndexTag<tDstSize> /*size*/,
                                                        BoolTag<tZero> zero_tag) {
  using Value = TVec::Value;
  using Half = VectorFor<Value, tDstSize / 2>;
  using Out = SuperVector<Half>;
  if constexpr (tZero) {
    return Out{
      .lower = expand(v, index_tag<Half::size>, zero_tag),
      .upper = zeros(type_tag<Half>),
    };
  } else {
    return Out{
      .lower = expand(v, index_tag<Half::size>, zero_tag),
      .upper = undefined(type_tag<Half>),
    };
  }
}
} // namespace grex::backend

#include "grex/backend/shared/operations/expand.hpp" // IWYU pragma: export

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_EXPAND_HPP
