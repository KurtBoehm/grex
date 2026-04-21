// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_NEON_OPERATIONS_EXPAND_HPP
#define INCLUDE_GREX_BACKEND_NEON_OPERATIONS_EXPAND_HPP

#include <cstddef>

#include <arm_neon.h>

#include "grex/backend/choosers.hpp"
#include "grex/backend/defs.hpp" // IWYU pragma: keep
#include "grex/backend/macros/for-each.hpp"
#include "grex/backend/macros/math.hpp"
#include "grex/backend/neon/macros/types.hpp"
#include "grex/backend/neon/operations/expand-register.hpp"
#include "grex/backend/neon/operations/insert-static.hpp"
#include "grex/backend/neon/operations/subnative.hpp"
#include "grex/backend/neon/operations/undefined.hpp"
#include "grex/backend/neon/sizes.hpp"
#include "grex/backend/shared/operations/expand.hpp" // IWYU pragma: export
#include "grex/base.hpp"

namespace grex::backend {
#define GREX_EXPAND_ANY(KIND, BITS, SIZE) \
  inline NativeVector<KIND##BITS, SIZE> expand(Scalar<KIND##BITS> x, IndexTag<SIZE> /*tag*/, \
                                               BoolTag<false> /*tag*/) { \
    return {.r = expand_register(x)}; \
  }
GREX_FOREACH_TYPE(GREX_EXPAND_ANY, 128)

template<Vectorizable T, std::size_t tSize>
inline VectorFor<T, tSize> expand(Scalar<T> x, IndexTag<tSize> /*tag*/, BoolTag<true> /*tag*/) {
  return insert(zeros(type_tag<VectorFor<T, tSize>>), index_tag<0>, x.value);
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

#if GREX_GCC
#define GREX_EXPAND_64(KIND, BITS, SIZE) \
  inline GREX_REGISTER(KIND, BITS, GREX_MULTIPLY(SIZE, 2)) \
    expand64(GREX_REGISTER(KIND, BITS, SIZE) v) { \
    GREX_REGISTER(KIND, BITS, GREX_MULTIPLY(SIZE, 2)) retval; \
    asm("" : "=w"(retval) : "0"(v)); /*NOLINT*/ \
    return retval; \
  }
#elif GREX_CLANG
#define GREX_EXPAND_64(KIND, BITS, SIZE) \
  inline GREX_REGISTER(KIND, BITS, GREX_MULTIPLY(SIZE, 2)) \
    expand64(GREX_REGISTER(KIND, BITS, SIZE) v) { \
    const auto undef = make_undefined<GREX_REGISTER(KIND, BITS, SIZE)>(); \
    return GREX_ISUFFIXED(vcombine, KIND, BITS)(v, undef); \
  }
#endif
GREX_EXPAND_64(f, 64, 1)
GREX_EXPAND_64(i, 64, 1)
GREX_EXPAND_64(u, 64, 1)
GREX_EXPAND_64(f, 32, 2)
GREX_EXPAND_64(i, 32, 2)
GREX_EXPAND_64(u, 32, 2)
GREX_EXPAND_64(i, 16, 4)
GREX_EXPAND_64(u, 16, 4)
GREX_EXPAND_64(i, 8, 8)
GREX_EXPAND_64(u, 8, 8)
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_EXPAND_HPP
