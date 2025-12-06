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
#include "grex/backend/defs.hpp"
#include "grex/backend/macros/for-each.hpp"
#include "grex/backend/macros/math.hpp"
#include "grex/backend/neon/macros/types.hpp"
#include "grex/backend/neon/operations/expand-register.hpp"
#include "grex/backend/neon/operations/insert-static.hpp"
#include "grex/backend/neon/operations/set.hpp"
#include "grex/backend/neon/operations/subnative.hpp"
#include "grex/backend/neon/sizes.hpp"
#include "grex/base.hpp"

namespace grex::backend {
#define GREX_EXPAND_ANY(KIND, BITS, SIZE) \
  inline Vector<KIND##BITS, SIZE> expand(Scalar<KIND##BITS> x, IndexTag<SIZE> /*tag*/, \
                                         BoolTag<false> /*tag*/) { \
    return {.r = expand_register(x)}; \
  }
GREX_FOREACH_TYPE(GREX_EXPAND_ANY, 128)

template<Vectorizable T, std::size_t tSize>
inline VectorFor<T, tSize> expand(Scalar<T> x, IndexTag<tSize> /*tag*/, BoolTag<true> /*tag*/) {
  return insert(zeros(type_tag<VectorFor<T, tSize>>), index_tag<0>, x.value);
}

// Sub-native: Delegate to the native version
template<Vectorizable T, std::size_t tSize, bool tZero>
requires(tSize < min_native_size<T>)
inline VectorFor<T, tSize> expand(Scalar<T> x, IndexTag<tSize> /*tag*/, BoolTag<tZero> zero) {
  return VectorFor<T, tSize>{expand(x, index_tag<min_native_size<T>>, zero)};
}

// Larger than the smallest native size: Merge with zero/undefined
template<Vectorizable T, std::size_t tSize, bool tZero>
requires(tSize > min_native_size<T>)
inline VectorFor<T, tSize> expand(Scalar<T> x, IndexTag<tSize> /*tag*/, BoolTag<tZero> zero) {
  constexpr std::size_t half = tSize / 2;
  return expand(expand(x, index_tag<half>, zero), index_tag<tSize>, zero);
}

template<Vectorizable T, std::size_t tSize>
inline VectorFor<T, tSize> expand_any(Scalar<T> x, IndexTag<tSize> size) {
  return expand(x, size, false_tag);
}
template<Vectorizable T, std::size_t tSize>
inline VectorFor<T, tSize> expand_zero(Scalar<T> x, IndexTag<tSize> size) {
  return expand(x, size, true_tag);
}

// unchanged size: no-op
template<AnyVector TVec, bool tZero>
inline TVec expand(TVec v, IndexTag<TVec::size> /*size*/, BoolTag<tZero> /*zero*/) {
  return v;
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

// sub-native → sub-native/native
template<typename T, std::size_t tPart, std::size_t tSize, std::size_t tDstSize, bool tZero>
inline VectorFor<T, tDstSize> expand(SubVector<T, tPart, tSize> v, IndexTag<tDstSize> size_tag,
                                     BoolTag<tZero> zero_tag) {
  using Work = VectorFor<T, std::min(tDstSize, min_native_size<T>)>;
  Work work = [&] {
    if constexpr (tZero) {
      return Work{full_cutoff(v).r};
    } else {
      return Work{v.registr()};
    }
  }();
  if constexpr (tDstSize <= min_native_size<T>) {
    return work;
  } else {
    return expand(work, size_tag, zero_tag);
  }
}

template<AnyVector TVec, std::size_t tSize>
inline VectorFor<typename TVec::Value, tSize> expand_any(TVec v, IndexTag<tSize> size) {
  return expand(v, size, false_tag);
}
template<std::size_t tSize, AnyVector TVec>
inline VectorFor<typename TVec::Value, tSize> expand_any(TVec v) {
  return expand(v, index_tag<tSize>, false_tag);
}

template<AnyVector TVec, std::size_t tSize>
inline VectorFor<typename TVec::Value, tSize> expand_zero(TVec v, IndexTag<tSize> size) {
  return expand(v, size, true_tag);
}
template<std::size_t tSize, AnyVector TVec>
inline VectorFor<typename TVec::Value, tSize> expand_zero(TVec v) {
  return expand(v, index_tag<tSize>, true_tag);
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
