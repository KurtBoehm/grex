// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_EXPAND_VECTOR_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_EXPAND_VECTOR_HPP

#include <cstddef>

#include <immintrin.h>

// #define GREX_X86_64_LEVEL 4
#include "grex/backend/choosers.hpp"
#include "grex/backend/defs.hpp"
#include "grex/backend/x86/helpers.hpp"
#include "grex/backend/x86/instruction-sets.hpp"
#include "grex/backend/x86/operations/set.hpp"
#include "grex/backend/x86/operations/subnative.hpp"
#include "grex/backend/x86/types.hpp" // IWYU pragma: keep
#include "grex/base/defs.hpp"

namespace grex::backend {
// unchanged size: no-op
template<AnyVector TVec, bool tZero>
inline TVec expand(TVec v, IndexTag<TVec::size> /*size*/, BoolTag<tZero> /*zero*/) {
  return v;
}

// expand with arbitrary values in the expanded area
// native → native
#define GREX_EXPANDV_ANY_INTRINSIC(KIND, BITS, DSTSIZE, SRCRBITS, DSTRBITS) \
  inline Vector<KIND##BITS, DSTSIZE> expand(Vector<KIND##BITS, GREX_ELEMENTS(SRCRBITS, BITS)> v, \
                                            IndexTag<DSTSIZE>, BoolTag<false>) { \
    return {.r = GREX_CAT(GREX_BITPREFIX(DSTRBITS), _cast, GREX_SIR_SUFFIX(KIND, BITS, SRCRBITS), \
                          _, GREX_SIR_SUFFIX(KIND, BITS, DSTRBITS))(v.r)}; \
  }
#if GREX_X86_64_LEVEL >= 3
GREX_FOREACH_TYPE(GREX_EXPANDV_ANY_INTRINSIC, 256, 128, 256)
#endif
#if GREX_X86_64_LEVEL >= 4
GREX_FOREACH_TYPE(GREX_EXPANDV_ANY_INTRINSIC, 512, 128, 512)
GREX_FOREACH_TYPE(GREX_EXPANDV_ANY_INTRINSIC, 512, 256, 512)
#endif
// native/super-native → super-native
template<AnyVector TVec, std::size_t tDstSize>
requires(tDstSize > TVec::size && is_supernative<typename TVec::Value, tDstSize> &&
         (AnyNativeVector<TVec> || AnySuperNativeVector<TVec>))
inline VectorFor<typename TVec::Value, tDstSize> expand(TVec v, IndexTag<tDstSize> /*size*/,
                                                        BoolTag<false> zero_tag) {
  using Value = TVec::Value;
  constexpr std::size_t tsize = tDstSize / 2;
  return merge(expand(v, index_tag<tsize>, zero_tag), undefined(type_tag<VectorFor<Value, tsize>>));
}

// expand with zeros in the expanded area
// native/super-native → native/super-native
template<AnyVector TVec, std::size_t tDstSize>
requires(tDstSize > TVec::size && (AnyNativeVector<TVec> || AnySuperNativeVector<TVec>))
inline VectorFor<typename TVec::Value, tDstSize> expand(TVec v, IndexTag<tDstSize> /*size*/,
                                                        BoolTag<true> zero_tag) {
  using Value = TVec::Value;
  constexpr std::size_t tsize = tDstSize / 2;
  return merge(expand(v, index_tag<tsize>, zero_tag), zeros(type_tag<VectorFor<Value, tsize>>));
}

// both kinds of expansion
// sub-native → sub-native/native
template<typename T, std::size_t tPart, std::size_t tSize, std::size_t tDstSize, bool tZero>
inline VectorFor<T, tDstSize> expand(SubVector<T, tPart, tSize> v, IndexTag<tDstSize> size_tag,
                                     BoolTag<tZero> zero_tag) {
  using Work = VectorFor<T, std::min(tDstSize, native_size<T, 0>)>;
  Work work = [&] {
    if constexpr (tZero) {
      return Work{full_cutoff(v).r};
    } else {
      return Work{v.registr()};
    }
  }();
  if constexpr (tDstSize <= native_size<T, 0>) {
    return work;
  } else {
    return expand(work, size_tag, zero_tag);
  }
}

template<AnyVector TVec, std::size_t tSize>
inline VectorFor<typename TVec::Value, tSize> expand_any(TVec v, IndexTag<tSize> size) {
  return expand(v, size, bool_tag<false>);
}
template<AnyVector TVec, std::size_t tSize>
inline VectorFor<typename TVec::Value, tSize> expand_zero(TVec v, IndexTag<tSize> size) {
  return expand(v, size, bool_tag<true>);
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_EXPAND_VECTOR_HPP
