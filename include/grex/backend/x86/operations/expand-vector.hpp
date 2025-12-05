// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_EXPAND_VECTOR_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_EXPAND_VECTOR_HPP

#include <cstddef>

#include <immintrin.h>

#include "grex/backend/choosers.hpp"
#include "grex/backend/defs.hpp"
#include "grex/backend/macros/base.hpp"
#include "grex/backend/macros/math.hpp"
#include "grex/backend/x86/instruction-sets.hpp"
#include "grex/backend/x86/macros/intrinsics.hpp"
#include "grex/backend/x86/operations/merge.hpp"
#include "grex/backend/x86/operations/set.hpp"
#include "grex/backend/x86/operations/subnative.hpp"
#include "grex/base.hpp"

#if GREX_X86_64_LEVEL >= 3
#include "grex/backend/macros/for-each.hpp"
#include "grex/backend/x86/types.hpp"
#endif

namespace grex::backend {
// unchanged size: no-op
template<AnyVector TVec, bool tZero>
inline TVec expand(TVec v, IndexTag<TVec::size> /*size*/, BoolTag<tZero> /*zero*/) {
  return v;
}

// native → native: use cast/zext intrinsics
#define GREX_EXPANDV_INTRINSIC(KIND, BITS, DSTSIZE, SRCRBITS, DSTRBITS) \
  inline Vector<KIND##BITS, DSTSIZE> expand(Vector<KIND##BITS, GREX_DIVIDE(SRCRBITS, BITS)> v, \
                                            IndexTag<DSTSIZE>, BoolTag<false>) { \
    return {.r = GREX_CAT(GREX_BITPREFIX(DSTRBITS), _cast, GREX_SIR_SUFFIX(KIND, BITS, SRCRBITS), \
                          _, GREX_SIR_SUFFIX(KIND, BITS, DSTRBITS))(v.r)}; \
  } \
  inline Vector<KIND##BITS, DSTSIZE> expand(Vector<KIND##BITS, GREX_DIVIDE(SRCRBITS, BITS)> v, \
                                            IndexTag<DSTSIZE>, BoolTag<true>) { \
    return {.r = GREX_CAT(GREX_BITPREFIX(DSTRBITS), _zext, GREX_SIR_SUFFIX(KIND, BITS, SRCRBITS), \
                          _, GREX_SIR_SUFFIX(KIND, BITS, DSTRBITS))(v.r)}; \
  }
#if GREX_X86_64_LEVEL >= 3
GREX_FOREACH_TYPE(GREX_EXPANDV_INTRINSIC, 256, 128, 256)
#endif
#if GREX_X86_64_LEVEL >= 4
GREX_FOREACH_TYPE(GREX_EXPANDV_INTRINSIC, 512, 128, 512)
GREX_FOREACH_TYPE(GREX_EXPANDV_INTRINSIC, 512, 256, 512)
#endif

// native/super-native → super-native
template<AnyVector TVec, std::size_t tDstSize, bool tZero>
requires(tDstSize > TVec::size && is_supernative<typename TVec::Value, tDstSize> &&
         (AnyNativeVector<TVec> || AnySuperNativeVector<TVec>))
inline VectorFor<typename TVec::Value, tDstSize> expand(TVec v, IndexTag<tDstSize> /*size*/,
                                                        BoolTag<tZero> zero_tag) {
  using Value = TVec::Value;
  using Half = VectorFor<Value, tDstSize / 2>;
  if constexpr (tZero) {
    return merge(expand(v, index_tag<Half::size>, zero_tag), zeros(type_tag<Half>));
  } else {
    return merge(expand(v, index_tag<Half::size>, zero_tag), undefined(type_tag<Half>));
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
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_EXPAND_VECTOR_HPP
