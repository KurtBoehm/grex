// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_SHRINK_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_SHRINK_HPP

#include <cstddef>

#include "grex/backend/choosers.hpp"
#include "grex/backend/defs.hpp"
#include "grex/backend/macros/base.hpp"
#include "grex/backend/macros/for-each.hpp"
#include "grex/backend/macros/math.hpp"
#include "grex/backend/x86/instruction-sets.hpp"
#include "grex/backend/x86/macros/intrinsics.hpp"
#include "grex/backend/x86/sizes.hpp"
#include "grex/backend/x86/types.hpp"
#include "grex/base.hpp"

namespace grex::backend {
#define GREX_SHRINK_INTRINSIC(KIND, BITS, DSTSIZE, SRCSIZE) \
  inline Vector<KIND##BITS, DSTSIZE> shrink(Vector<KIND##BITS, SRCSIZE> v, \
                                            IndexTag<DSTSIZE> /*dst_size*/) { \
    return {.r = GREX_CAT(GREX_BITPREFIX(GREX_MULTIPLY(BITS, SRCSIZE)), _cast, \
                          GREX_SIR_SUFFIX(KIND, BITS, GREX_MULTIPLY(BITS, SRCSIZE)), _, \
                          GREX_SIR_SUFFIX(KIND, BITS, GREX_MULTIPLY(BITS, DSTSIZE)))(v.r)}; \
  }
#if GREX_X86_64_LEVEL >= 3
#define GREX_SHRINK256(KIND, BITS, SIZE) \
  GREX_SHRINK_INTRINSIC(KIND, BITS, SIZE, GREX_MULTIPLY(SIZE, 2))
GREX_FOREACH_TYPE(GREX_SHRINK256, 128)
#endif
#if GREX_X86_64_LEVEL >= 4
#define GREX_SHRINK512(KIND, BITS, SIZE) \
  GREX_SHRINK_INTRINSIC(KIND, BITS, SIZE, GREX_MULTIPLY(SIZE, 4)) \
  GREX_SHRINK_INTRINSIC(KIND, BITS, GREX_MULTIPLY(SIZE, 2), GREX_MULTIPLY(SIZE, 4))
GREX_FOREACH_TYPE(GREX_SHRINK512, 128)
#endif

// Shrink to same size: No-op
template<AnyVector TVec>
inline TVec shrink(TVec v, IndexTag<TVec::size> /*dst_size*/) {
  return v;
}
// Shrink native to sub-native: Shrink to smallest native and convert to sub-native
template<Vectorizable T, std::size_t tSrcSize, std::size_t tDstSize>
requires(is_subnative<T, tDstSize>)
inline VectorFor<T, tDstSize> shrink(Vector<T, tSrcSize> v, IndexTag<tDstSize> /*dst_size*/) {
  const auto min_native = shrink(v, index_tag<16 / sizeof(T)>);
  return VectorFor<T, tDstSize>{min_native};
}
// Shrink super-native: Shrink the lower half
template<AnyVector THalf, std::size_t tDstSize>
requires(tDstSize <= THalf::size)
inline VectorFor<typename THalf::Value, tDstSize> shrink(SuperVector<THalf> v,
                                                         IndexTag<tDstSize> dst_size) {
  return shrink(v.lower, dst_size);
}
// Shrink sub-native: Change the wrapper class
template<Vectorizable T, std::size_t tSrcPart, std::size_t tSrcSize, std::size_t tDstSize>
requires(tDstSize < tSrcPart)
inline VectorFor<T, tDstSize> shrink(SubVector<T, tSrcPart, tSrcSize> v,
                                     IndexTag<tDstSize> /*dst_size*/) {
  return SubVector<T, tDstSize, tSrcSize>{v.full};
}

template<std::size_t tDstSize, AnyVector TVec>
inline VectorFor<typename TVec::Value, tDstSize> shrink(TVec v) {
  return shrink(v, index_tag<tDstSize>);
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_SHRINK_HPP
