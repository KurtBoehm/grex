// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_SHRINK_HPP
#define INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_SHRINK_HPP

#include <cstddef>

#include "grex/backend/active/sizes.hpp"
#include "grex/backend/base.hpp"
#include "grex/backend/choosers.hpp"
#include "grex/base.hpp"

namespace grex::backend {
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

#endif // INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_SHRINK_HPP
