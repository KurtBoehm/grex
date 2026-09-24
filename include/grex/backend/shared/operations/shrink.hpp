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
template<AnyVector Vec>
inline Vec shrink(Vec v, IndexTag<Vec::size> /*dst_size*/) {
  return v;
}
// Shrink native to sub-native: Shrink to smallest native and convert to sub-native
template<Vectorizable T, std::size_t SrcN, std::size_t DstN>
requires(is_subnative<T, DstN>)
inline VectorFor<T, DstN> shrink(NativeVector<T, SrcN> v, IndexTag<DstN> /*dst_size*/) {
  const auto min_native = shrink(v, index_tag<16 / sizeof(T)>);
  return VectorFor<T, DstN>{min_native};
}
// Shrink super-native: Shrink the lower half
template<AnyVector Half, std::size_t DstN>
requires(DstN <= Half::size)
inline VectorFor<typename Half::Value, DstN> shrink(SuperVector<Half> v, IndexTag<DstN> dst_size) {
  return shrink(v.lower, dst_size);
}
// Shrink sub-native: Change the wrapper class
template<Vectorizable T, std::size_t SrcN, std::size_t DstN>
requires(DstN < SrcN)
inline VectorFor<T, DstN> shrink(SubVector<T, SrcN> v, IndexTag<DstN> /*dst_size*/) {
  return SubVector<T, DstN>{v.full};
}

template<std::size_t DstN, AnyVector Vec>
inline VectorFor<typename Vec::Value, DstN> shrink(Vec v) {
  return shrink(v, index_tag<DstN>);
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_SHRINK_HPP
