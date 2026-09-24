// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_EXPAND_HPP
#define INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_EXPAND_HPP

#include <cstddef>

#include "grex/backend/active/sizes.hpp"
#include "grex/backend/base.hpp"
#include "grex/backend/choosers.hpp"
#include "grex/base.hpp"

namespace grex::backend {
//==================================================================================================
// Scalar
//==================================================================================================

// Sub-native: Delegate to the native version
template<Vectorizable T, std::size_t N, bool Zero>
requires(N < min_native_size<T>)
inline VectorFor<T, N> expand(T x, IndexTag<N> /*tag*/, BoolTag<Zero> zero) {
  return VectorFor<T, N>{expand(x, index_tag<min_native_size<T>>, zero)};
}

// Larger than the smallest native size: Merge with zero/undefined
template<Vectorizable T, std::size_t N, bool Zero>
requires(N > min_native_size<T>)
inline VectorFor<T, N> expand(T x, IndexTag<N> /*tag*/, BoolTag<Zero> zero) {
  constexpr std::size_t half = N / 2;
  return expand(expand(x, index_tag<half>, zero), index_tag<N>, zero);
}

template<Vectorizable T, std::size_t N>
inline VectorFor<T, N> expand_any(T x, IndexTag<N> size) {
  return expand(x, size, false_tag);
}
template<Vectorizable T, std::size_t N>
inline VectorFor<T, N> expand_zero(T x, IndexTag<N> size) {
  return expand(x, size, true_tag);
}

//==================================================================================================
// Vector
//==================================================================================================

// unchanged size: no-op
template<AnyVector Vec, bool Zero>
inline Vec expand(Vec v, IndexTag<Vec::size> /*size*/, BoolTag<Zero> /*zero*/) {
  return v;
}

// sub-native → sub-native/native
template<typename T, std::size_t N, std::size_t DstN, bool Zero>
inline VectorFor<T, DstN> expand(SubVector<T, N> v, IndexTag<DstN> size_tag,
                                 BoolTag<Zero> zero_tag) {
  using Work = VectorFor<T, std::min(DstN, min_native_size<T>)>;
  const Work work = [&] {
    if constexpr (Zero) {
      return Work{full_cutoff(v).r};
    } else {
      return Work{v.registr()};
    }
  }();
  if constexpr (DstN <= min_native_size<T>) {
    return work;
  } else {
    return expand(work, size_tag, zero_tag);
  }
}

template<AnyVector Vec, std::size_t N>
inline VectorFor<typename Vec::Value, N> expand_any(Vec v, IndexTag<N> size) {
  return expand(v, size, false_tag);
}
template<std::size_t N, AnyVector Vec>
inline VectorFor<typename Vec::Value, N> expand_any(Vec v) {
  return expand(v, index_tag<N>, false_tag);
}

template<AnyVector Vec, std::size_t N>
inline VectorFor<typename Vec::Value, N> expand_zero(Vec v, IndexTag<N> size) {
  return expand(v, size, true_tag);
}
template<std::size_t N, AnyVector Vec>
inline VectorFor<typename Vec::Value, N> expand_zero(Vec v) {
  return expand(v, index_tag<N>, true_tag);
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_EXPAND_HPP
