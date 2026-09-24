// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_CONVERT_HPP
#define INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_CONVERT_HPP

#include <concepts>
#include <cstddef>

#include "grex/backend/active/operations/merge.hpp"
#include "grex/backend/active/operations/reinterpret.hpp"
#include "grex/backend/base.hpp"
#include "grex/backend/choosers.hpp"
#include "grex/base.hpp"

namespace grex::backend {
//==================================================================================================
// Trivial no-op cases
//==================================================================================================

// Source and destination scalar types are identical: return the vector unchanged.
template<AnyVector Vec>
inline Vec convert(Vec v, TypeTag<ValueOf<Vec>> /*tag*/) {
  return v;
}

// Integer vectors with the same element width but different signedness:
// reinterpret the register without modifying the bits.
template<IntVectorizable Dst, IntVector Src>
requires(!std::same_as<ValueOf<Src>, Dst> && sizeof(Dst) == sizeof(ValueOf<Src>))
inline NativeVector<Dst, size_of<Src>> convert(Src v, TypeTag<Dst> /*tag*/) {
  return as<Dst>(v);
}

//==================================================================================================
// Generic cases
//==================================================================================================

// Sub-native vector → sub-native vector:
// expand to the smallest size where the source or destination element type becomes native,
// perform the conversion there, then wrap back into a sub-vector.
template<Vectorizable Dst, Vectorizable Src, std::size_t N>
requires(is_subnative<Dst, N>)
inline VectorFor<Dst, N> convert(SubVector<Src, N> v, TypeTag<Dst> /*tag*/) {
  using Out = VectorFor<Dst, N>;
  constexpr std::size_t work_size = std::min(min_native_size<Src>, Out::Full::size);
  static_assert(work_size > N);
  const auto s = convert(VectorFor<Src, work_size>{v.registr()}, type_tag<Dst>);
  return Out{s.registr()};
}

// Super-native vector → super-native vector:
// convert both halves to the destination type independently and merge.
template<typename Half, Vectorizable Dst>
requires(is_supernative<Dst, Half::size * 2>)
inline VectorFor<Dst, Half::size * 2> convert(SuperVector<Half> v, TypeTag<Dst> /*tag*/) {
  return merge(convert(v.lower, type_tag<Dst>), convert(v.upper, type_tag<Dst>));
}

// Convenience functions taking the destination element type as a template parameter, not a tag.
template<Vectorizable Dst, AnyVector Src>
inline VectorFor<Dst, Src::size> convert(Src v) {
  return convert(v, type_tag<Dst>);
}
template<Vectorizable Dst, AnyMask Src>
inline MaskFor<Dst, Src::size> convert(Src v) {
  return convert(v, type_tag<Dst>);
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_CONVERT_HPP
