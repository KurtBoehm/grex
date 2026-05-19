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
/////////////////////////
// Trivial no-op cases //
/////////////////////////

// Source and destination scalar types are identical: return the vector unchanged.
template<AnyVector TVec>
inline TVec convert(TVec v, TypeTag<ValueOf<TVec>> /*tag*/) {
  return v;
}

// Integer vectors with the same element width but different signedness:
// reinterpret the register without modifying the bits.
template<IntVectorizable TDst, IntVector TSrc>
requires(!std::same_as<ValueOf<TSrc>, TDst> && sizeof(TDst) == sizeof(ValueOf<TSrc>))
inline NativeVector<TDst, size_of<TSrc>> convert(TSrc v, TypeTag<TDst> /*tag*/) {
  return as<TDst>(v);
}

///////////////////
// Generic cases //
///////////////////

// Sub-native vector → sub-native vector:
// expand to the smallest size where the source or destination element type becomes native,
// perform the conversion there, then wrap back into a sub-vector.
template<Vectorizable TDst, Vectorizable TSrc, std::size_t tPart, std::size_t tSize>
requires(is_subnative<TDst, tPart>)
inline VectorFor<TDst, tPart> convert(SubVector<TSrc, tPart, tSize> v, TypeTag<TDst> /*tag*/) {
  using Out = VectorFor<TDst, tPart>;
  constexpr std::size_t work_size = std::min(tSize, Out::Full::size);
  static_assert(work_size > tPart);
  const auto s = convert(VectorFor<TSrc, work_size>{v.registr()}, type_tag<TDst>);
  return Out{s.registr()};
}

// Super-native vector → super-native vector:
// convert both halves to the destination type independently and merge.
template<typename THalf, Vectorizable TDst>
requires(is_supernative<TDst, THalf::size * 2>)
inline VectorFor<TDst, THalf::size * 2> convert(SuperVector<THalf> v, TypeTag<TDst> /*tag*/) {
  return merge(convert(v.lower, type_tag<TDst>), convert(v.upper, type_tag<TDst>));
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_CONVERT_HPP
