// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_GATHER_HPP
#define INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_GATHER_HPP

#include <cstddef>
#include <span>

#include "grex/backend/active/operations/extract.hpp"
#include "grex/backend/active/operations/merge.hpp"
#include "grex/backend/active/operations/set.hpp"
#include "grex/backend/active/operations/split.hpp"
#include "grex/backend/base.hpp"
#include "grex/backend/choosers.hpp"
#include "grex/base.hpp"

namespace grex::backend {
template<Vectorizable V, std::size_t Extent, Vectorizable Index, std::size_t N>
inline VectorFor<V, N> gather(std::span<const V, Extent> data, NativeVector<Index, N> idxs) {
  return static_apply<N>([&]<std::size_t... I> {
    return set(type_tag<VectorFor<V, N>>, data[std::size_t(extract(idxs, index_tag<I>))]...);
  });
}
template<Vectorizable V, std::size_t Extent, Vectorizable Index, std::size_t N>
inline VectorFor<V, N> gather(std::span<const V, Extent> data, SubVector<Index, N> idxs) {
  return static_apply<N>([&]<std::size_t... I> {
    return set(type_tag<VectorFor<V, N>>, data[std::size_t(extract(idxs, index_tag<I>))]...);
  });
}
template<Vectorizable V, std::size_t Extent, typename Half>
inline VectorFor<V, 2 * Half::size> gather(std::span<const V, Extent> data,
                                           SuperVector<Half> idxs) {
  return merge(gather(data, idxs.lower), gather(data, idxs.upper));
}

template<Vectorizable V, std::size_t Extent, Vectorizable Index, std::size_t N>
inline VectorFor<V, N> mask_gather(std::span<const V, Extent> data, MaskFor<V, N> m,
                                   NativeVector<Index, N> idxs) {
  return static_apply<N>([&]<std::size_t... I> {
    return set(
      type_tag<VectorFor<V, N>>,
      (extract(m, index_tag<I>) ? data[std::size_t(extract(idxs, index_tag<I>))] : V{})...);
  });
}
template<Vectorizable V, std::size_t Extent, Vectorizable Index, std::size_t N>
inline VectorFor<V, N> mask_gather(std::span<const V, Extent> data, MaskFor<V, N> m,
                                   SubVector<Index, N> idxs) {
  return static_apply<N>([&]<std::size_t... I> {
    return set(
      type_tag<VectorFor<V, N>>,
      (extract(m, index_tag<I>) ? data[std::size_t(extract(idxs, index_tag<I>))] : V{})...);
  });
}
template<Vectorizable V, std::size_t Extent, typename VecHalf>
inline VectorFor<V, 2 * VecHalf::size> mask_gather(std::span<const V, Extent> data,
                                                   MaskFor<V, 2 * VecHalf::size> m,
                                                   SuperVector<VecHalf> idxs) {
  return merge(mask_gather(data, get_low(m), idxs.lower),
               mask_gather(data, get_high(m), idxs.upper));
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_GATHER_HPP
