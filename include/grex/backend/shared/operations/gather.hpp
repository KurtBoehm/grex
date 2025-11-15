// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_GATHER_HPP
#define INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_GATHER_HPP

#include <cstddef>
#include <span>

#include "grex/backend/choosers.hpp"
#include "grex/backend/defs.hpp"
#include "grex/base/defs.hpp"

namespace grex::backend {
template<Vectorizable TValue, std::size_t tExtent, Vectorizable TIndex, std::size_t tSize>
inline VectorFor<TValue, tSize> gather(std::span<const TValue, tExtent> data,
                                       Vector<TIndex, tSize> idxs) {
  return static_apply<tSize>([&]<std::size_t... tIdxs> {
    return set(type_tag<VectorFor<TValue, tSize>>,
               data[std::size_t(extract(idxs, index_tag<tIdxs>))]...);
  });
}
template<Vectorizable TValue, std::size_t tExtent, Vectorizable TIndex, std::size_t tPart,
         std::size_t tSize>
inline VectorFor<TValue, tPart> gather(std::span<const TValue, tExtent> data,
                                       SubVector<TIndex, tPart, tSize> idxs) {
  return static_apply<tPart>([&]<std::size_t... tIdxs> {
    return set(type_tag<VectorFor<TValue, tPart>>,
               data[std::size_t(extract(idxs, index_tag<tIdxs>))]...);
  });
}
template<Vectorizable TValue, std::size_t tExtent, typename THalf>
inline VectorFor<TValue, 2 * THalf::size> gather(std::span<const TValue, tExtent> data,
                                                 SuperVector<THalf> idxs) {
  return merge(gather(data, idxs.lower), gather(data, idxs.upper));
}

template<Vectorizable TValue, std::size_t tExtent, Vectorizable TIndex, std::size_t tSize>
inline VectorFor<TValue, tSize> mask_gather(std::span<const TValue, tExtent> data,
                                            MaskFor<TValue, tSize> m, Vector<TIndex, tSize> idxs) {
  return static_apply<tSize>([&]<std::size_t... tIdxs> {
    return set(type_tag<VectorFor<TValue, tSize>>,
               (extract(m, index_tag<tIdxs>) ? data[std::size_t(extract(idxs, index_tag<tIdxs>))]
                                             : TValue{})...);
  });
}
template<Vectorizable TValue, std::size_t tExtent, Vectorizable TIndex, std::size_t tPart,
         std::size_t tSize>
inline VectorFor<TValue, tPart> mask_gather(std::span<const TValue, tExtent> data,
                                            MaskFor<TValue, tPart> m,
                                            SubVector<TIndex, tPart, tSize> idxs) {
  return static_apply<tPart>([&]<std::size_t... tIdxs> {
    return set(type_tag<VectorFor<TValue, tPart>>,
               (extract(m, index_tag<tIdxs>) ? data[std::size_t(extract(idxs, index_tag<tIdxs>))]
                                             : TValue{})...);
  });
}
template<Vectorizable TValue, std::size_t tExtent, typename TVecHalf>
inline VectorFor<TValue, 2 * TVecHalf::size> mask_gather(std::span<const TValue, tExtent> data,
                                                         MaskFor<TValue, 2 * TVecHalf::size> m,
                                                         SuperVector<TVecHalf> idxs) {
  return merge(mask_gather(data, get_low(m), idxs.lower),
               mask_gather(data, get_high(m), idxs.upper));
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_GATHER_HPP
