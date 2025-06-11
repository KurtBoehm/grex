// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_GATHER_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_GATHER_HPP

#include <cstddef>
#include <span>

#include "grex/backend/choosers.hpp"
#include "grex/backend/defs.hpp"
#include "grex/backend/x86/operations/extract.hpp"
#include "grex/backend/x86/operations/merge.hpp"
#include "grex/backend/x86/operations/set.hpp"
#include "grex/base/defs.hpp"

namespace grex::backend {
template<Vectorizable TValue, Vectorizable TIndex, std::size_t tSize>
inline VectorFor<TValue, tSize> gather(std::span<const TValue> data, Vector<TIndex, tSize> idxs) {
  return static_apply<tSize>([&]<std::size_t... tIdxs> {
    return set(type_tag<VectorFor<TValue, tSize>>, data[std::size_t(extract(idxs, tIdxs))]...);
  });
}
template<Vectorizable TValue, Vectorizable TIndex, std::size_t tPart, std::size_t tSize>
inline VectorFor<TValue, tPart> gather(std::span<const TValue> data,
                                       SubVector<TIndex, tPart, tSize> idxs) {
  return static_apply<tPart>([&]<std::size_t... tIdxs> {
    return set(type_tag<VectorFor<TValue, tPart>>, data[std::size_t(extract(idxs, tIdxs))]...);
  });
}
template<Vectorizable TValue, typename THalf>
inline VectorFor<TValue, 2 * THalf::size> gather(std::span<const TValue> data,
                                                 SuperVector<THalf> idxs) {
  return merge(gather(data, idxs.lower), gather(data, idxs.upper));
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_GATHER_HPP
