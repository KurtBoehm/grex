// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_BROADCAST_LANE_HPP
#define INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_BROADCAST_LANE_HPP

#include <cstddef>

#include "grex/backend/base.hpp"
#include "grex/base.hpp"

namespace grex::backend {
template<Vectorizable T, std::size_t N, std::size_t I>
requires(I < N)
[[nodiscard, GREX_ALWAYS_INLINE]] inline SubVector<T, N> broadcast_lane(SubVector<T, N> v,
                                                                        IndexTag<I> i) {
  return SubVector<T, N>{broadcast_lane(v.full, i)};
}
template<AnyVector Half, std::size_t I>
requires(I < 2 * size_of<Half>)
[[nodiscard, GREX_ALWAYS_INLINE]] inline SuperVector<Half> broadcast_lane(SuperVector<Half> v,
                                                                          IndexTag<I> i) {
  if constexpr (I < size_of<Half>) {
    const auto bc = broadcast_lane(v.lower, i);
    return {.lower = bc, .upper = bc};
  } else {
    const auto bc = broadcast_lane(v.upper, index_tag<I - size_of<Half>>);
    return {.lower = bc, .upper = bc};
  }
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_BROADCAST_LANE_HPP
