// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_BLEND_HPP
#define INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_BLEND_HPP

#include <cstddef>

#include "grex/backend/defs.hpp"

namespace grex::backend {
template<Vectorizable T, std::size_t tPart, std::size_t tSize>
inline SubVector<T, tPart, tSize> blend_zero(SubMask<T, tPart, tSize> m,
                                             SubVector<T, tPart, tSize> v1) {
  return SubVector<T, tPart, tSize>{blend_zero(m.full, v1.full)};
}
template<Vectorizable T, std::size_t tPart, std::size_t tSize>
inline SubVector<T, tPart, tSize> blend(SubMask<T, tPart, tSize> m, SubVector<T, tPart, tSize> v0,
                                        SubVector<T, tPart, tSize> v1) {
  return SubVector<T, tPart, tSize>{blend(m.full, v0.full, v1.full)};
}

template<typename TVecHalf, typename TMaskHalf>
inline SuperVector<TVecHalf> blend_zero(SuperMask<TMaskHalf> m, SuperVector<TVecHalf> v1) {
  return {.lower = blend_zero(m.lower, v1.lower), .upper = blend_zero(m.upper, v1.upper)};
}
template<typename TVecHalf, typename TMaskHalf>
inline SuperVector<TVecHalf> blend(SuperMask<TMaskHalf> m, SuperVector<TVecHalf> v0,
                                   SuperVector<TVecHalf> v1) {
  return {.lower = blend(m.lower, v0.lower, v1.lower), .upper = blend(m.upper, v0.upper, v1.upper)};
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_BLEND_HPP
