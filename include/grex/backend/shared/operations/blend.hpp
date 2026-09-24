// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_BLEND_HPP
#define INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_BLEND_HPP

#include <cstddef>

#include "grex/backend/base.hpp"
#include "grex/base.hpp"

namespace grex::backend {
template<Vectorizable T, std::size_t N>
inline SubVector<T, N> blend_zero(SubMask<T, N> m, SubVector<T, N> v1) {
  return SubVector<T, N>{blend_zero(m.full, v1.full)};
}
template<Vectorizable T, std::size_t N>
inline SubVector<T, N> blend(SubMask<T, N> m, SubVector<T, N> v0, SubVector<T, N> v1) {
  return SubVector<T, N>{blend(m.full, v0.full, v1.full)};
}

template<typename VecHalf, typename MaskHalf>
inline SuperVector<VecHalf> blend_zero(SuperMask<MaskHalf> m, SuperVector<VecHalf> v1) {
  return {.lower = blend_zero(m.lower, v1.lower), .upper = blend_zero(m.upper, v1.upper)};
}
template<typename VecHalf, typename MaskHalf>
inline SuperVector<VecHalf> blend(SuperMask<MaskHalf> m, SuperVector<VecHalf> v0,
                                  SuperVector<VecHalf> v1) {
  return {.lower = blend(m.lower, v0.lower, v1.lower), .upper = blend(m.upper, v0.upper, v1.upper)};
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_BLEND_HPP
