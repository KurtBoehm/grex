// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_MASK_CONVERT_HPP
#define INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_MASK_CONVERT_HPP

#include <cstddef>

#include "grex/backend/base.hpp"
#include "grex/backend/choosers.hpp"
#include "grex/base.hpp"

namespace grex::backend {
// Convert a mask to signed integers
template<Vectorizable T, std::size_t N>
inline SubVector<SignedInt<sizeof(T)>, N> mask2vector(SubMask<T, N> m) {
  return SubVector<SignedInt<sizeof(T)>, N>{mask2vector(m.full)};
}
template<typename Half>
inline VectorFor<SignedInt<sizeof(typename Half::VectorValue)>, 2 * Half::size>
mask2vector(SuperMask<Half> m) {
  return {.lower = mask2vector(m.lower), .upper = mask2vector(m.upper)};
}

// Convert (signed) integers to a mask
template<SignedIntVectorizable T, std::size_t N, Vectorizable Dst>
requires(sizeof(T) == sizeof(Dst))
inline SubMask<Dst, N> vector2mask(SubVector<T, N> m, TypeTag<Dst> tag) {
  return SubMask<Dst, N>{vector2mask(m.full, tag)};
}
template<typename Half, Vectorizable Dst>
requires(SignedIntVectorizable<typename Half::Value> && sizeof(typename Half::Value) == sizeof(Dst))
inline MaskFor<Dst, 2 * Half::size> vector2mask(SuperVector<Half> m, TypeTag<Dst> tag) {
  return {.lower = vector2mask(m.lower, tag), .upper = vector2mask(m.upper, tag)};
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_MASK_CONVERT_HPP
