// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_MASK_CONVERT_HPP
#define INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_MASK_CONVERT_HPP

#include <cstddef>

#include "grex/backend/choosers.hpp"
#include "grex/backend/defs.hpp"
#include "grex/base/defs.hpp"

namespace grex::backend {
// Convert a mask to signed integers
template<Vectorizable T, std::size_t tPart, std::size_t tSize>
inline SubVector<SignedInt<sizeof(T)>, tPart, tSize> mask2vector(SubMask<T, tPart, tSize> m) {
  return SubVector<SignedInt<sizeof(T)>, tPart, tSize>{mask2vector(m.full)};
}
template<typename THalf>
inline VectorFor<SignedInt<sizeof(typename THalf::VectorValue)>, 2 * THalf::size>
mask2vector(SuperMask<THalf> m) {
  return {.lower = mask2vector(m.lower), .upper = mask2vector(m.upper)};
}

// Convert (signed) integers to a mask
template<SignedIntVectorizable T, std::size_t tPart, std::size_t tSize, Vectorizable TDst>
requires(sizeof(T) == sizeof(TDst))
inline SubMask<TDst, tPart, tSize> vector2mask(SubVector<T, tPart, tSize> m,
                                               TypeTag<TDst> tag) {
  return SubMask<TDst, tPart, tSize>{vector2mask(m.full, tag)};
}
template<typename THalf, Vectorizable TDst>
requires(SignedIntVectorizable<typename THalf::Value> &&
         sizeof(typename THalf::Value) == sizeof(TDst))
inline MaskFor<TDst, 2 * THalf::size> vector2mask(SuperVector<THalf> m, TypeTag<TDst> tag) {
  return {.lower = vector2mask(m.lower, tag), .upper = vector2mask(m.upper, tag)};
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_MASK_CONVERT_HPP
