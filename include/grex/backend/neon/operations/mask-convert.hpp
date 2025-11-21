// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_NEON_OPERATIONS_MASK_CONVERT_HPP
#define INCLUDE_GREX_BACKEND_NEON_OPERATIONS_MASK_CONVERT_HPP

#include <cstddef>

#include "grex/backend/defs.hpp"
#include "grex/base.hpp"

namespace grex::backend {
// Convert a mask to signed integers
template<Vectorizable T, std::size_t tSize>
inline Vector<SignedInt<sizeof(T)>, tSize> mask2vector(Mask<T, tSize> m) {
  return {.r = reinterpret<SignedInt<sizeof(T)>>(m.r)};
}

// Convert (signed) integers to a mask
template<SignedIntVectorizable T, std::size_t tSize, Vectorizable TDst>
requires(sizeof(T) == sizeof(TDst))
inline Mask<TDst, tSize> vector2mask(Vector<T, tSize> m, TypeTag<TDst> /*tag*/) {
  return {.r = reinterpret<UnsignedInt<sizeof(T)>>(m.r)};
}
} // namespace grex::backend

#include "grex/backend/shared/operations/mask-convert.hpp" // IWYU pragma: export

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_MASK_CONVERT_HPP
