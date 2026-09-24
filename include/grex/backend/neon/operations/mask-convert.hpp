// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_NEON_OPERATIONS_MASK_CONVERT_HPP
#define INCLUDE_GREX_BACKEND_NEON_OPERATIONS_MASK_CONVERT_HPP

#include <cstddef>

#include "grex/backend/base.hpp"
#include "grex/backend/defs.hpp" // IWYU pragma: keep
#include "grex/backend/neon/operations/reinterpret.hpp"
#include "grex/base.hpp"

namespace grex::backend {
// Convert a mask to signed integers
template<Vectorizable T, std::size_t N>
inline NativeVector<SignedInt<sizeof(T)>, N> mask2vector(NativeMask<T, N> m) {
  return {.r = as<SignedInt<sizeof(T)>>(m.r)};
}

// Convert (signed) integers to a mask
template<SignedIntVectorizable T, std::size_t N, Vectorizable Dst>
requires(sizeof(T) == sizeof(Dst))
inline NativeMask<Dst, N> vector2mask(NativeVector<T, N> m, TypeTag<Dst> /*tag*/) {
  return {.r = as<UnsignedInt<sizeof(T)>>(m.r)};
}
} // namespace grex::backend

#include "grex/backend/shared/operations/mask-convert.hpp" // IWYU pragma: export

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_MASK_CONVERT_HPP
