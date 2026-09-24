// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_NEON_OPERATIONS_MASK_INDEX_HPP
#define INCLUDE_GREX_BACKEND_NEON_OPERATIONS_MASK_INDEX_HPP

#include <cstddef>

#include <arm_neon.h>

#include "grex/backend/base.hpp"
#include "grex/backend/defs.hpp" // IWYU pragma: keep
#include "grex/backend/neon/operations/compare.hpp"
#include "grex/backend/neon/operations/set.hpp"
#include "grex/backend/neon/types.hpp" // IWYU pragma: keep
#include "grex/base.hpp"

namespace grex::backend {
template<Vectorizable T, std::size_t N>
inline NativeMask<T, N> cutoff_mask(std::size_t i, TypeTag<NativeMask<T, N>> /*tag*/) {
  using U = UnsignedInt<sizeof(T)>;
  const auto idxs = indices(type_tag<NativeVector<U, N>>);
  const auto ref = broadcast(U(i), type_tag<NativeVector<U, N>>);
  return {.r = compare_lt(idxs, ref).r};
}

template<Vectorizable T, std::size_t N>
inline NativeMask<T, N> single_mask(std::size_t i, TypeTag<NativeMask<T, N>> /*tag*/) {
  using U = UnsignedInt<sizeof(T)>;
  const auto idxs = indices(type_tag<NativeVector<U, N>>);
  const auto ref = broadcast(U(i), type_tag<NativeVector<U, N>>);
  return {.r = compare_eq(idxs, ref).r};
}
} // namespace grex::backend

#include "grex/backend/shared/operations/mask-index.hpp" // IWYU pragma: export

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_MASK_INDEX_HPP
