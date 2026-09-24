// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_MASK_INDEX_HPP
#define INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_MASK_INDEX_HPP

#include <cstddef>

#include "grex/backend/active/operations/blend.hpp"
#include "grex/backend/active/operations/set.hpp"
#include "grex/backend/base.hpp"
#include "grex/base.hpp"

namespace grex::backend {
template<Vectorizable T, std::size_t N>
inline SubMask<T, N> cutoff_mask(std::size_t i, TypeTag<SubMask<T, N>> /*tag*/) {
  return SubMask<T, N>{cutoff_mask(i, type_tag<NativeMask<T, min_native_size<T>>>)};
}
template<typename Half>
inline SuperMask<Half> cutoff_mask(std::size_t i, TypeTag<SuperMask<Half>> /*tag*/) {
  if (i <= Half::size) {
    return {.lower = cutoff_mask(i, type_tag<Half>), .upper = zeros(type_tag<Half>)};
  }
  return {.lower = ones(type_tag<Half>), .upper = cutoff_mask(i - Half::size, type_tag<Half>)};
}

template<Vectorizable T, std::size_t N>
inline SubMask<T, N> single_mask(std::size_t i, TypeTag<SubMask<T, N>> /*tag*/) {
  return SubMask<T, N>{single_mask(i, type_tag<NativeMask<T, min_native_size<T>>>)};
}
template<typename Half>
inline SuperMask<Half> single_mask(std::size_t i, TypeTag<SuperMask<Half>> /*tag*/) {
  if (i < Half::size) {
    return {.lower = single_mask(i, type_tag<Half>), .upper = zeros(type_tag<Half>)};
  }
  return {.lower = zeros(type_tag<Half>), .upper = single_mask(i - Half::size, type_tag<Half>)};
}

template<Vectorizable T, std::size_t N>
inline NativeVector<T, N> cutoff(std::size_t i, NativeVector<T, N> v) {
  return blend_zero(cutoff_mask(i, type_tag<NativeMask<T, N>>), v);
}
template<Vectorizable T, std::size_t N>
inline SubVector<T, N> cutoff(std::size_t i, SubVector<T, N> v) {
  return SubVector<T, N>{cutoff(i, v.full)};
}
template<typename Half>
inline SuperVector<Half> cutoff(std::size_t i, SuperVector<Half> v) {
  if (i <= Half::size) {
    return {.lower = cutoff(i, v.lower), .upper = zeros(type_tag<Half>)};
  }
  return {.lower = v.lower, .upper = cutoff(i - Half::size, v.upper)};
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_MASK_INDEX_HPP
