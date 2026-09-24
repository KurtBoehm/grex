// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_SET_HPP
#define INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_SET_HPP

#include <array>
#include <bit>
#include <cstddef>
#include <cstring>

#include "grex/backend/base.hpp"
#include "grex/base.hpp"

namespace grex::backend {
template<Vectorizable T, std::size_t N>
inline NativeVector<T, N> indices(TypeTag<NativeVector<T, N>> /*tag*/) {
  return static_apply<N>(
    []<std::size_t... I> { return set(type_tag<NativeVector<T, N>>, T(I)...); });
}

// SubVector
template<Vectorizable T, std::size_t N>
inline SubVector<T, N> zeros(TypeTag<SubVector<T, N>> /*tag*/) {
  return SubVector<T, N>{zeros(type_tag<NativeVector<T, min_native_size<T>>>)};
}
template<Vectorizable T, std::size_t N>
inline SubVector<T, N> undefined(TypeTag<SubVector<T, N>> /*tag*/) {
  return SubVector<T, N>{undefined(type_tag<NativeVector<T, min_native_size<T>>>)};
}
template<Vectorizable T, std::size_t N>
inline SubVector<T, N> broadcast(T value, TypeTag<SubVector<T, N>> /*tag*/) {
  return SubVector<T, N>{broadcast(value, type_tag<NativeVector<T, min_native_size<T>>>)};
}
template<Vectorizable T, std::size_t N>
inline SubVector<T, N> indices(TypeTag<SubVector<T, N>> /*tag*/) {
  return SubVector<T, N>{indices(type_tag<NativeVector<T, min_native_size<T>>>)};
}

// SubMask
template<Vectorizable T, std::size_t N>
inline SubMask<T, N> zeros(TypeTag<SubMask<T, N>> /*tag*/) {
  return SubMask<T, N>{zeros(type_tag<NativeMask<T, min_native_size<T>>>)};
}
template<Vectorizable T, std::size_t N>
inline SubMask<T, N> ones(TypeTag<SubMask<T, N>> /*tag*/) {
  return SubMask<T, N>{ones(type_tag<NativeMask<T, min_native_size<T>>>)};
}
template<Vectorizable T, std::size_t N>
inline SubMask<T, N> broadcast(bool value, TypeTag<SubMask<T, N>> /*tag*/) {
  return SubMask<T, N>{broadcast(value, type_tag<NativeMask<T, min_native_size<T>>>)};
}

// SuperVector
template<typename Half>
inline SuperVector<Half> zeros(TypeTag<SuperVector<Half>> /*tag*/) {
  const auto half = zeros(type_tag<Half>);
  return {.lower = half, .upper = half};
}
template<typename Half>
inline SuperVector<Half> undefined(TypeTag<SuperVector<Half>> /*tag*/) {
  const auto half = undefined(type_tag<Half>);
  return {.lower = half, .upper = half};
}
template<typename Half>
inline SuperVector<Half> broadcast(typename Half::Value value, TypeTag<SuperVector<Half>> /*tag*/) {
  const auto half = broadcast(value, type_tag<Half>);
  return {.lower = half, .upper = half};
}
template<typename Half, typename... Ts>
requires(sizeof...(Ts) == 2 * Half::size && std::has_single_bit(sizeof...(Ts)))
inline SuperVector<Half> set(TypeTag<SuperVector<Half>> /*tag*/, Ts... values) {
  constexpr std::size_t size = sizeof...(Ts);
  const std::array buf{values...};
  const auto op = [&]<std::size_t... I> { return set(type_tag<Half>, std::get<I>(buf)...); };
  return {.lower = static_apply<0, size / 2>(op), .upper = static_apply<size / 2, size>(op)};
}
template<typename Half>
inline SuperVector<Half> indices(TypeTag<SuperVector<Half>> /*tag*/) {
  using Vec = SuperVector<Half>;
  using Value = Vec::Value;
  constexpr std::size_t size = Vec::size;
  const auto op = []<std::size_t... I> { return set(type_tag<Vec>, Value(I)...); };
  return static_apply<0, size>(op);
}

// SuperMask
template<typename Half>
inline SuperMask<Half> zeros(TypeTag<SuperMask<Half>> /*tag*/) {
  const auto half = zeros(type_tag<Half>);
  return {.lower = half, .upper = half};
}
template<typename Half>
inline SuperMask<Half> ones(TypeTag<SuperMask<Half>> /*tag*/) {
  const auto half = ones(type_tag<Half>);
  return {.lower = half, .upper = half};
}
template<typename Half>
inline SuperMask<Half> broadcast(bool value, TypeTag<SuperMask<Half>> /*tag*/) {
  const auto half = broadcast(value, type_tag<Half>);
  return {.lower = half, .upper = half};
}
template<typename Half, typename... Ts>
requires(sizeof...(Ts) == 2 * Half::size && std::has_single_bit(sizeof...(Ts)))
inline SuperMask<Half> set(TypeTag<SuperMask<Half>> /*tag*/, Ts... values) {
  constexpr std::size_t size = sizeof...(Ts);
  const std::array buf{values...};
  const auto op = [&]<std::size_t... I> { return set(type_tag<Half>, std::get<I>(buf)...); };
  return {.lower = static_apply<0, size / 2>(op), .upper = static_apply<size / 2, size>(op)};
}

template<AnyVector Vec>
inline Vec zeros() {
  return zeros(type_tag<Vec>);
}
template<AnyVector Vec>
inline Vec broadcast(typename Vec::Value value) {
  return broadcast(value, type_tag<Vec>);
}

// Binary16: simple delegation to `u16` for basic construction.
template<std::size_t N>
inline NativeVector<f16, N> zeros(TypeTag<NativeVector<f16, N>> /*tag*/) {
  return {.r = zeros(type_tag<NativeVector<u16, N>>).r};
}
template<std::size_t N>
inline NativeVector<f16, N> undefined(TypeTag<NativeVector<f16, N>> /*tag*/) {
  return {.r = undefined(type_tag<NativeVector<u16, N>>).r};
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_SET_HPP
