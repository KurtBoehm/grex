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

#include "grex/backend/defs.hpp"
#include "grex/base.hpp"

namespace grex::backend {
template<Vectorizable T, std::size_t tSize>
inline Vector<T, tSize> indices(TypeTag<Vector<T, tSize>> /*tag*/) {
  return static_apply<tSize>(
    []<std::size_t... tIdxs>() { return set(type_tag<Vector<T, tSize>>, T(tIdxs)...); });
}

// SubVector
template<Vectorizable T, std::size_t tPart, std::size_t tSize, typename... Ts>
inline SubVector<T, tPart, tSize> zeros(TypeTag<SubVector<T, tPart, tSize>> /*tag*/) {
  return SubVector<T, tPart, tSize>{zeros(type_tag<Vector<T, tSize>>)};
}
template<Vectorizable T, std::size_t tPart, std::size_t tSize, typename... Ts>
inline SubVector<T, tPart, tSize> undefined(TypeTag<SubVector<T, tPart, tSize>> /*tag*/) {
  return SubVector<T, tPart, tSize>{undefined(type_tag<Vector<T, tSize>>)};
}
template<Vectorizable T, std::size_t tPart, std::size_t tSize, typename... Ts>
inline SubVector<T, tPart, tSize> broadcast(T value, TypeTag<SubVector<T, tPart, tSize>> /*tag*/) {
  return SubVector<T, tPart, tSize>{broadcast(value, type_tag<Vector<T, tSize>>)};
}
template<Vectorizable T, std::size_t tPart, std::size_t tSize, typename... Ts>
inline SubVector<T, tPart, tSize> set(TypeTag<SubVector<T, tPart, tSize>> /*tag*/, Ts... values) {
  std::array<T, tSize> elements{values...};
  const auto full = static_apply<tSize>([&]<std::size_t... tIdxs> {
    return set(type_tag<Vector<T, tSize>>, std::get<tIdxs>(elements)...);
  });
  return SubVector<T, tPart, tSize>{full};
}
template<Vectorizable T, std::size_t tPart, std::size_t tSize, typename... Ts>
inline SubVector<T, tPart, tSize> indices(TypeTag<SubVector<T, tPart, tSize>> /*tag*/) {
  return SubVector<T, tPart, tSize>{indices(type_tag<Vector<T, tSize>>)};
}

// SubMask
template<Vectorizable T, std::size_t tPart, std::size_t tSize>
inline SubMask<T, tPart, tSize> zeros(TypeTag<SubMask<T, tPart, tSize>> /*tag*/) {
  return SubMask<T, tPart, tSize>{zeros(type_tag<Mask<T, tSize>>)};
}
template<Vectorizable T, std::size_t tPart, std::size_t tSize>
inline SubMask<T, tPart, tSize> ones(TypeTag<SubMask<T, tPart, tSize>> /*tag*/) {
  return SubMask<T, tPart, tSize>{ones(type_tag<Mask<T, tSize>>)};
}
template<Vectorizable T, std::size_t tPart, std::size_t tSize, typename... Ts>
inline SubMask<T, tPart, tSize> broadcast(bool value, TypeTag<SubMask<T, tPart, tSize>> /*tag*/) {
  return SubMask<T, tPart, tSize>{broadcast(value, type_tag<Mask<T, tSize>>)};
}
template<Vectorizable T, std::size_t tPart, std::size_t tSize, typename... Ts>
inline SubMask<T, tPart, tSize> set(TypeTag<SubMask<T, tPart, tSize>> /*tag*/, Ts... values) {
  std::array<bool, tSize> buf{values...};
  const auto full = static_apply<tSize>(
    [&]<std::size_t... tIdxs>() { return set(type_tag<Mask<T, tSize>>, buf[tIdxs]...); });
  return SubMask<T, tPart, tSize>{full};
}

// SuperVector
template<typename THalf>
inline SuperVector<THalf> zeros(TypeTag<SuperVector<THalf>> /*tag*/) {
  const auto half = zeros(type_tag<THalf>);
  return {.lower = half, .upper = half};
}
template<typename THalf>
inline SuperVector<THalf> undefined(TypeTag<SuperVector<THalf>> /*tag*/) {
  const auto half = undefined(type_tag<THalf>);
  return {.lower = half, .upper = half};
}
template<typename THalf>
inline SuperVector<THalf> broadcast(typename THalf::Value value,
                                    TypeTag<SuperVector<THalf>> /*tag*/) {
  const auto half = broadcast(value, type_tag<THalf>);
  return {.lower = half, .upper = half};
}
template<typename THalf, typename... Ts>
requires(sizeof...(Ts) == 2 * THalf::size && std::has_single_bit(sizeof...(Ts)))
inline SuperVector<THalf> set(TypeTag<SuperVector<THalf>> /*tag*/, Ts... values) {
  constexpr std::size_t size = sizeof...(Ts);
  const std::array buf{values...};
  auto op = [&]<std::size_t... tIdxs>() { return set(type_tag<THalf>, std::get<tIdxs>(buf)...); };
  return {.lower = static_apply<0, size / 2>(op), .upper = static_apply<size / 2, size>(op)};
}
template<typename THalf>
inline SuperVector<THalf> indices(TypeTag<SuperVector<THalf>> /*tag*/) {
  using Vec = SuperVector<THalf>;
  using Value = Vec::Value;
  constexpr std::size_t size = Vec::size;
  auto op = []<std::size_t... tIdxs>() { return set(type_tag<THalf>, Value(tIdxs)...); };
  return {.lower = static_apply<0, size / 2>(op), .upper = static_apply<size / 2, size>(op)};
}

// SuperMask
template<typename THalf>
inline SuperMask<THalf> zeros(TypeTag<SuperMask<THalf>> /*tag*/) {
  const auto half = zeros(type_tag<THalf>);
  return {.lower = half, .upper = half};
}
template<typename THalf>
inline SuperMask<THalf> ones(TypeTag<SuperMask<THalf>> /*tag*/) {
  const auto half = ones(type_tag<THalf>);
  return {.lower = half, .upper = half};
}
template<typename THalf>
inline SuperMask<THalf> broadcast(bool value, TypeTag<SuperMask<THalf>> /*tag*/) {
  const auto half = broadcast(value, type_tag<THalf>);
  return {.lower = half, .upper = half};
}
template<typename THalf, typename... Ts>
requires(sizeof...(Ts) == 2 * THalf::size && std::has_single_bit(sizeof...(Ts)))
inline SuperMask<THalf> set(TypeTag<SuperMask<THalf>> /*tag*/, Ts... values) {
  constexpr std::size_t size = sizeof...(Ts);
  const std::array buf{values...};
  auto op = [&]<std::size_t... tIdxs>() { return set(type_tag<THalf>, std::get<tIdxs>(buf)...); };
  return {.lower = static_apply<0, size / 2>(op), .upper = static_apply<size / 2, size>(op)};
}

template<AnyVector TVec>
inline TVec zeros() {
  return zeros(type_tag<TVec>);
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_SET_HPP
