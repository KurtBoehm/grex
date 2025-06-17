// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_DEFS_HPP
#define INCLUDE_GREX_BACKEND_DEFS_HPP

#include <cstddef>

#include "grex/base/defs.hpp"

namespace grex::backend {
template<Vectorizable T, std::size_t tSize>
struct Vector;
template<Vectorizable T, std::size_t tPart, std::size_t tSize>
struct SubVector {
  using Full = Vector<T, tSize>;
  using Register = Full::Register;
  using Value = T;
  static constexpr std::size_t size = tPart;

  Full full;

  explicit SubVector(Full v) : full{v} {}
  explicit SubVector(Register r) : full{.r = r} {}

  Register registr() const {
    return full.r;
  }
};
template<typename THalf>
struct SuperVector {
  using Value = THalf::Value;
  static constexpr std::size_t size = 2 * THalf::size;

  THalf lower;
  THalf upper;
};
template<Vectorizable T, std::size_t tSize>
using VectorPair = SuperVector<Vector<T, tSize>>;

enum struct VectorKind : u8 { none, native, subnative, supernative };
template<typename T>
struct AnyVectorTrait {
  static constexpr bool is_vector = false;
  static constexpr bool has_register = false;
  static constexpr VectorKind kind = VectorKind::none;
};
template<Vectorizable T, std::size_t tSize>
struct AnyVectorTrait<Vector<T, tSize>> {
  static constexpr bool is_vector = true;
  static constexpr bool has_register = true;
  static constexpr VectorKind kind = VectorKind::native;
};
template<Vectorizable T, std::size_t tPart, std::size_t tSize>
struct AnyVectorTrait<SubVector<T, tPart, tSize>> {
  static constexpr bool is_vector = true;
  static constexpr bool has_register = true;
  static constexpr VectorKind kind = VectorKind::subnative;
};
template<typename THalf>
struct AnyVectorTrait<SuperVector<THalf>> {
  static constexpr bool is_vector = true;
  static constexpr bool has_register = false;
  static constexpr VectorKind kind = VectorKind::supernative;
};
template<typename T>
concept AnyVector = AnyVectorTrait<T>::is_vector;
template<typename T>
concept AnyNativeVector = AnyVectorTrait<T>::kind == VectorKind::native;
template<typename T>
concept AnySubNativeVector = AnyVectorTrait<T>::kind == VectorKind::subnative;
template<typename T>
concept AnySuperNativeVector = AnyVectorTrait<T>::kind == VectorKind::supernative;

template<Vectorizable T, std::size_t tSize>
struct Mask;
template<Vectorizable T, std::size_t tPart, std::size_t tSize>
struct SubMask {
  using Full = Mask<T, tSize>;
  using Register = Full::Register;
  static constexpr std::size_t size = tPart;

  Full full;

  explicit SubMask(Full m) : full{m} {}
  explicit SubMask(Register r) : full{.r = r} {}

  Register registr() const {
    return full.r;
  }
};
template<typename THalf>
struct SuperMask {
  static constexpr std::size_t size = 2 * THalf::size;

  THalf lower;
  THalf upper;
};
template<Vectorizable T, std::size_t tSize>
using MaskPair = SuperMask<Mask<T, tSize>>;
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_DEFS_HPP
