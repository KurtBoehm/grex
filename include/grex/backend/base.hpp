// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_BASE_HPP
#define INCLUDE_GREX_BACKEND_BASE_HPP

#include <cstddef>

#include "grex/base.hpp"

namespace grex::backend {
template<Vectorizable T>
struct Scalar {
  T value;
};

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

  [[nodiscard]] Full native() const {
    return full;
  }
  [[nodiscard]] Register registr() const {
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

enum struct SimdKind : u8 { none, native, subnative, supernative };
template<typename T>
struct AnyVectorTrait {
  static constexpr bool is_vector = false;
  static constexpr bool has_register = false;
  static constexpr SimdKind kind = SimdKind::none;
};
template<Vectorizable T, std::size_t tSize>
struct AnyVectorTrait<Vector<T, tSize>> {
  static constexpr bool is_vector = true;
  static constexpr bool has_register = true;
  static constexpr SimdKind kind = SimdKind::native;
};
template<Vectorizable T, std::size_t tPart, std::size_t tSize>
struct AnyVectorTrait<SubVector<T, tPart, tSize>> {
  static constexpr bool is_vector = true;
  static constexpr bool has_register = true;
  static constexpr SimdKind kind = SimdKind::subnative;
};
template<typename THalf>
struct AnyVectorTrait<SuperVector<THalf>> {
  static constexpr bool is_vector = true;
  static constexpr bool has_register = false;
  static constexpr SimdKind kind = SimdKind::supernative;
};
template<typename T>
concept AnyVector = AnyVectorTrait<T>::is_vector;
template<typename T>
concept AnyNativeVector = AnyVectorTrait<T>::kind == SimdKind::native;
template<typename T>
concept AnySubNativeVector = AnyVectorTrait<T>::kind == SimdKind::subnative;
template<typename T>
concept AnySuperNativeVector = AnyVectorTrait<T>::kind == SimdKind::supernative;

template<Vectorizable T, std::size_t tSize>
struct Mask;
template<Vectorizable T, std::size_t tPart, std::size_t tSize>
struct SubMask {
  using Full = Mask<T, tSize>;
  using Register = Full::Register;
  using VectorValue = T;
  static constexpr std::size_t size = tPart;

  Full full;

  explicit SubMask(Full m) : full{m} {}
  explicit SubMask(Register r) : full{.r = r} {}

  [[nodiscard]] Register registr() const {
    return full.r;
  }
};
template<typename THalf>
struct SuperMask {
  using VectorValue = THalf::VectorValue;
  static constexpr std::size_t size = 2 * THalf::size;

  THalf lower;
  THalf upper;
};
template<Vectorizable T, std::size_t tSize>
using MaskPair = SuperMask<Mask<T, tSize>>;

template<typename T>
struct AnyMaskTrait {
  static constexpr bool is_vector = false;
  static constexpr bool has_register = false;
  static constexpr SimdKind kind = SimdKind::none;
};
template<Vectorizable T, std::size_t tSize>
struct AnyMaskTrait<Mask<T, tSize>> {
  static constexpr bool is_vector = true;
  static constexpr bool has_register = true;
  static constexpr SimdKind kind = SimdKind::native;
};
template<Vectorizable T, std::size_t tPart, std::size_t tSize>
struct AnyMaskTrait<SubMask<T, tPart, tSize>> {
  static constexpr bool is_vector = true;
  static constexpr bool has_register = true;
  static constexpr SimdKind kind = SimdKind::subnative;
};
template<typename THalf>
struct AnyMaskTrait<SuperMask<THalf>> {
  static constexpr bool is_vector = true;
  static constexpr bool has_register = false;
  static constexpr SimdKind kind = SimdKind::supernative;
};
template<typename T>
concept AnyMask = AnyMaskTrait<T>::is_vector;
template<typename T>
concept AnyNativeMask = AnyMaskTrait<T>::kind == SimdKind::native;
template<typename T>
concept AnySubNativeMask = AnyMaskTrait<T>::kind == SimdKind::subnative;
template<typename T>
concept AnySuperNativeMask = AnyMaskTrait<T>::kind == SimdKind::supernative;
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_BASE_HPP
