// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_BASE_HPP
#define INCLUDE_GREX_BACKEND_BASE_HPP

#include <concepts>
#include <cstddef>

#include "grex/backend/active/sizes.hpp"
#include "grex/backend/defs.hpp" // IWYU pragma: keep
#include "grex/base.hpp"

namespace grex {
//==================================================================================================
// Type concepts
//==================================================================================================

// All floating-point types with native arithmetic support.
#if GREX_F16_NATIVE_ARITHMETIC
template<typename T>
concept NativeFloatVectorizable = FloatVectorizable<T>;
#else
template<typename T>
concept NativeFloatVectorizable = FullFloatVectorizable<T>;
#endif
} // namespace grex

namespace grex::backend {
//==================================================================================================
// Fused multiply-add family
//==================================================================================================

/** Tag selecting `a · b + c`. */
struct MultiplyAdd {};

/** Tag selecting `a · b - c`. */
struct MultiplySubtract {};

/** Tag selecting `-(a · b) + c`. */
struct NegatedMultiplyAdd {};

/** Tag selecting `-(a · b) - c`. */
struct NegatedMultiplySubtract {};

template<typename T>
concept FusedTag = std::same_as<T, MultiplyAdd> || std::same_as<T, MultiplySubtract> ||
                   std::same_as<T, NegatedMultiplyAdd> || std::same_as<T, NegatedMultiplySubtract>;

//==================================================================================================
// Vector types
//==================================================================================================

template<Vectorizable T, std::size_t N>
struct NativeVector;

#if !GREX_BACKEND_SCALAR
// Binary16 vectors are stored in the register that the back-end uses for `u16`, which makes every
// operation that only moves bits around a plain delegation to `u16`, whereas operations that
// natively operate on binary16 must cast to the appropriate register type.
template<std::size_t N>
requires(is_native<u16, N>)
struct NativeVector<f16, N> {
  using Register = NativeVector<u16, N>::Register;
  using Value = f16;
  static constexpr std::size_t size = N;
  static constexpr std::size_t bytes = sizeof(Value) * size;

  Register r;

  [[nodiscard]] NativeVector native() const {
    return *this;
  }
  [[nodiscard]] Register registr() const {
    return r;
  }
};
#endif

template<Vectorizable T, std::size_t N>
struct SubVector {
  using Full = NativeVector<T, min_native_size<T>>;
  using Register = Full::Register;
  using Value = T;
  static constexpr std::size_t size = N;
  static constexpr std::size_t full_size = min_native_size<T>;
  static constexpr std::size_t bytes = sizeof(Value) * size;

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
template<typename Half>
struct SuperVector {
  using Value = Half::Value;
  static constexpr std::size_t size = 2 * Half::size;
  static constexpr std::size_t bytes = sizeof(Value) * size;

  Half lower;
  Half upper;
};

enum struct SimdKind : u8 { none, native, subnative, supernative };
template<typename T>
struct AnyVectorTrait {
  static constexpr bool is_vector = false;
  static constexpr bool has_register = false;
  static constexpr SimdKind kind = SimdKind::none;
};
template<Vectorizable T, std::size_t N>
struct AnyVectorTrait<NativeVector<T, N>> {
  static constexpr bool is_vector = true;
  static constexpr bool has_register = true;
  static constexpr SimdKind kind = SimdKind::native;
};
template<Vectorizable T, std::size_t N>
struct AnyVectorTrait<SubVector<T, N>> {
  static constexpr bool is_vector = true;
  static constexpr bool has_register = true;
  static constexpr SimdKind kind = SimdKind::subnative;
};
template<typename Half>
struct AnyVectorTrait<SuperVector<Half>> {
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

template<AnyVector Vec>
using ValueOf = Vec::Value;
template<AnyVector Vec>
inline constexpr std::size_t size_of = Vec::size;

template<typename T>
concept IntVector = AnyVector<T> && IntVectorizable<ValueOf<T>>;
template<typename T>
concept UnsignedIntVector = AnyVector<T> && UnsignedIntVectorizable<ValueOf<T>>;
template<typename T>
concept FloatVector = AnyVector<T> && FloatVectorizable<ValueOf<T>>;
/** Any floating-point vector with natively supported arithmetic. */
template<typename T>
concept NativeFloatVector = AnyVector<T> && NativeFloatVectorizable<ValueOf<T>>;

/** Any vector of binary16 values. */
template<typename T>
concept Float16Vector = AnyVector<T> && Float16<ValueOf<T>>;
/** Any vector of 8-bit integer values. */
template<typename T>
concept Int8Vector = AnyVector<T> && Int8<ValueOf<T>>;

template<typename T, typename V>
concept TypedVector = AnyVector<T> && std::same_as<ValueOf<T>, V>;

template<Vectorizable T, std::size_t N>
struct NativeMask;
#if !GREX_BACKEND_SCALAR
// See the comment on `NativeVector<f16, N>` above: binary16 masks reuse `u16`’s register too.
template<std::size_t N>
requires(is_native<u16, N>)
struct NativeMask<f16, N> {
  using Register = NativeMask<u16, N>::Register;
  using VectorValue = f16;
  static constexpr std::size_t size = N;
  static constexpr std::size_t bytes = sizeof(VectorValue) * size;
  // Only used by the x86-64 back-end, where it denotes the width of the associated vector register
  static constexpr std::size_t rbits = 16 * N;

  Register r;

  [[nodiscard]] Register registr() const {
    return r;
  }
};
#endif

template<Vectorizable T, std::size_t N>
struct SubMask {
  using Full = NativeMask<T, min_native_size<T>>;
  using Register = Full::Register;
  using VectorValue = T;
  static constexpr std::size_t size = N;
  static constexpr std::size_t full_size = min_native_size<T>;
  static constexpr std::size_t bytes = sizeof(VectorValue) * size;

  Full full;

  explicit SubMask(Full m) : full{m} {}
  explicit SubMask(Register r) : full{.r = r} {}

  [[nodiscard]] Register registr() const {
    return full.r;
  }
};
template<typename Half>
struct SuperMask {
  using VectorValue = Half::VectorValue;
  static constexpr std::size_t size = 2 * Half::size;
  static constexpr std::size_t bytes = sizeof(VectorValue) * size;

  Half lower;
  Half upper;
};
template<Vectorizable T, std::size_t N>
using MaskPair = SuperMask<NativeMask<T, N>>;

template<typename T>
struct AnyMaskTrait {
  static constexpr bool is_vector = false;
  static constexpr bool has_register = false;
  static constexpr SimdKind kind = SimdKind::none;
};
template<Vectorizable T, std::size_t N>
struct AnyMaskTrait<NativeMask<T, N>> {
  static constexpr bool is_vector = true;
  static constexpr bool has_register = true;
  static constexpr SimdKind kind = SimdKind::native;
};
template<Vectorizable T, std::size_t N>
struct AnyMaskTrait<SubMask<T, N>> {
  static constexpr bool is_vector = true;
  static constexpr bool has_register = true;
  static constexpr SimdKind kind = SimdKind::subnative;
};
template<typename Half>
struct AnyMaskTrait<SuperMask<Half>> {
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
