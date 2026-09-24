// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_TAGS_HPP
#define INCLUDE_GREX_TAGS_HPP

#include <concepts>
#include <cstddef>

#include "grex/backend/defs.hpp" // IWYU pragma: keep
#include "grex/base.hpp"

#if !GREX_BACKEND_SCALAR
#include "grex/operations.hpp"
#include "grex/types.hpp"
#endif

namespace grex {
//==================================================================================================
// Scalar tags
//==================================================================================================

template<Vectorizable T>
struct TypedScalarTag;

struct ScalarTag {
  using Full = ScalarTag;
  static constexpr std::size_t size = 1;

  template<Vectorizable T>
  [[nodiscard]] TypedScalarTag<T> instantiate(TypeTag<T> /*tag*/) const;
  template<Vectorizable T>
  [[nodiscard]] TypedScalarTag<T> cast(TypeTag<T> /*tag*/) const;

  template<Vectorizable T>
  [[nodiscard]] static constexpr T mask(T x) {
    return x;
  }
  [[nodiscard]] static constexpr bool mask(bool b) {
    return b;
  }
  [[nodiscard]] static constexpr bool mask() {
    return true;
  }
};
inline constexpr ScalarTag scalar_tag{};

template<Vectorizable T>
struct TypedScalarTag : ScalarTag {
  using Value = T;
};
template<Vectorizable T>
inline constexpr TypedScalarTag<T> typed_scalar_tag{};

template<Vectorizable T>
[[nodiscard]] TypedScalarTag<T> ScalarTag::instantiate(TypeTag<T> /*tag*/) const {
  return {};
}
template<Vectorizable T>
[[nodiscard]] TypedScalarTag<T> ScalarTag::cast(TypeTag<T> /*tag*/) const {
  return {};
}

#if !GREX_BACKEND_SCALAR
//==================================================================================================
// Full tags
//==================================================================================================

template<Vectorizable T, std::size_t N>
struct TypedFullTag;

template<std::size_t N>
struct FullTag {
  using Full = FullTag;
  static constexpr std::size_t size = N;

  template<Vectorizable T>
  [[nodiscard]] TypedFullTag<T, N> instantiate(TypeTag<T> /*tag*/) const;
  template<Vectorizable T>
  [[nodiscard]] TypedFullTag<T, N> cast(TypeTag<T> /*tag*/) const;

  template<SizedVector<size> Vec>
  [[nodiscard]] Vec mask(Vec v) const {
    return v;
  }
  template<SizedMask<size> Mask>
  [[nodiscard]] Mask mask(Mask m) const {
    return m;
  }
  template<Vectorizable T>
  [[nodiscard]] static auto mask(TypeTag<T> /*tag*/ = {}) {
    return Mask<T, N>::ones();
  }

  [[nodiscard]] std::size_t part() const {
    return size;
  }
};
template<std::size_t N>
inline constexpr FullTag<N> full_tag{};

template<Vectorizable T, std::size_t N>
struct TypedFullTag : FullTag<N> {
  using Value = T;

  using FullTag<N>::mask;
  [[nodiscard]] static auto mask() {
    return Mask<T, N>::ones();
  }
};
template<Vectorizable T, std::size_t N>
inline constexpr TypedFullTag<T, N> typed_full_tag{};

template<std::size_t N>
template<Vectorizable T>
[[nodiscard]] TypedFullTag<T, N> FullTag<N>::instantiate(TypeTag<T> /*tag*/) const {
  return {};
}
template<std::size_t N>
template<Vectorizable T>
[[nodiscard]] TypedFullTag<T, N> FullTag<N>::cast(TypeTag<T> /*tag*/) const {
  return {};
}

//==================================================================================================
// Partial tags, incl. masked tags
//==================================================================================================

template<Vectorizable T, std::size_t N>
struct TypedMaskedTag {
  using Full = TypedFullTag<T, N>;
  using Value = T;
  static constexpr std::size_t size = N;

  explicit TypedMaskedTag(Mask<Value, N> mask) : mask_(mask) {}

  template<Vectorizable OtherT>
  [[nodiscard]] TypedMaskedTag<OtherT, N> cast(TypeTag<OtherT> /*tag*/) const {
    return TypedMaskedTag<OtherT, N>{convert<OtherT>(mask_)};
  }

  [[nodiscard]] Vector<T, N> mask(Vector<T, N> v) const {
    return blend_zero(mask_, v);
  }
  [[nodiscard]] Mask<Value, N> mask(Mask<Value, N> m) const {
    return mask_ && m;
  }

  [[nodiscard]] Mask<Value, N> mask(TypeTag<T> /*tag*/ = {}) const {
    return mask_;
  }

private:
  Mask<Value, N> mask_;
};
template<Vectorizable T, std::size_t N>
inline TypedMaskedTag<T, N> typed_masked_tag(Mask<T, N> mask) {
  return TypedMaskedTag<T, N>{mask};
}

template<std::size_t N>
struct PartTag {
  using Full = FullTag<N>;
  static constexpr std::size_t size = N;

  explicit constexpr PartTag(std::size_t part) : part_(part) {}

  template<typename T>
  [[nodiscard]] TypedMaskedTag<T, size> instantiate(TypeTag<T> /*tag*/ = {}) const {
    return TypedMaskedTag<T, size>{mask<T>()};
  }

  template<SizedMask<size> Mask>
  [[nodiscard]] Mask mask(Mask m) const {
    return m && mask<typename Mask::VectorValue>();
  }
  template<SizedVector<size> Vec>
  [[nodiscard]] Vec mask(Vec v) const {
    return v.cutoff(part_);
  }
  template<Vectorizable T>
  [[nodiscard]] auto mask(TypeTag<T> /*tag*/ = {}) const {
    return Mask<T, N>::cutoff_mask(part_);
  }

  [[nodiscard]] std::size_t part() const {
    return part_;
  }

private:
  std::size_t part_;
};
template<std::size_t N>
inline PartTag<N> part_tag(std::size_t part) {
  return PartTag<N>{part};
}
#endif

//==================================================================================================
// Tag traits
//==================================================================================================

template<typename Tag>
struct TagTraits {
  static constexpr bool is_tag = false;
  static constexpr bool is_vector_tag = false;
  static constexpr bool is_full_tag = false;
  static constexpr bool is_part_tag = false;
};
template<>
struct TagTraits<ScalarTag> {
  static constexpr bool is_tag = true;
  static constexpr bool is_vector_tag = false;
  static constexpr bool is_full_tag = true;
  static constexpr bool is_part_tag = false;

  using Value = void;
  using Type = void;
  template<typename T>
  using AugmentedType = T;
};
template<typename T>
struct TagTraits<TypedScalarTag<T>> {
  static constexpr bool is_tag = true;
  static constexpr bool is_vector_tag = false;
  static constexpr bool is_full_tag = true;
  static constexpr bool is_part_tag = false;

  using Value = T;
  using Type = T;
  template<std::same_as<T>>
  using AugmentedType = T;
};
#if !GREX_BACKEND_SCALAR
template<std::size_t N>
struct TagTraits<FullTag<N>> {
  static constexpr bool is_tag = true;
  static constexpr bool is_vector_tag = true;
  static constexpr bool is_full_tag = true;
  static constexpr bool is_part_tag = false;

  using Value = void;
  using Type = void;
  template<typename T>
  using AugmentedType = Vector<T, N>;
};
template<typename T, std::size_t N>
struct TagTraits<TypedFullTag<T, N>> {
  static constexpr bool is_tag = true;
  static constexpr bool is_vector_tag = true;
  static constexpr bool is_full_tag = true;
  static constexpr bool is_part_tag = false;

  using Value = T;
  using Type = Vector<T, N>;
  template<std::same_as<T>>
  using AugmentedType = Vector<T, N>;
};
template<std::size_t N>
struct TagTraits<PartTag<N>> {
  static constexpr bool is_tag = true;
  static constexpr bool is_vector_tag = true;
  static constexpr bool is_full_tag = false;
  static constexpr bool is_part_tag = true;

  using Value = void;
  using Type = void;
  template<typename T>
  using AugmentedType = Vector<T, N>;
};
template<typename T, std::size_t N>
struct TagTraits<TypedMaskedTag<T, N>> {
  static constexpr bool is_tag = true;
  static constexpr bool is_vector_tag = true;
  static constexpr bool is_full_tag = false;
  static constexpr bool is_part_tag = false;

  using Value = T;
  using Type = Vector<T, N>;
  template<std::same_as<T>>
  using AugmentedType = Vector<T, N>;
};
#endif

//==================================================================================================
// Tag concepts
//==================================================================================================

template<typename Tag>
concept AnyTag = TagTraits<Tag>::is_tag;

template<typename Tag>
concept AnyVectorTag = TagTraits<Tag>::is_vector_tag;
template<typename Tag>
concept AnyScalarTag = AnyTag<Tag> && !AnyVectorTag<Tag>;

template<typename Tag>
concept AnyFullTag = TagTraits<Tag>::is_full_tag;
template<typename Tag>
concept FullVectorTag = AnyVectorTag<Tag> && AnyFullTag<Tag>;
template<typename Tag>
concept PartialVectorTag = AnyVectorTag<Tag> && !AnyFullTag<Tag>;
template<typename Tag>
concept PartVectorTag = AnyVectorTag<Tag> && TagTraits<Tag>::is_part_tag;

template<typename Tag, typename T>
concept OptValuedTag = AnyTag<Tag> && (std::is_void_v<typename TagTraits<Tag>::Value> ||
                                       std::same_as<T, typename TagTraits<Tag>::Value>);
template<typename Tag, typename T>
concept OptValuedScalarTag = AnyScalarTag<Tag> && OptValuedTag<Tag, T>;
template<typename Tag, typename T>
concept OptValuedVectorTag = AnyVectorTag<Tag> && OptValuedTag<Tag, T>;
template<typename Tag, typename T>
concept OptValuedFullVectorTag = FullVectorTag<Tag> && OptValuedTag<Tag, T>;
template<typename Tag, typename T>
concept OptValuedPartialVectorTag = PartialVectorTag<Tag> && OptValuedTag<Tag, T>;
template<typename Tag, typename T>
concept OptValuedPartVectorTag = PartVectorTag<Tag> && OptValuedTag<Tag, T>;

template<typename Tag, typename T>
concept OptTypedTag = AnyTag<Tag> && (std::is_void_v<typename TagTraits<Tag>::Type> ||
                                      std::same_as<T, typename TagTraits<Tag>::Type>);
template<typename Tag, typename T>
concept OptTypedVectorTag = AnyVectorTag<Tag> && OptTypedTag<Tag, T>;
template<typename Tag, typename T>
concept OptTypedFullVectorTag = FullVectorTag<Tag> && OptTypedTag<Tag, T>;
template<typename Tag, typename T>
concept OptTypedPartialVectorTag = PartialVectorTag<Tag> && OptTypedTag<Tag, T>;
template<typename Tag, typename T>
concept OptTypedPartVectorTag = PartVectorTag<Tag> && OptTypedTag<Tag, T>;

//==================================================================================================
// Tag-related type aliases
//==================================================================================================

template<AnyTag Tag, Vectorizable TValue>
using TagType = TagTraits<Tag>::template AugmentedType<TValue>;

#if !GREX_BACKEND_SCALAR
template<Vectorizable T>
using MinNativeTag = FullTag<min_native_size<T>>;
template<Vectorizable T>
using MaxNativeTag = FullTag<max_native_size<T>>;
#else
template<Vectorizable T>
using MinNativeTag = ScalarTag;
template<Vectorizable T>
using MaxNativeTag = ScalarTag;
#endif

template<Vectorizable T>
inline constexpr MinNativeTag<T> min_native_tag{};
template<Vectorizable T>
inline constexpr MaxNativeTag<T> max_native_tag{};

template<typename T>
struct FullTagForTrait;
template<Vectorizable T>
struct FullTagForTrait<T> : TypeTag<TypedScalarTag<T>> {};
#if !GREX_BACKEND_SCALAR
template<Vectorizable T, std::size_t N>
struct FullTagForTrait<Vector<T, N>> : TypeTag<TypedFullTag<T, N>> {};
#endif
template<typename T>
using FullTagFor = FullTagForTrait<T>::Type;
template<typename T>
inline constexpr FullTagFor<T> full_tag_for{};
} // namespace grex

#endif // INCLUDE_GREX_TAGS_HPP
