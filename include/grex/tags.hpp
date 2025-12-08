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
/////////////////
// scalar tags //
/////////////////

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
  [[nodiscard]] constexpr T mask(T x) const {
    return x;
  }
  [[nodiscard]] constexpr bool mask(bool b) const {
    return b;
  }
  [[nodiscard]] constexpr bool mask() const {
    return true;
  }
};
inline constexpr ScalarTag scalar_tag{};

template<Vectorizable T>
struct TypedScalarTag : public ScalarTag {
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
///////////////
// full tags //
///////////////

template<Vectorizable T, std::size_t tSize>
struct TypedFullTag;

template<std::size_t tSize>
struct FullTag {
  using Full = FullTag;
  static constexpr std::size_t size = tSize;

  template<Vectorizable T>
  [[nodiscard]] TypedFullTag<T, tSize> instantiate(TypeTag<T> /*tag*/) const;
  template<Vectorizable T>
  [[nodiscard]] TypedFullTag<T, tSize> cast(TypeTag<T> /*tag*/) const;

  template<SizedVector<size> TVec>
  [[nodiscard]] TVec mask(TVec v) const {
    return v;
  }
  template<SizedMask<size> TMask>
  [[nodiscard]] TMask mask(TMask m) const {
    return m;
  }
  template<Vectorizable T>
  auto mask(TypeTag<T> /*tag*/ = {}) const {
    return Mask<T, tSize>::ones();
  }

  [[nodiscard]] std::size_t part() const {
    return size;
  }
};
template<std::size_t tSize>
inline constexpr FullTag<tSize> full_tag{};

template<Vectorizable T, std::size_t tSize>
struct TypedFullTag : public FullTag<tSize> {
  using Value = T;

  using FullTag<tSize>::mask;
  auto mask() const {
    return Mask<T, tSize>::ones();
  }
};
template<Vectorizable T, std::size_t tSize>
inline constexpr TypedFullTag<T, tSize> typed_full_tag{};

template<std::size_t tSize>
template<Vectorizable T>
[[nodiscard]] TypedFullTag<T, tSize> FullTag<tSize>::instantiate(TypeTag<T> /*tag*/) const {
  return {};
}
template<std::size_t tSize>
template<Vectorizable T>
[[nodiscard]] TypedFullTag<T, tSize> FullTag<tSize>::cast(TypeTag<T> /*tag*/) const {
  return {};
}

////////////////////////////////////
// partial tags incl. masked tags //
////////////////////////////////////

template<Vectorizable T, std::size_t tSize>
struct TypedMaskedTag {
  using Full = TypedFullTag<T, tSize>;
  using Value = T;
  static constexpr std::size_t size = tSize;

  explicit TypedMaskedTag(Mask<Value, tSize> mask) : mask_(mask) {}

  template<Vectorizable TOther>
  [[nodiscard]] TypedMaskedTag<TOther, tSize> cast(TypeTag<TOther> /*tag*/) const {
    return TypedMaskedTag<TOther, tSize>{convert_unsafe<TOther>(mask_)};
  }

  [[nodiscard]] Vector<T, tSize> mask(Vector<T, tSize> v) const {
    return blend_zero(mask_, v);
  }
  [[nodiscard]] Mask<Value, tSize> mask(Mask<Value, tSize> m) const {
    return mask_ && m;
  }

  [[nodiscard]] Mask<Value, tSize> mask(TypeTag<T> /*tag*/ = {}) const {
    return mask_;
  }

private:
  Mask<Value, tSize> mask_;
};
template<Vectorizable T, std::size_t tSize>
inline TypedMaskedTag<T, tSize> typed_masked_tag(Mask<T, tSize> mask) {
  return TypedMaskedTag<T, tSize>{mask};
}

template<std::size_t tSize>
struct PartTag {
  using Full = FullTag<tSize>;
  static constexpr std::size_t size = tSize;

  explicit constexpr PartTag(std::size_t part) : part_(part) {}

  template<typename T>
  [[nodiscard]] TypedMaskedTag<T, size> instantiate(TypeTag<T> /*tag*/ = {}) const {
    return TypedMaskedTag<T, size>{mask<T>()};
  }

  template<SizedMask<size> TMask>
  [[nodiscard]] TMask mask(TMask m) const {
    return m && mask<typename TMask::VectorValue>();
  }
  template<SizedVector<size> TVec>
  [[nodiscard]] TVec mask(TVec v) const {
    return v.cutoff(part_);
  }
  template<Vectorizable T>
  auto mask(TypeTag<T> /*tag*/ = {}) const {
    return Mask<T, tSize>::cutoff_mask(part_);
  }

  [[nodiscard]] std::size_t part() const {
    return part_;
  }

private:
  std::size_t part_;
};
template<std::size_t tSize>
inline PartTag<tSize> part_tag(std::size_t part) {
  return PartTag<tSize>{part};
}
#endif

template<typename TTag>
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
template<std::size_t tSize>
struct TagTraits<FullTag<tSize>> {
  static constexpr bool is_tag = true;
  static constexpr bool is_vector_tag = true;
  static constexpr bool is_full_tag = true;
  static constexpr bool is_part_tag = false;

  using Value = void;
  using Type = void;
  template<typename T>
  using AugmentedType = Vector<T, tSize>;
};
template<typename T, std::size_t tSize>
struct TagTraits<TypedFullTag<T, tSize>> {
  static constexpr bool is_tag = true;
  static constexpr bool is_vector_tag = true;
  static constexpr bool is_full_tag = true;
  static constexpr bool is_part_tag = false;

  using Value = T;
  using Type = Vector<T, tSize>;
  template<std::same_as<T>>
  using AugmentedType = Vector<T, tSize>;
};
template<std::size_t tSize>
struct TagTraits<PartTag<tSize>> {
  static constexpr bool is_tag = true;
  static constexpr bool is_vector_tag = true;
  static constexpr bool is_full_tag = false;
  static constexpr bool is_part_tag = true;

  using Value = void;
  using Type = void;
  template<typename T>
  using AugmentedType = Vector<T, tSize>;
};
template<typename T, std::size_t tSize>
struct TagTraits<TypedMaskedTag<T, tSize>> {
  static constexpr bool is_tag = true;
  static constexpr bool is_vector_tag = true;
  static constexpr bool is_full_tag = false;
  static constexpr bool is_part_tag = false;

  using Value = T;
  using Type = Vector<T, tSize>;
  template<std::same_as<T>>
  using AugmentedType = Vector<T, tSize>;
};
#endif

template<typename TTag>
concept AnyTag = TagTraits<TTag>::is_tag;

template<typename TTag>
concept AnyVectorTag = TagTraits<TTag>::is_vector_tag;
template<typename TTag>
concept AnyScalarTag = AnyTag<TTag> && !AnyVectorTag<TTag>;

template<typename TTag>
concept AnyFullTag = TagTraits<TTag>::is_full_tag;
template<typename TTag>
concept FullVectorTag = AnyVectorTag<TTag> && AnyFullTag<TTag>;
template<typename TTag>
concept PartialVectorTag = AnyVectorTag<TTag> && !AnyFullTag<TTag>;
template<typename TTag>
concept PartVectorTag = AnyVectorTag<TTag> && TagTraits<TTag>::is_part_tag;

template<typename TTag, typename T>
concept OptValuedTag = AnyTag<TTag> && (std::is_void_v<typename TagTraits<TTag>::Value> ||
                                        std::same_as<T, typename TagTraits<TTag>::Value>);
template<typename TTag, typename T>
concept OptValuedScalarTag = AnyScalarTag<TTag> && OptValuedTag<TTag, T>;
template<typename TTag, typename T>
concept OptValuedVectorTag = AnyVectorTag<TTag> && OptValuedTag<TTag, T>;
template<typename TTag, typename T>
concept OptValuedFullVectorTag = FullVectorTag<TTag> && OptValuedTag<TTag, T>;
template<typename TTag, typename T>
concept OptValuedPartialVectorTag = PartialVectorTag<TTag> && OptValuedTag<TTag, T>;
template<typename TTag, typename T>
concept OptValuedPartVectorTag = PartVectorTag<TTag> && OptValuedTag<TTag, T>;

template<typename TTag, typename T>
concept OptTypedTag = AnyTag<TTag> && (std::is_void_v<typename TagTraits<TTag>::Type> ||
                                       std::same_as<T, typename TagTraits<TTag>::Type>);
template<typename TTag, typename T>
concept OptTypedVectorTag = AnyVectorTag<TTag> && OptTypedTag<TTag, T>;
template<typename TTag, typename T>
concept OptTypedFullVectorTag = FullVectorTag<TTag> && OptTypedTag<TTag, T>;
template<typename TTag, typename T>
concept OptTypedPartialVectorTag = PartialVectorTag<TTag> && OptTypedTag<TTag, T>;
template<typename TTag, typename T>
concept OptTypedPartVectorTag = PartVectorTag<TTag> && OptTypedTag<TTag, T>;

template<AnyTag TTag, Vectorizable TValue>
using TagType = TagTraits<TTag>::template AugmentedType<TValue>;

template<Vectorizable TValue, typename TTag>
struct TagValueTrait;
template<Vectorizable TValue, AnyScalarTag TTag>
struct TagValueTrait<TValue, TTag> {
  using Type = TValue;
};
template<Vectorizable TVector, AnyVectorTag TTag>
struct TagValueTrait<TVector, TTag> {
  using Type = TVector::Value;
};
template<Vectorizable TValue, AnyTag TTag>
using TagValue = TagValueTrait<TValue, TTag>::Type;

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
} // namespace grex

#endif // INCLUDE_GREX_TAGS_HPP
