// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_OPERATIONS_TAGGED_HPP
#define INCLUDE_GREX_OPERATIONS_TAGGED_HPP

#include <bit>
#include <cstddef>
#include <span>

#include "grex/backend.hpp" // IWYU pragma: keep
#include "grex/base.hpp"
#include "grex/tags.hpp"

#if !GREX_BACKEND_SCALAR
#include <concepts>

#include "grex/types.hpp"
#endif

namespace grex {
// zeros
template<Vectorizable T, OptValuedTag<T> Tag>
inline TagType<Tag, T> zeros(Tag /*tag*/) {
  return TagType<Tag, T>{};
}

// broadcast
template<Vectorizable T, OptValuedTag<T> Tag>
inline TagType<Tag, T> broadcast(T value, Tag /*tag*/) {
  return TagType<Tag, T>{value};
}

// indices
template<Vectorizable T>
inline T indices(OptValuedScalarTag<T> auto /*tag*/) {
  return T{};
}
template<Vectorizable TIdx>
inline TIdx indices(TIdx start, OptValuedScalarTag<TIdx> auto /*tag*/) {
  return start;
}
template<Vectorizable TIdx, OptValuedVectorTag<TIdx> Tag>
inline TagType<Tag, TIdx> indices(Tag /*tag*/) {
  return TagType<Tag, TIdx>::indices();
}
template<Vectorizable TIdx, OptValuedVectorTag<TIdx> Tag>
inline TagType<Tag, TIdx> indices(TIdx start, Tag /*tag*/) {
  return TagType<Tag, TIdx>::indices(start);
}

// Adding initialization operations for masks seems unnecessary, as a constant mask is of little use

// load
template<Vectorizable T>
inline T& load(T* src, OptValuedScalarTag<T> auto /*tag*/) {
  return *src;
}
template<Vectorizable T>
inline T load(const T* src, OptValuedScalarTag<T> auto /*tag*/) {
  return *src;
}
template<Vectorizable T, OptValuedFullVectorTag<T> Tag>
inline TagType<Tag, T> load(const T* src, Tag /*tag*/) {
  return TagType<Tag, T>::load(src);
}
template<Vectorizable T, OptValuedPartVectorTag<T> Tag>
inline TagType<Tag, T> load(const T* src, Tag tag) {
  return TagType<Tag, T>::load_part(src, tag.part());
}
// TODO Add masked loading?

// load_extended
template<Vectorizable T>
inline T& load_extended(T* src, OptValuedScalarTag<T> auto /*tag*/) {
  return *src;
}
template<Vectorizable T>
inline T load_extended(const T* src, OptValuedScalarTag<T> auto /*tag*/) {
  return *src;
}
template<Vectorizable T, OptValuedVectorTag<T> Tag>
inline TagType<Tag, T> load_extended(const T* src, Tag /*tag*/) {
  return TagType<Tag, T>::load(src);
}

// is_load_valid
inline bool is_load_valid(std::size_t remaining, AnyScalarTag auto /*tag*/) {
  return remaining > 0;
}
template<FullVectorTag Tag>
inline bool is_load_valid(std::size_t remaining, Tag /*tag*/) {
  return remaining >= Tag::size;
}
template<PartVectorTag Tag>
inline bool is_load_valid(std::size_t remaining, Tag tag) {
  return remaining >= tag.part();
}
// TODO Support masked loading?

// store
template<Vectorizable T>
inline void store(T* dst, T src, OptValuedScalarTag<T> auto /*tag*/) {
  *dst = src;
}
template<Vectorizable T, OptValuedFullVectorTag<T> Tag>
inline void store(T* dst, TagType<Tag, T> src, Tag /*tag*/) {
  src.store(dst);
}
template<Vectorizable T, OptValuedPartVectorTag<T> Tag>
inline void store(T* dst, TagType<Tag, T> src, Tag tag) {
  src.store_part(dst, tag.part());
}
// TODO Support masked storing?

// gather
template<Vectorizable T, std::size_t Extent>
inline T gather(std::span<const T, Extent> data, IntVectorizable auto idx,
                OptValuedScalarTag<T> auto /*tag*/) {
  return data[std::size_t(idx)];
}
#if !GREX_BACKEND_SCALAR
template<Vectorizable T, std::size_t Extent, OptValuedFullVectorTag<T> Tag>
inline Vector<T, Tag::size> gather(std::span<const T, Extent> data, IntVector auto idxs,
                                   Tag /*tag*/) {
  return gather(data, idxs);
}
template<Vectorizable T, std::size_t Extent, OptValuedPartialVectorTag<T> Tag>
inline Vector<T, Tag::size> gather(std::span<const T, Extent> data, IntVector auto idxs, Tag tag) {
  return mask_gather(data, tag.mask(type_tag<T>), idxs);
}
#endif

// mask_gather
template<Vectorizable T, std::size_t Extent>
inline T mask_gather(std::span<const T, Extent> data, bool mask, IntVectorizable auto idx,
                     OptValuedScalarTag<T> auto /*tag*/) {
  return mask ? data[std::size_t(idx)] : T{};
}
#if !GREX_BACKEND_SCALAR
template<Vectorizable T, std::size_t Extent, OptTypedVectorTag<T> Tag>
inline Vector<T, Tag::size> mask_gather(std::span<const T, Extent> data, AnyMask auto mask,
                                        IntVector auto idxs, Tag tag) {
  return mask_gather(data, tag.mask(mask), idxs);
}
#endif

// expand scalar with anything
template<Vectorizable T>
inline T expand_any(T x, OptValuedScalarTag<T> auto /*tag*/) {
  return x;
}
template<Vectorizable T, OptValuedVectorTag<T> Tag>
inline TagType<Tag, T> expand_any(T x, Tag /*tag*/) {
  return TagType<Tag, T>::expanded_any(x);
}

// expand scalar with zero
template<Vectorizable T>
inline T expand_zero(T x, OptValuedScalarTag<T> auto /*tag*/) {
  return x;
}
template<Vectorizable T, OptValuedVectorTag<T> Tag>
inline TagType<Tag, T> expand_zero(T x, Tag /*tag*/) {
  return TagType<Tag, T>::expanded_zero(x);
}

// shingle_up with front=0
template<Vectorizable T>
inline T shingle_up(T /*base*/, OptValuedScalarTag<T> auto /*tag*/) {
  return T{};
}
#if !GREX_BACKEND_SCALAR
template<AnyVector Vec, typename Tag>
requires(OptTypedFullVectorTag<Tag, Vec> || OptTypedPartVectorTag<Tag, Vec>)
inline Vec shingle_up(Vec base, Tag /*tag*/) {
  return base.shingle_up();
}
// TODO I do not know what this would be for masked tags
#endif

// shingle_up with a given front
template<Vectorizable T>
inline T shingle_up(T front, T /*base*/, OptValuedScalarTag<T> auto /*tag*/) {
  return front;
}
#if !GREX_BACKEND_SCALAR
template<AnyVector Vec, typename Tag>
requires(OptTypedFullVectorTag<Tag, Vec> || OptTypedPartVectorTag<Tag, Vec>)
inline Vec shingle_up(typename Vec::Value front, Vec base, Tag /*tag*/) {
  return base.shingle_up(front);
}
// TODO I do not know what this would be for masked tags
#endif

// shingle_down with back=0
template<Vectorizable T>
inline T shingle_down(T /*base*/, OptValuedScalarTag<T> auto /*tag*/) {
  return T{};
}
#if !GREX_BACKEND_SCALAR
template<AnyVector Vec>
inline Vec shingle_down(Vec base, OptTypedFullVectorTag<Vec> auto /*tag*/) {
  return base.shingle_down();
}
template<AnyVector Vec>
inline Vec shingle_down(Vec base, OptTypedPartVectorTag<Vec> auto tag) {
  return tag.mask(base).shingle_down();
}
// TODO I do not know what this would be for masked tags
#endif

// shingle_down with a given back
template<Vectorizable T>
inline T shingle_down(T /*base*/, T back, OptValuedScalarTag<T> auto /*tag*/) {
  return back;
}
#if !GREX_BACKEND_SCALAR
template<AnyVector Vec>
inline Vec shingle_down(Vec base, typename Vec::Value back,
                        OptTypedFullVectorTag<Vec> auto /*tag*/) {
  return base.shingle_down(back);
}
template<AnyVector Vec>
inline Vec shingle_down(Vec base, typename Vec::Value back, OptTypedPartVectorTag<Vec> auto tag) {
  if (tag.part() == 0) [[unlikely]] {
    return Vec{};
  }
  return base.shingle_down(back).insert(tag.part() - 1, back);
}
// TODO I do not know what this would be for masked tags
#endif

// horizontal_add
template<Vectorizable T>
inline T horizontal_add(T value, OptValuedScalarTag<T> auto /*tag*/) {
  return value;
}
#if !GREX_BACKEND_SCALAR
template<AnyVector Vec>
inline Vec::Value horizontal_add(Vec value, OptTypedVectorTag<Vec> auto tag) {
  return horizontal_add(tag.mask(value));
}
#endif

// horizontal_min/horizontal_max
#define GREX_OPS_HMINMAX_SCALAR(OP) \
  template<Vectorizable T> \
  inline T OP(T value, OptValuedScalarTag<T> auto /*tag*/) { \
    return value; \
  }
#if GREX_BACKEND_SCALAR
#define GREX_OPS_HMINMAX GREX_OPS_HMINMAX_SCALAR
#else
#define GREX_OPS_HMINMAX(OP) \
  GREX_OPS_HMINMAX_SCALAR(OP) \
  template<AnyVector Vec> \
  inline Vec::Value OP(Vec value, OptTypedFullVectorTag<Vec> auto /*tag*/) { \
    return OP(value); \
  }
#endif
// TODO Partial min/max is problematic if the mask is empty: What should the placeholder be?
GREX_OPS_HMINMAX(horizontal_min)
GREX_OPS_HMINMAX(horizontal_max)
#undef GREX_OPS_HMINMAX

// horizontal_and
inline bool horizontal_and(bool mask, AnyScalarTag auto /*tag*/) {
  return mask;
}
#if !GREX_BACKEND_SCALAR
template<AnyMask Mask>
inline bool horizontal_and(Mask mask, OptTypedFullVectorTag<VectorFor<Mask>> auto /*tag*/) {
  return horizontal_and(mask);
}
template<AnyMask Mask>
inline bool horizontal_and(Mask mask, OptTypedPartialVectorTag<VectorFor<Mask>> auto tag) {
  return horizontal_and(mask || !tag.mask(type_tag<typename Mask::VectorValue>));
}
#endif

// load_multibyte
template<std::size_t SrcBytes, OptValuedScalarTag<UnsignedInt<std::bit_ceil(SrcBytes)>> Tag>
static UnsignedInt<std::bit_ceil(SrcBytes)>
load_multibyte(const std::byte* data, IndexTag<SrcBytes> src_bytes, Tag /*tag*/) {
  return backend::load_multibyte(data, src_bytes);
}
#if !GREX_BACKEND_SCALAR
template<std::size_t SrcBytes, OptValuedVectorTag<UnsignedInt<std::bit_ceil(SrcBytes)>> Tag>
static Vector<UnsignedInt<std::bit_ceil(SrcBytes)>, Tag::size>
load_multibyte(const std::byte* data, IndexTag<SrcBytes> src_bytes, Tag /*tag*/) {
  using Out = Vector<UnsignedInt<std::bit_ceil(SrcBytes)>, Tag::size>;
  return Out::load_multibyte(data, src_bytes);
}
#endif
template<MultiByteIterator It, AnyTag Tag>
static auto load_multibyte(It it, Tag tag) {
  return load_multibyte(it.raw(), index_tag<It::Container::element_bytes>, tag);
}

// transform
template<typename TSize = u64>
GREX_ALWAYS_INLINE inline auto transform(auto op, OptValuedScalarTag<TSize> auto /*tag*/) {
  return op(value_tag<TSize, 0>);
}
#if !GREX_BACKEND_SCALAR
template<typename TSize = u64, OptValuedFullVectorTag<TSize> Tag>
GREX_ALWAYS_INLINE inline auto transform(auto op, Tag /*tag*/) {
  static constexpr std::size_t size = Tag::size;
  using Value = decltype(op(value_tag<TSize, 0>));
  return static_apply<size>([&]<std::size_t... I> {
    static_assert((... && std::same_as<Value, decltype(op(value_tag<TSize, I>))>));
    return Vector<Value, size>{op(value_tag<TSize, I>)...};
  });
}
template<typename TSize = u64, OptValuedPartVectorTag<TSize> Tag>
GREX_ALWAYS_INLINE inline auto transform(auto op, Tag tag) {
  static constexpr std::size_t size = Tag::size;
  using Value = decltype(op(value_tag<TSize, 0>));
  return static_apply<size>([&]<std::size_t... I> {
    static_assert((... && std::same_as<Value, decltype(op(value_tag<TSize, I>))>));
    return Vector<Value, size>{((I < tag.part()) ? op(value_tag<TSize, I>) : Value{})...};
  });
}
// TODO Support for masked transform?
#endif

template<typename TSize = u64>
inline void for_each(auto op, TypedValueTag<IterDirection> auto /*tag*/,
                     OptValuedScalarTag<TSize> auto /*tag*/) {
  op(value_tag<TSize, 0>);
}
#if !GREX_BACKEND_SCALAR
template<typename TSize = u64, OptValuedFullVectorTag<TSize> Tag>
inline void for_each(auto op, TypedValueTag<IterDirection> auto dir, Tag /*tag*/) {
  static constexpr std::size_t size = Tag::size;
  if constexpr (dir.value == IterDirection::forward) {
    for (TSize i = 0; i < size; ++i) {
      op(i);
    }
  } else {
    for (TSize i = size; i > 0; --i) {
      op(i - 1);
    }
  }
}
template<typename TSize = u64, OptValuedPartVectorTag<TSize> Tag>
inline auto for_each(auto op, TypedValueTag<IterDirection> auto dir, Tag tag) {
  const auto part = TSize(tag.part());
  if constexpr (dir.value == IterDirection::forward) {
    for (TSize i = 0; i < part; ++i) {
      op(i);
    }
  } else {
    for (TSize i = part; i > 0; --i) {
      op(i - 1);
    }
  }
}
#endif
template<typename TSize = u64>
inline void for_each(auto op, AnyTag auto tag) {
  for_each<TSize>(std::move(op), auto_tag<IterDirection::forward>, tag);
}
// TODO Support for masked for_each?
} // namespace grex

#endif // INCLUDE_GREX_OPERATIONS_TAGGED_HPP
