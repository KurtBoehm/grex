// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_OPERATIONS_TAGGED_HPP
#define INCLUDE_GREX_OPERATIONS_TAGGED_HPP

#include <bit>
#include <concepts>
#include <cstddef>
#include <span>

#include "grex/base/defs.hpp"
#include "grex/tags.hpp"
#include "grex/types.hpp"

namespace grex {
// zeros
template<Vectorizable T, OptValuedTag<T> TTag>
inline TagType<TTag, T> zeros(TTag /*tag*/) {
  return TagType<TTag, T>{};
}

// broadcast
template<Vectorizable T, OptValuedTag<T> TTag>
inline TagType<TTag, T> broadcast(T value, TTag /*tag*/) {
  return TagType<TTag, T>{value};
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
template<Vectorizable TIdx, OptValuedVectorTag<TIdx> TTag>
inline TagType<TTag, TIdx> indices(TTag /*tag*/) {
  return TagType<TTag, TIdx>::indices();
}
template<Vectorizable TIdx, OptValuedVectorTag<TIdx> TTag>
inline TagType<TTag, TIdx> indices(TIdx start, TTag /*tag*/) {
  return TagType<TTag, TIdx>::indices(start);
}

// Adding initialization operations for masks seems unnecessary, as a constant mask is of little use

// load
template<Vectorizable T>
inline T load(const T* src, OptValuedScalarTag<T> auto /*tag*/) {
  return *src;
}
template<Vectorizable T, OptValuedFullVectorTag<T> TTag>
inline TagType<TTag, T> load(const T* src, TTag /*tag*/) {
  return TagType<TTag, T>::load(src);
}
template<Vectorizable T, OptValuedPartVectorTag<T> TTag>
inline TagType<TTag, T> load(const T* src, TTag tag) {
  return TagType<TTag, T>::load_part(src, tag.part());
}
// TODO Add masked loading?

// load_extended
template<Vectorizable T>
inline T load_extended(const T* src, OptValuedScalarTag<T> auto /*tag*/) {
  return *src;
}
template<Vectorizable T, OptValuedVectorTag<T> TTag>
inline TagType<TTag, T> load_extended(const T* src, TTag /*tag*/) {
  return TagType<TTag, T>::load(src);
}

// is_load_valid
inline bool is_load_valid(std::size_t remaining, AnyScalarTag auto /*tag*/) {
  return remaining > 0;
}
template<AnyFullTag TTag>
inline bool is_load_valid(std::size_t remaining, TTag /*tag*/) {
  return remaining >= TTag::size;
}
template<PartVectorTag TTag>
inline bool is_load_valid(std::size_t remaining, TTag tag) {
  return remaining >= tag.part();
}
// TODO Support masked loading?

// store
template<Vectorizable T>
inline void store(T* dst, T src, OptValuedScalarTag<T> auto /*tag*/) {
  *dst = src;
}
template<Vectorizable T, OptValuedFullVectorTag<T> TTag>
inline void store(T* dst, TagType<TTag, T> src, TTag /*tag*/) {
  src.store(dst);
}
template<Vectorizable T, OptValuedPartVectorTag<T> TTag>
inline void store(T* dst, TagType<TTag, T> src, TTag tag) {
  src.store_part(dst, tag.part());
}
// TODO Support masked storing?

// gather
template<Vectorizable T, std::size_t tExtent>
inline T gather(std::span<const T, tExtent> data, IntVectorizable auto idx,
                OptValuedScalarTag<T> auto /*tag*/) {
  return data[idx];
}
template<Vectorizable T, std::size_t tExtent, OptValuedFullVectorTag<T> TTag>
inline Vector<T, TTag::size> gather(std::span<const T, tExtent> data, IntVector auto idxs,
                                    TTag /*tag*/) {
  return gather(data, idxs);
}
template<Vectorizable T, std::size_t tExtent, OptValuedPartialVectorTag<T> TTag>
inline Vector<T, TTag::size> gather(std::span<const T, tExtent> data, IntVector auto idxs,
                                    TTag tag) {
  return mask_gather(data, tag.mask(type_tag<T>), idxs);
}

// mask_gather
template<Vectorizable T, std::size_t tExtent>
inline T mask_gather(std::span<const T, tExtent> data, bool mask, IntVectorizable auto idx,
                     OptValuedScalarTag<T> auto /*tag*/) {
  return mask ? data[idx] : T{};
}
template<Vectorizable T, std::size_t tExtent, OptTypedVectorTag<T> TTag>
inline Vector<T, TTag::size> mask_gather(std::span<const T, tExtent> data, AnyMask auto mask,
                                         IntVector auto idxs, TTag tag) {
  return mask_gather(data, tag.mask(mask), idxs);
}

// expand scalar with anything
template<Vectorizable T>
inline T expand_any(T x, OptValuedScalarTag<T> auto /*tag*/) {
  return x;
}
template<Vectorizable T, OptValuedVectorTag<T> TTag>
inline TagType<TTag, T> expand_any(T x, TTag /*tag*/) {
  return TagType<TTag, T>::expanded_any(x);
}

// expand scalar with zero
template<Vectorizable T>
inline T expand_zero(T x, OptValuedScalarTag<T> auto /*tag*/) {
  return x;
}
template<Vectorizable T, OptValuedVectorTag<T> TTag>
inline TagType<TTag, T> expand_zero(T x, TTag /*tag*/) {
  return TagType<TTag, T>::expanded_zero(x);
}

// shingle_up with front=0
template<Vectorizable T>
inline T shingle_up(T /*base*/, OptValuedScalarTag<T> auto /*tag*/) {
  return T{};
}
template<AnyVector TVec, typename TTag>
requires(OptTypedFullVectorTag<TTag, TVec> || OptTypedPartVectorTag<TTag, TVec>)
inline TVec shingle_up(TVec base, TTag /*tag*/) {
  return base.shingle_up();
}
// TODO I do not know what this would be for masked tags

// shingle_up with a given front
template<Vectorizable T>
inline T shingle_up(T front, T /*base*/, OptValuedScalarTag<T> auto /*tag*/) {
  return front;
}
template<AnyVector TVec, typename TTag>
requires(OptTypedFullVectorTag<TTag, TVec> || OptTypedPartVectorTag<TTag, TVec>)
inline TVec shingle_up(typename TVec::Value front, TVec base, TTag /*tag*/) {
  return base.shingle_up(front);
}
// TODO I do not know what this would be for masked tags

// shingle_down with back=0
template<Vectorizable T>
inline T shingle_down(T /*base*/, OptValuedScalarTag<T> auto /*tag*/) {
  return T{};
}
template<AnyVector TVec>
inline TVec shingle_down(TVec base, OptTypedFullVectorTag<TVec> auto /*tag*/) {
  return base.shingle_down();
}
template<AnyVector TVec>
inline TVec shingle_down(TVec base, OptTypedPartVectorTag<TVec> auto tag) {
  return tag.mask(base).shingle_down();
}
// TODO I do not know what this would be for masked tags

// shingle_down with a given back
template<Vectorizable T>
inline T shingle_down(T /*base*/, T back, OptValuedScalarTag<T> auto /*tag*/) {
  return back;
}
template<AnyVector TVec>
inline TVec shingle_down(TVec base, typename TVec::Value back,
                         OptTypedFullVectorTag<TVec> auto /*tag*/) {
  return base.shingle_down(back);
}
template<AnyVector TVec>
inline TVec shingle_down(TVec base, typename TVec::Value back,
                         OptTypedPartVectorTag<TVec> auto tag) {
  if (tag.part() == 0) [[unlikely]] {
    return TVec{};
  }
  return base.shingle_down(back).insert(tag.part() - 1, back);
}
// TODO I do not know what this would be for masked tags

// horizontal_add
template<Vectorizable T>
inline T horizontal_add(T value, OptValuedScalarTag<T> auto /*tag*/) {
  return value;
}
template<AnyVector TVec>
inline TVec::Value horizontal_add(TVec value, OptTypedVectorTag<TVec> auto tag) {
  return horizontal_add(tag.mask(value));
}

// horizontal_min/horizontal_max
#define GREX_OPS_HMINMAX(OP) \
  template<Vectorizable T> \
  inline T OP(T value, OptValuedScalarTag<T> auto /*tag*/) { \
    return value; \
  } \
  template<AnyVector TVec> \
  inline TVec::Value OP(TVec value, OptTypedFullVectorTag<TVec> auto /*tag*/) { \
    return OP(value); \
  } \
/* TODO Partial min/max is problematic if the mask is empty: What should the placeholder be? */
GREX_OPS_HMINMAX(horizontal_min)
GREX_OPS_HMINMAX(horizontal_max)
#undef GREX_OPS_HMINMAX

// horizontal_and
inline bool horizontal_and(bool mask, AnyScalarTag auto /*tag*/) {
  return mask;
}
template<AnyMask TMask>
inline bool horizontal_and(TMask mask, OptTypedFullVectorTag<VectorFor<TMask>> auto /*tag*/) {
  return horizontal_and(mask);
}
template<AnyMask TMask>
inline bool horizontal_and(TMask mask, OptTypedPartialVectorTag<VectorFor<TMask>> auto tag) {
  return horizontal_and(mask || !tag.mask(type_tag<typename TMask::VectorValue>));
}

// load_multibyte
template<std::size_t tSrcBytes, OptValuedScalarTag<UnsignedInt<std::bit_ceil(tSrcBytes)>> TTag>
static UnsignedInt<std::bit_ceil(tSrcBytes)>
load_multibyte(const std::byte* data, IndexTag<tSrcBytes> src_bytes, TTag /*tag*/) {
  return backend::load_multibyte(data, src_bytes);
}
template<std::size_t tSrcBytes, OptValuedVectorTag<UnsignedInt<std::bit_ceil(tSrcBytes)>> TTag>
static Vector<UnsignedInt<std::bit_ceil(tSrcBytes)>, TTag::size>
load_multibyte(const std::byte* data, IndexTag<tSrcBytes> src_bytes, TTag /*tag*/) {
  using Out = Vector<UnsignedInt<std::bit_ceil(tSrcBytes)>, TTag::size>;
  return Out::load_multibyte(data, src_bytes);
}
template<MultiByteIterator TIt, AnyTag TTag>
static auto load_multibyte(TIt it, TTag tag) {
  return load_multibyte(it.raw(), index_tag<TIt::Container::element_bytes>, tag);
}

// transform
template<typename TSize = std::size_t>
GREX_ALWAYS_INLINE inline auto transform(auto op, AnyScalarTag auto /*tag*/) {
  return op(value_tag<TSize, 0>);
}
template<typename TSize = std::size_t, FullVectorTag TTag>
GREX_ALWAYS_INLINE inline auto transform(auto op, TTag /*tag*/) {
  static constexpr std::size_t size = TTag::size;
  using Val0 = decltype(op(value_tag<TSize, 0>));
  return static_apply<size>([&]<std::size_t... tIdxs>() {
    static_assert((... && std::same_as<Val0, decltype(op(value_tag<TSize, tIdxs>))>));
    return Vector<Val0, size>{op(value_tag<TSize, tIdxs>)...};
  });
}
template<typename TSize = std::size_t, PartVectorTag TTag>
GREX_ALWAYS_INLINE inline auto transform(auto op, TTag tag) {
  static constexpr std::size_t size = TTag::size;
  using Val0 = decltype(op(value_tag<TSize, 0>));
  return static_apply<size>([&]<std::size_t... tIdxs>() {
    static_assert((... && std::same_as<Val0, decltype(op(value_tag<TSize, tIdxs>))>));
    return Vector<Val0, size>{((tIdxs < tag.part()) ? op(value_tag<TSize, tIdxs>) : Val0{})...};
  });
}
// TODO Support for masked transformations?
} // namespace grex

#endif // INCLUDE_GREX_OPERATIONS_TAGGED_HPP
