// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_OPERATIONS_HPP
#define INCLUDE_GREX_OPERATIONS_HPP

#include <cmath>
#include <cstddef>
#include <limits>
#include <span>

#include "grex/base/defs.hpp"
#include "grex/tags.hpp"
#include "grex/types.hpp"

namespace grex {
// indices
template<Vectorizable TIdx>
inline TIdx indices(TIdx start, OptValuedScalarTag<TIdx> auto /*tag*/) {
  return start;
}
template<Vectorizable TIdx, OptValuedVectorTag<TIdx> TTag>
inline TagType<TTag, TIdx> indices(TIdx start, TTag /*tag*/) {
  return TagType<TTag, TIdx>::indices(start);
}

// blend
template<typename T>
inline T blend(bool selector, T v0, T v1, OptValuedScalarTag<T> auto /*tag*/) {
  return selector ? v1 : v0;
}
template<AnyVector TVec>
inline TVec blend(MaskFor<TVec> mask, TVec v0, TVec v1, OptTypedVectorTag<TVec> auto /*tag*/) {
  return blend(mask, v0, v1);
}

// load_ptr
template<Vectorizable T>
inline T load_ptr(const T* src, OptValuedScalarTag<T> auto /*tag*/) {
  return *src;
}
template<Vectorizable T, OptValuedFullVectorTag<T> TTag>
inline TagType<TTag, T> load_ptr(const T* src, TTag /*tag*/) {
  return TagType<TTag, T>::load(src);
}
template<Vectorizable T, OptValuedPartVectorTag<T> TTag>
inline TagType<TTag, T> load_ptr(const T* src, TTag tag) {
  return TagType<TTag, T>::load_part(src, tag.part());
}
// TODO Add masked loading?

// load_ptr_extended
template<Vectorizable T>
inline T load_ptr_extended(const T* src, OptValuedScalarTag<T> auto /*tag*/) {
  return *src;
}
template<Vectorizable T, OptValuedVectorTag<T> TTag>
inline TagType<TTag, T> load_ptr_extended(const T* src, TTag /*tag*/) {
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

// store_ptr
template<Vectorizable T>
inline void store_ptr(T* dst, T src, OptValuedScalarTag<T> auto /*tag*/) {
  *dst = src;
}
template<Vectorizable T, OptValuedFullVectorTag<T> TTag>
inline void store_ptr(T* dst, TagType<TTag, T> src, TTag /*tag*/) {
  src.store(dst);
}
template<Vectorizable T, OptValuedPartVectorTag<T> TTag>
inline void store_ptr(T* dst, TagType<TTag, T> src, TTag tag) {
  src.store_part(dst, tag.part());
}
// TODO Support masked storing?

template<typename TDst, typename TSrc>
concept SafeConversion = std::numeric_limits<TDst>::digits >= std::numeric_limits<TSrc>::digits;

// convert
template<Vectorizable TDst, Vectorizable TSrc, bool tSafe>
requires(!tSafe || SafeConversion<TDst, TSrc>)
inline TDst convert(TSrc src, OptValuedScalarTag<TSrc> auto /*tag*/, BoolTag<tSafe> /*tag*/) {
  return TDst(src);
}
template<Vectorizable TDst, AnyVector TSrc, bool tSafe>
requires(!tSafe || SafeConversion<TDst, typename TSrc::Value>)
inline Vector<TDst, TSrc::size> convert(TSrc src, OptTypedVectorTag<TSrc> auto /*tag*/,
                                        BoolTag<tSafe> /*tag*/) {
  return src.convert(type_tag<TDst>);
}
template<Vectorizable TDst, AnyMask TSrc, bool tSafe>
requires(!tSafe || SafeConversion<TDst, typename TSrc::VecValue>)
inline Mask<TDst, TSrc::size> convert(TSrc src, OptTypedVectorTag<TSrc> auto /*tag*/,
                                      BoolTag<tSafe> /*tag*/) {
  return src.convert(type_tag<TDst>);
}
template<Vectorizable TDst, typename TSrc>
inline TDst convert_unsafe(TSrc src, AnyTag auto tag) {
  return convert(src, tag, false_tag);
}
template<Vectorizable TDst, typename TSrc>
inline TDst convert_safe(TSrc src, AnyTag auto tag) {
  return convert(src, tag, true_tag);
}

// gather
template<Vectorizable T, std::size_t tExtent>
inline T gather(std::span<const T, tExtent> data, IntVectorizable auto idx,
                OptValuedScalarTag<T> auto /*tag*/) {
  return data[idx];
}
template<Vectorizable T, std::size_t tExtent, OptTypedFullVectorTag<T> TTag>
inline Vector<T, TTag::size> gather(std::span<const T, tExtent> data, IntVector auto idxs,
                                    TTag /*tag*/) {
  return gather(data, idxs);
}
template<Vectorizable T, std::size_t tExtent, std::size_t tSize>
inline Vector<T, tSize> gather(std::span<const T, tExtent> data, IntVector auto idxs,
                               OptValuedPartialVectorTag<T> auto tag) {
  return mask_gather(data, tag.mask(), idxs);
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

// zero
template<Vectorizable T, OptValuedTag<T> TTag>
inline TagType<TTag, T> zero(TTag /*tag*/) {
  return TagType<TTag, T>{};
}

// constant
template<Vectorizable T, OptValuedTag<T> TTag>
inline TagType<TTag, T> constant(T value, TTag /*tag*/) {
  return TagType<TTag, T>{value};
}

// abs
template<Vectorizable T>
inline T abs(T value, OptValuedScalarTag<T> auto /*tag*/) {
  return std::abs(value);
}
template<AnyVector TVec>
inline TVec abs(TVec value, OptTypedVectorTag<TVec> auto /*tag*/) {
  return grex::abs(value);
}

// max
template<Vectorizable T>
inline T max(T v1, T v2, OptValuedScalarTag<T> auto /*tag*/) {
  return std::max(v1, v2);
}
template<AnyVector TVec>
inline TVec max(TVec v1, TVec v2, OptTypedVectorTag<TVec> auto /*tag*/) {
  return max(v1, v2);
}

#define GREX_OPS_TERNARY(NAME) \
  template<Vectorizable T> \
  inline T NAME(T a, T b, T c, OptValuedScalarTag<T> auto /*tag*/) { \
    return NAME(a, b, c); \
  } \
  template<AnyVector TVec, OptTypedVectorTag<TVec> TTag> \
  inline TVec NAME(TVec a, TVec b, TVec c, TTag /*tag*/) { \
    return NAME(a, b, c); \
  }
GREX_OPS_TERNARY(fmadd)
GREX_OPS_TERNARY(fmsub)
GREX_OPS_TERNARY(fnmadd)
GREX_OPS_TERNARY(fnmsub)
#undef GREX_OPS_TERNARY

#define GREX_OPS_MASKARITH(NAME, OP) \
  template<Vectorizable T> \
  inline T NAME(bool mask, T a, T b, OptValuedScalarTag<T> auto /*tag*/) { \
    return mask ? (a OP b) : a; \
  } \
  template<AnyVector TVec> \
  inline TVec NAME(MaskFor<TVec> mask, TVec a, TVec b, OptTypedVectorTag<TVec> auto /*tag*/) { \
    return NAME(mask, a, b); \
  }
GREX_OPS_MASKARITH(mask_add, +)
GREX_OPS_MASKARITH(mask_subtract, +)
GREX_OPS_MASKARITH(mask_multiply, +)
GREX_OPS_MASKARITH(mask_divide, +)

// shingle_up
template<Vectorizable T>
inline T shingle_up(T /*base*/, OptValuedScalarTag<T> auto /*tag*/) {
  return T{};
}
template<AnyVector TVec>
inline TVec shingle_up(TVec base, OptTypedVectorTag<TVec> auto /*tag*/) {
  return base.shingle_up();
}
template<Vectorizable T>
inline T shingle_up(T front, T /*base*/, OptValuedScalarTag<T> auto /*tag*/) {
  return front;
}
template<AnyVector TVec>
inline TVec shingle_up(typename TVec::Value front, TVec base,
                       OptTypedVectorTag<TVec> auto /*tag*/) {
  return base.shingle_up(front);
}

// shingle_down
template<Vectorizable T>
inline T shingle_down(T /*base*/, OptValuedScalarTag<T> auto /*tag*/) {
  return T{};
}
template<AnyVector TVec>
inline TVec shingle_down(TVec base, OptTypedVectorTag<TVec> auto /*tag*/) {
  return base.shingle_down();
}
template<Vectorizable T>
inline T shingle_down(T /*base*/, T back, OptValuedScalarTag<T> auto /*tag*/) {
  return back;
}
template<AnyVector TVec>
inline TVec shingle_down(TVec base, typename TVec::Value back,
                         OptTypedVectorTag<TVec> auto /*tag*/) {
  return base.shingle_down(back);
}

// is_finite
template<Vectorizable T>
inline bool is_finite(T v, OptValuedScalarTag<T> auto /*tag*/) {
  return std::isfinite(v);
}
template<AnyVector TVec>
inline MaskFor<TVec> is_finite(TVec vec, OptTypedVectorTag<TVec> auto /*tag*/) {
  return is_finite(vec);
}

// horizontal_add
template<Vectorizable T>
inline T horizontal_add(T value, OptValuedScalarTag<T> auto /*tag*/) {
  return value;
}
template<AnyVector TVec>
inline TVec::Value horizontal_add(TVec value, OptTypedVectorTag<TVec> auto tag) {
  return horizontal_add(tag.mask(value));
}

// horizontal_max
template<Vectorizable T>
inline T horizontal_max(T value, OptValuedScalarTag<T> auto /*tag*/) {
  return value;
}
template<AnyVector TVec, std::size_t tSize>
inline TVec::Value horizontal_max(TVec value, OptTypedFullVectorTag<TVec> auto /*tag*/) {
  return horizontal_max(value);
}
// TODO Partial maxima are problematic if there mask is empty: What should the placeholder be?

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
  return horizontal_and(mask | ~tag.mask(type_tag<typename TMask::VecValue>));
}
} // namespace grex

#endif // INCLUDE_GREX_OPERATIONS_HPP
