// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_NEON_OPERATIONS_CONVERT_HPP
#define INCLUDE_GREX_BACKEND_NEON_OPERATIONS_CONVERT_HPP

#include <cstddef>

#include <arm_neon.h>

#include "grex/backend/choosers.hpp"
#include "grex/backend/defs.hpp"
#include "grex/backend/macros/base.hpp"
#include "grex/backend/neon/macros/types.hpp"
#include "grex/backend/neon/operations/expand.hpp"
#include "grex/backend/neon/operations/merge.hpp"
#include "grex/backend/neon/operations/reinterpret.hpp"
#include "grex/backend/neon/operations/split.hpp"
#include "grex/base.hpp"

namespace grex::backend {
#define GREX_CVT_VCVTQ(DSTKIND, DSTBITS, SRCKIND, SRCBITS, SIZE) \
  return {.r = GREX_CAT(vcvtq_, GREX_ISUFFIX(DSTKIND, DSTBITS), _, \
                        GREX_ISUFFIX(SRCKIND, SRCBITS))(v.registr())};
#define GREX_CVT_WIDEN(DSTKIND, DSTBITS, SRCKIND, SRCBITS, SIZE) \
  const auto low = GREX_ISUFFIXED(vget_low, SRCKIND, SRCBITS)(v.registr()); \
  return {.r = GREX_ISUFFIXED(vmovl, SRCKIND, SRCBITS)(low)};
#define GREX_CVT_NARROW(DSTKIND, DSTBITS, SRCKIND, SRCBITS, SIZE) \
  const auto lo = GREX_ISUFFIXED(vmovn, SRCKIND, SRCBITS)(v.r); \
  const auto combined = expand64(lo); \
  return VectorFor<DSTKIND##DSTBITS, SIZE>{combined};
#define GREX_CVT_REINTERPRET(DSTKIND, DSTBITS, SRCKIND, SRCBITS, SIZE) \
  return as<DSTKIND##DSTBITS>(v);
#define GREX_CVT_f64_f32(...) return {.r = vcvt_f64_f32(vget_low_f32(v.full.r))};
#define GREX_CVT_f32_f64(...) return VectorFor<f32, 2>{expand64(vcvt_f32_f64(v.r))};

#define GREX_CVT(DSTKIND, DSTBITS, SRCKIND, SRCBITS, SIZE, INTRINSIC, ...) \
  inline VectorFor<DSTKIND##DSTBITS, SIZE> convert(VectorFor<SRCKIND##SRCBITS, SIZE> v, \
                                                   TypeTag<DSTKIND##DSTBITS>) { \
    INTRINSIC(DSTKIND, DSTBITS, SRCKIND, SRCBITS, SIZE __VA_OPT__(, ) __VA_ARGS__) \
  }

// f64
GREX_CVT(f, 64, i, 64, 2, GREX_CVT_VCVTQ)
GREX_CVT(f, 64, u, 64, 2, GREX_CVT_VCVTQ)
GREX_CVT(f, 64, f, 32, 2, GREX_CVT_f64_f32)
// f32
GREX_CVT(f, 32, f, 64, 2, GREX_CVT_f32_f64)
GREX_CVT(f, 32, i, 32, 4, GREX_CVT_VCVTQ)
GREX_CVT(f, 32, u, 32, 4, GREX_CVT_VCVTQ)
// i64
GREX_CVT(i, 64, f, 64, 2, GREX_CVT_VCVTQ)
GREX_CVT(i, 64, u, 64, 2, GREX_CVT_REINTERPRET)
GREX_CVT(i, 64, i, 32, 2, GREX_CVT_WIDEN)
// u64
GREX_CVT(u, 64, f, 64, 2, GREX_CVT_VCVTQ)
GREX_CVT(u, 64, i, 64, 2, GREX_CVT_REINTERPRET)
GREX_CVT(u, 64, u, 32, 2, GREX_CVT_WIDEN)
// i32
GREX_CVT(i, 32, i, 64, 2, GREX_CVT_NARROW)
GREX_CVT(i, 32, f, 32, 4, GREX_CVT_VCVTQ)
GREX_CVT(i, 32, u, 32, 4, GREX_CVT_REINTERPRET)
GREX_CVT(i, 32, i, 16, 4, GREX_CVT_WIDEN)
// i32
GREX_CVT(u, 32, u, 64, 2, GREX_CVT_NARROW)
GREX_CVT(u, 32, f, 32, 4, GREX_CVT_VCVTQ)
GREX_CVT(u, 32, i, 32, 4, GREX_CVT_REINTERPRET)
GREX_CVT(u, 32, u, 16, 4, GREX_CVT_WIDEN)
// i16
GREX_CVT(i, 16, i, 32, 4, GREX_CVT_NARROW)
GREX_CVT(i, 16, u, 16, 8, GREX_CVT_REINTERPRET)
GREX_CVT(i, 16, i, 8, 8, GREX_CVT_WIDEN)
// i16
GREX_CVT(u, 16, u, 32, 4, GREX_CVT_NARROW)
GREX_CVT(u, 16, i, 16, 8, GREX_CVT_REINTERPRET)
GREX_CVT(u, 16, u, 8, 8, GREX_CVT_WIDEN)
// i8
GREX_CVT(i, 8, i, 16, 8, GREX_CVT_NARROW)
GREX_CVT(i, 8, u, 8, 16, GREX_CVT_REINTERPRET)
// i8
GREX_CVT(u, 8, u, 16, 8, GREX_CVT_NARROW)
GREX_CVT(u, 8, i, 8, 16, GREX_CVT_REINTERPRET)

// Source and destination type identical: no-op
template<Vectorizable T, std::size_t tSize>
inline Vector<T, tSize> convert(Vector<T, tSize> v, TypeTag<T> /*tag*/) {
  return v;
}
// sub-native → sub-native: Make the smaller one native
template<Vectorizable TDst, Vectorizable TSrc, std::size_t tPart, std::size_t tSize>
requires(is_subnative<TDst, tPart>)
inline VectorFor<TDst, tPart> convert(SubVector<TSrc, tPart, tSize> v, TypeTag<TDst> tag) {
  constexpr std::size_t work_size = std::min(min_native_size<TDst>, min_native_size<TSrc>);
  return VectorFor<TDst, tPart>{convert(VectorFor<TSrc, work_size>(v.full.r), tag).registr()};
}
// from super-native vector: Work on the halves separately
// TODO This might introduce unnecessary work when converting to sub-native
template<typename THalf, Vectorizable TDst>
inline VectorFor<TDst, THalf::size * 2> convert(SuperVector<THalf> v, TypeTag<TDst> /*tag*/) {
  return merge(convert(v.lower, type_tag<TDst>), convert(v.upper, type_tag<TDst>));
}
// native → super-native: Split native
template<Vectorizable TDst, Vectorizable TSrc, std::size_t tSize>
requires(is_supernative<TDst, tSize>)
inline VectorFor<TDst, tSize> convert(Vector<TSrc, tSize> v, TypeTag<TDst> tag) {
  return merge(convert(get_low(v), tag), convert(get_high(v), tag));
}

// integer → smaller integer: Convert to half-size and go on from there
template<IntVectorizable TDst, IntVectorizable TSrc, std::size_t tSize>
requires(sizeof(TDst) < sizeof(TSrc))
inline VectorFor<TDst, tSize> convert(Vector<TSrc, tSize> v, TypeTag<TDst> tag) {
  return convert(convert(v, type_tag<CopySignInt<TSrc, sizeof(TSrc) / 2>>), tag);
}
// integer → bigger integer: Convert to double-size while retaining signedness and go on from there
template<IntVectorizable TDst, IntVectorizable TSrc, std::size_t tPart, std::size_t tSize>
requires(sizeof(TSrc) < sizeof(TDst) && !is_subnative<TDst, tPart>)
inline VectorFor<TDst, tPart> convert(SubVector<TSrc, tPart, tSize> v, TypeTag<TDst> tag) {
  using Inter = CopySignInt<TSrc, sizeof(TSrc) * 2>;
  constexpr std::size_t inter_size = min_native_size<Inter>;
  static_assert(inter_size >= tPart);

  const auto inter = convert(VectorFor<TSrc, inter_size>{v.registr()}, type_tag<Inter>);
  return convert(VectorFor<Inter, tPart>{inter.registr()}, tag);
}

// integer → bigger floating-point: Convert to destination-sized integer and then cast
template<FloatVectorizable TDst, IntVectorizable TSrc, std::size_t tPart, std::size_t tSize>
requires(sizeof(TSrc) < sizeof(TDst) && !is_subnative<TDst, tPart>)
inline VectorFor<TDst, tPart> convert(SubVector<TSrc, tPart, tSize> v, TypeTag<TDst> tag) {
  return convert(convert(v, type_tag<CopySignInt<TSrc, sizeof(TDst)>>), tag);
}
// floating-point → bigger integer: Convert to destination-sized floating-point and then cast
template<IntVectorizable TDst, FloatVectorizable TSrc, std::size_t tPart, std::size_t tSize>
requires(sizeof(TSrc) < sizeof(TDst))
inline VectorFor<TDst, tPart> convert(SubVector<TSrc, tPart, tSize> v, TypeTag<TDst> tag) {
  return convert(convert(v, type_tag<Float<sizeof(TDst)>>), tag);
}
// integer → smaller floating-point: Convert to source-sized floating-point and then cast
template<FloatVectorizable TDst, IntVectorizable TSrc, std::size_t tSize>
requires(sizeof(TDst) < sizeof(TSrc))
inline VectorFor<TDst, tSize> convert(Vector<TSrc, tSize> v, TypeTag<TDst> tag) {
  return convert(convert(v, type_tag<Float<sizeof(TSrc)>>), tag);
}
// floating-point → smaller integer: Convert to destination-sized integer and then cast
template<IntVectorizable TDst, FloatVectorizable TSrc, std::size_t tSize>
requires(sizeof(TDst) < sizeof(TSrc))
inline VectorFor<TDst, tSize> convert(Vector<TSrc, tSize> v, TypeTag<TDst> tag) {
  return convert(convert(v, type_tag<CopySignInt<TDst, sizeof(TSrc)>>), tag);
}
} // namespace grex::backend

namespace grex::backend {
template<AnyMask TMask, typename TDst>
inline auto convert(TMask mask, TypeTag<TDst> /*tag*/) {
  return vector2mask(convert(mask2vector(mask), type_tag<SignedInt<sizeof(TDst)>>), type_tag<TDst>);
}

template<Vectorizable TDst, typename THalf>
inline MaskFor<TDst, 2 * THalf::size> convert(SuperMask<THalf> m, TypeTag<TDst> tag) {
  return merge(convert(m.lower, tag), convert(m.upper, tag));
}

template<Vectorizable TDst, AnyVector TSrc>
inline VectorFor<TDst, TSrc::size> convert(TSrc v) {
  return convert(v, type_tag<TDst>);
}
template<Vectorizable TDst, AnyMask TSrc>
inline MaskFor<TDst, TSrc::size> convert(TSrc v) {
  return convert(v, type_tag<TDst>);
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_CONVERT_HPP
