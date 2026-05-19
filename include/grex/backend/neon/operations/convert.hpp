// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_NEON_OPERATIONS_CONVERT_HPP
#define INCLUDE_GREX_BACKEND_NEON_OPERATIONS_CONVERT_HPP

#include <cstddef>

#include <arm_neon.h>

#include "grex/backend/base.hpp"
#include "grex/backend/choosers.hpp"
#include "grex/backend/defs.hpp" // IWYU pragma: keep
#include "grex/backend/macros/base.hpp"
#include "grex/backend/neon/macros/types.hpp"
#include "grex/backend/neon/operations/expand.hpp"
#include "grex/backend/neon/operations/mask-convert.hpp"
#include "grex/backend/neon/operations/reinterpret.hpp"
#include "grex/base.hpp"

namespace grex::backend {
#define GREX_CVT_VCVTQ(DSTKIND, DSTBITS, SRCKIND, SRCBITS, SIZE) \
  return {.r = GREX_CAT(vcvtq_, GREX_ISUFFIX(DSTKIND, DSTBITS), _, \
                        GREX_ISUFFIX(SRCKIND, SRCBITS))(v.registr())};
#define GREX_CVT_MOVL(DSTKIND, DSTBITS, SRCKIND, SRCBITS, SIZE) \
  const auto low = GREX_ISUFFIXED(vget_low, SRCKIND, SRCBITS)(v.registr()); \
  return {.r = GREX_ISUFFIXED(vmovl, SRCKIND, SRCBITS)(low)};
#define GREX_CVT_MOVL2(DSTKIND, DSTBITS, SRCKIND, SRCBITS, SIZE) \
  const auto low = GREX_ISUFFIXED(vget_low, SRCKIND, SRCBITS)(v.registr()); \
  return {.lower = {.r = GREX_ISUFFIXED(vmovl, SRCKIND, SRCBITS)(low)}, \
          .upper = {.r = GREX_ISUFFIXED(vmovl_high, SRCKIND, SRCBITS)(v.registr())}};
#define GREX_CVT_MOVN(DSTKIND, DSTBITS, SRCKIND, SRCBITS, SIZE) \
  const auto lo = GREX_ISUFFIXED(vmovn, SRCKIND, SRCBITS)(v.r); \
  const auto combined = expand64(lo); \
  return VectorFor<DSTKIND##DSTBITS, SIZE>{combined};
#define GREX_CVT_UZP1(DSTKIND, DSTBITS, SRCKIND, SRCBITS, SIZE) \
  const auto a = as<DSTKIND##DSTBITS>(v.lower.r); \
  const auto b = as<DSTKIND##DSTBITS>(v.upper.r); \
  return {.r = GREX_ISUFFIXED(vuzp1q, DSTKIND, DSTBITS)(a, b)};
#define GREX_CVT_f64_f32x2(...) return {.r = vcvt_f64_f32(vget_low_f32(v.full.r))};
#define GREX_CVT_f64_f32x4(...) \
  return {.lower = {.r = vcvt_f64_f32(vget_low_f32(v.r))}, .upper = {.r = vcvt_high_f64_f32(v.r)}};
#define GREX_CVT_f32_f64x2(...) return VectorFor<f32, 2>{expand64(vcvt_f32_f64(v.r))};
#define GREX_CVT_f32_f64x4(...) return {.r = vcvt_high_f32_f64(vcvt_f32_f64(v.lower.r), v.upper.r)};

// Backend entry point for Neon conversions.
// The INTRINSIC macro encodes the implementation strategy (single intrinsic, widening/narrowing
// via movl/movn, structure-of-halves expansion, etc.).
#define GREX_CVT(DSTKIND, DSTBITS, SRCKIND, SRCBITS, SIZE, INTRINSIC, ...) \
  inline VectorFor<DSTKIND##DSTBITS, SIZE> convert(VectorFor<SRCKIND##SRCBITS, SIZE> v, \
                                                   TypeTag<DSTKIND##DSTBITS>) { \
    INTRINSIC(DSTKIND, DSTBITS, SRCKIND, SRCBITS, SIZE __VA_OPT__(, ) __VA_ARGS__) \
  }

// f64
GREX_CVT(f, 64, i, 64, 2, GREX_CVT_VCVTQ)
GREX_CVT(f, 64, u, 64, 2, GREX_CVT_VCVTQ)
GREX_CVT(f, 64, f, 32, 2, GREX_CVT_f64_f32x2)
GREX_CVT(f, 64, f, 32, 4, GREX_CVT_f64_f32x4)
// f32
GREX_CVT(f, 32, f, 64, 2, GREX_CVT_f32_f64x2)
GREX_CVT(f, 32, f, 64, 4, GREX_CVT_f32_f64x4)
GREX_CVT(f, 32, i, 32, 4, GREX_CVT_VCVTQ)
GREX_CVT(f, 32, u, 32, 4, GREX_CVT_VCVTQ)
// i64
GREX_CVT(i, 64, f, 64, 2, GREX_CVT_VCVTQ)
GREX_CVT(i, 64, i, 32, 2, GREX_CVT_MOVL)
GREX_CVT(i, 64, i, 32, 4, GREX_CVT_MOVL2)
// u64
GREX_CVT(u, 64, f, 64, 2, GREX_CVT_VCVTQ)
GREX_CVT(u, 64, u, 32, 2, GREX_CVT_MOVL)
GREX_CVT(u, 64, u, 32, 4, GREX_CVT_MOVL2)
// i32
GREX_CVT(i, 32, i, 64, 4, GREX_CVT_UZP1)
GREX_CVT(i, 32, i, 64, 2, GREX_CVT_MOVN)
GREX_CVT(i, 32, f, 32, 4, GREX_CVT_VCVTQ)
GREX_CVT(i, 32, i, 16, 4, GREX_CVT_MOVL)
GREX_CVT(i, 32, i, 16, 8, GREX_CVT_MOVL2)
// u32
GREX_CVT(u, 32, u, 64, 4, GREX_CVT_UZP1)
GREX_CVT(u, 32, u, 64, 2, GREX_CVT_MOVN)
GREX_CVT(u, 32, f, 32, 4, GREX_CVT_VCVTQ)
GREX_CVT(u, 32, u, 16, 4, GREX_CVT_MOVL)
GREX_CVT(u, 32, u, 16, 8, GREX_CVT_MOVL2)
// i16
GREX_CVT(i, 16, i, 32, 8, GREX_CVT_UZP1)
GREX_CVT(i, 16, i, 32, 4, GREX_CVT_MOVN)
GREX_CVT(i, 16, i, 8, 8, GREX_CVT_MOVL)
GREX_CVT(i, 16, i, 8, 16, GREX_CVT_MOVL2)
// u16
GREX_CVT(u, 16, u, 32, 8, GREX_CVT_UZP1)
GREX_CVT(u, 16, u, 32, 4, GREX_CVT_MOVN)
GREX_CVT(u, 16, u, 8, 8, GREX_CVT_MOVL)
GREX_CVT(u, 16, u, 8, 16, GREX_CVT_MOVL2)
// i8
GREX_CVT(i, 8, i, 16, 16, GREX_CVT_UZP1)
GREX_CVT(i, 8, i, 16, 8, GREX_CVT_MOVN)
// u8
GREX_CVT(u, 8, u, 16, 16, GREX_CVT_UZP1)
GREX_CVT(u, 8, u, 16, 8, GREX_CVT_MOVN)

// Integer → smaller integer (factor other than two):
// narrow to the next smaller integer type with preserved signedness, then recurse.
template<IntVectorizable TDst, IntVector TSrc>
requires(sizeof(TDst) < sizeof(ValueOf<TSrc>))
inline VectorFor<TDst, size_of<TSrc>> convert(TSrc v, TypeTag<TDst> tag) {
  return convert(convert(v, type_tag<CopySignInt<ValueOf<TSrc>, sizeof(ValueOf<TSrc>) / 2>>), tag);
}

// Integer → larger integer (factor other than two):
// widen to the next larger integer type with preserved signedness, then recurse.
template<IntVectorizable TDst, IntVector TSrc>
requires(sizeof(ValueOf<TSrc>) < sizeof(TDst))
inline VectorFor<TDst, size_of<TSrc>> convert(TSrc v, TypeTag<TDst> tag) {
  return convert(convert(v, type_tag<CopySignInt<ValueOf<TSrc>, sizeof(ValueOf<TSrc>) * 2>>), tag);
}

// Integer → larger floating-point:
// first convert to an integer with the destination element size, then cast to floating-point.
template<FloatVectorizable TDst, IntVector TSrc>
requires(sizeof(ValueOf<TSrc>) < sizeof(TDst))
inline VectorFor<TDst, size_of<TSrc>> convert(TSrc v, TypeTag<TDst> tag) {
  return convert(convert(v, type_tag<CopySignInt<ValueOf<TSrc>, sizeof(TDst)>>), tag);
}

// Floating-point → larger integer:
// first convert to a floating-point type with the destination element size, then cast to integer.
template<IntVectorizable TDst, FloatVector TSrc>
requires(sizeof(ValueOf<TSrc>) < sizeof(TDst))
inline VectorFor<TDst, size_of<TSrc>> convert(TSrc v, TypeTag<TDst> tag) {
  return convert(convert(v, type_tag<Float<sizeof(TDst)>>), tag);
}

// Integer → smaller floating-point:
// first convert to a floating-point type matching the source element size, then cast down.
template<FloatVectorizable TDst, IntVector TSrc>
requires(sizeof(TDst) < sizeof(ValueOf<TSrc>))
inline VectorFor<TDst, size_of<TSrc>> convert(TSrc v, TypeTag<TDst> tag) {
  return convert(convert(v, type_tag<Float<sizeof(ValueOf<TSrc>)>>), tag);
}

// Floating-point → smaller integer:
// first convert to an integer type matching the destination element size, then cast down.
template<IntVectorizable TDst, FloatVector TSrc>
requires(sizeof(TDst) < sizeof(ValueOf<TSrc>))
inline VectorFor<TDst, size_of<TSrc>> convert(TSrc v, TypeTag<TDst> tag) {
  return convert(convert(v, type_tag<CopySignInt<TDst, sizeof(ValueOf<TSrc>)>>), tag);
}

// Mask → mask: convert via the corresponding signed integer vector.
template<AnyMask TMask, typename TDst>
inline auto convert(TMask mask, TypeTag<TDst> /*tag*/) {
  return vector2mask(convert(mask2vector(mask), type_tag<SignedInt<sizeof(TDst)>>), type_tag<TDst>);
}

// Super-mask → mask: convert both halves and merge.
template<Vectorizable TDst, typename THalf>
inline MaskFor<TDst, 2 * THalf::size> convert(SuperMask<THalf> m, TypeTag<TDst> tag) {
  return merge(convert(m.lower, tag), convert(m.upper, tag));
}

// Convenience functions taking the destination element type as a template parameter, not a tag.
template<Vectorizable TDst, AnyVector TSrc>
inline VectorFor<TDst, TSrc::size> convert(TSrc v) {
  return convert(v, type_tag<TDst>);
}
template<Vectorizable TDst, AnyMask TSrc>
inline MaskFor<TDst, TSrc::size> convert(TSrc v) {
  return convert(v, type_tag<TDst>);
}
} // namespace grex::backend

#include "grex/backend/shared/operations/convert.hpp" // IWYU pragma: export

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_CONVERT_HPP
