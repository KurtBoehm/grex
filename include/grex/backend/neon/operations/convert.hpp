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
#include "grex/backend/neon/operations/expand64.hpp"
#include "grex/backend/neon/operations/f16.hpp"
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

// Binary16 ↔ binary32 is always available, binary16 ↔ `i16`/`u16` only with FP16.
// These are then used as starting points for further conversions (if necessary).
#define GREX_CVT_f32_f16x4(...) return {.r = vcvt_f32_f16(vget_low_f16(as_f16(v.registr())))};
#define GREX_CVT_f32_f16x8(...) \
  return { \
    .lower = {.r = vcvt_f32_f16(vget_low_f16(as_f16(v.registr())))}, \
    .upper = {.r = vcvt_high_f32_f16(as_f16(v.registr()))}, \
  };
#define GREX_CVT_f16_f32x4(...) return VectorFor<f16, 4>{expand64(as_u16(vcvt_f16_f32(v.r)))};
#define GREX_CVT_f16_f32x8(...) \
  return {.r = as_u16(vcvt_high_f16_f32(vcvt_f16_f32(v.lower.r), v.upper.r))};

#if GREX_F16_NATIVE_ARITHMETIC
#define GREX_CVT_i16_f16x8(...) return {.r = vcvtq_s16_f16(as_f16(v.r))};
#define GREX_CVT_u16_f16x8(...) return {.r = vcvtq_u16_f16(as_f16(v.r))};
#define GREX_CVT_f16_i16x8(...) return {.r = as_u16(vcvtq_f16_s16(v.r))};
#define GREX_CVT_f16_u16x8(...) return {.r = as_u16(vcvtq_f16_u16(v.r))};
#endif

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
GREX_CVT(f, 32, f, 16, 4, GREX_CVT_f32_f16x4)
GREX_CVT(f, 32, f, 16, 8, GREX_CVT_f32_f16x8)
// f16
GREX_CVT(f, 16, f, 32, 4, GREX_CVT_f16_f32x4)
GREX_CVT(f, 16, f, 32, 8, GREX_CVT_f16_f32x8)
#if GREX_F16_NATIVE_ARITHMETIC
GREX_CVT(f, 16, i, 16, 8, GREX_CVT_f16_i16x8)
GREX_CVT(f, 16, u, 16, 8, GREX_CVT_f16_u16x8)
#endif
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
#if GREX_F16_NATIVE_ARITHMETIC
GREX_CVT(i, 16, f, 16, 8, GREX_CVT_i16_f16x8)
#endif
GREX_CVT(i, 16, i, 8, 8, GREX_CVT_MOVL)
GREX_CVT(i, 16, i, 8, 16, GREX_CVT_MOVL2)
// u16
GREX_CVT(u, 16, u, 32, 8, GREX_CVT_UZP1)
GREX_CVT(u, 16, u, 32, 4, GREX_CVT_MOVN)
#if GREX_F16_NATIVE_ARITHMETIC
GREX_CVT(u, 16, f, 16, 8, GREX_CVT_u16_f16x8)
#endif
GREX_CVT(u, 16, u, 8, 8, GREX_CVT_MOVL)
GREX_CVT(u, 16, u, 8, 16, GREX_CVT_MOVL2)
// i8
GREX_CVT(i, 8, i, 16, 16, GREX_CVT_UZP1)
GREX_CVT(i, 8, i, 16, 8, GREX_CVT_MOVN)
// u8
GREX_CVT(u, 8, u, 16, 16, GREX_CVT_UZP1)
GREX_CVT(u, 8, u, 16, 8, GREX_CVT_MOVN)

// Binary16 ↔ binary64: `f16_to_f64`/`f64_to_f16` also pass through binary32, but round only once.
template<Float16Vector Src>
inline VectorFor<f64, size_of<Src>> convert(Src v, TypeTag<f64> /*tag*/) {
  return f16_to_f64(v);
}
template<TypedVector<f64> Src>
inline VectorFor<f16, size_of<Src>> convert(Src v, TypeTag<f16> /*tag*/) {
  return f64_to_f16(v);
}

#if !GREX_F16_NATIVE_ARITHMETIC
// Integer → binary16 fallback: convert to binary32 and go from there.
template<Float16 Dst, IntVector Src>
inline VectorFor<Dst, size_of<Src>> convert(Src v, TypeTag<Dst> tag) {
  return convert(convert(v, type_tag<f32>), tag);
}
// Binary16 → integer fallback: convert to binary32 and go from there.
template<IntVectorizable Dst, Float16Vector Src>
inline VectorFor<Dst, size_of<Src>> convert(Src v, TypeTag<Dst> tag) {
  return convert(convert(v, type_tag<f32>), tag);
}
#endif

// Integer → smaller integer (factor other than two):
// narrow to the next smaller integer type with preserved signedness, then recurse.
template<IntVectorizable Dst, IntVector Src>
requires(sizeof(Dst) < sizeof(ValueOf<Src>))
inline VectorFor<Dst, size_of<Src>> convert(Src v, TypeTag<Dst> tag) {
  return convert(convert(v, type_tag<CopySignInt<ValueOf<Src>, sizeof(ValueOf<Src>) / 2>>), tag);
}

// Integer → larger integer (factor other than two):
// widen to the next larger integer type with preserved signedness, then recurse.
template<IntVectorizable Dst, IntVector Src>
requires(sizeof(ValueOf<Src>) < sizeof(Dst))
inline VectorFor<Dst, size_of<Src>> convert(Src v, TypeTag<Dst> tag) {
  return convert(convert(v, type_tag<CopySignInt<ValueOf<Src>, sizeof(ValueOf<Src>) * 2>>), tag);
}

// Integer → larger floating-point:
// first convert to an integer with the destination element size, then cast to floating-point.
template<NativeFloatVectorizable Dst, IntVector Src>
requires(sizeof(ValueOf<Src>) < sizeof(Dst))
inline VectorFor<Dst, size_of<Src>> convert(Src v, TypeTag<Dst> tag) {
  return convert(convert(v, type_tag<CopySignInt<ValueOf<Src>, sizeof(Dst)>>), tag);
}

// Floating-point → larger integer:
// first convert to a floating-point type with the destination element size, then cast to integer.
template<IntVectorizable Dst, NativeFloatVector Src>
requires(sizeof(ValueOf<Src>) < sizeof(Dst))
inline VectorFor<Dst, size_of<Src>> convert(Src v, TypeTag<Dst> tag) {
  return convert(convert(v, type_tag<Float<sizeof(Dst)>>), tag);
}

// Integer → smaller floating-point:
// first convert to a floating-point type matching the source element size, then cast down.
template<NativeFloatVectorizable Dst, IntVector Src>
requires(sizeof(Dst) < sizeof(ValueOf<Src>))
inline VectorFor<Dst, size_of<Src>> convert(Src v, TypeTag<Dst> tag) {
  return convert(convert(v, type_tag<Float<sizeof(ValueOf<Src>)>>), tag);
}

// Floating-point → smaller integer:
// first convert to an integer type matching the destination element size, then cast down.
template<IntVectorizable Dst, NativeFloatVector Src>
requires(sizeof(Dst) < sizeof(ValueOf<Src>))
inline VectorFor<Dst, size_of<Src>> convert(Src v, TypeTag<Dst> tag) {
  return convert(convert(v, type_tag<CopySignInt<Dst, sizeof(ValueOf<Src>)>>), tag);
}

// Mask → mask: convert via the corresponding signed integer vector.
template<AnyMask Mask, typename Dst>
inline auto convert(Mask mask, TypeTag<Dst> /*tag*/) {
  return vector2mask(convert(mask2vector(mask), type_tag<SignedInt<sizeof(Dst)>>), type_tag<Dst>);
}

// Super-mask → mask: convert both halves and merge.
template<Vectorizable Dst, typename Half>
inline MaskFor<Dst, 2 * Half::size> convert(SuperMask<Half> m, TypeTag<Dst> tag) {
  return merge(convert(m.lower, tag), convert(m.upper, tag));
}
} // namespace grex::backend

#include "grex/backend/shared/operations/convert.hpp" // IWYU pragma: export

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_CONVERT_HPP
