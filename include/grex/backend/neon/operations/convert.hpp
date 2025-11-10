// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_NEON_OPERATIONS_CONVERT_HPP
#define INCLUDE_GREX_BACKEND_NEON_OPERATIONS_CONVERT_HPP

#include <arm_neon.h>

#include "grex/backend/choosers.hpp"
#include "grex/backend/macros/base.hpp"
#include "grex/backend/neon/operations/set.hpp"
#include "grex/base/defs.hpp"

namespace grex::backend {
#define GREX_CVT_VCVTQ(DSTKIND, DSTBITS, SRCKIND, SRCBITS, SIZE) \
  return {.r = GREX_CAT(vcvtq_, GREX_ISUFFIX(DSTKIND, DSTBITS), _, \
                        GREX_ISUFFIX(SRCKIND, SRCBITS))(v.registr())};
#define GREX_CVT_VCVTQ_LOW(DSTKIND, DSTBITS, SRCKIND, SRCBITS, SIZE) \
  return {.r = GREX_CAT(vcvt_, GREX_ISUFFIX(DSTKIND, DSTBITS), _, GREX_ISUFFIX(SRCKIND, SRCBITS))( \
            GREX_CAT(vget_low_, GREX_ISUFFIX(SRCKIND, SRCBITS))(v.registr()))};
#define GREX_CVT_INTx2(DSTKIND, DSTBITS, SRCKIND, SRCBITS, SIZE) \
  const auto low = GREX_CAT(vget_low_, GREX_ISUFFIX(SRCKIND, SRCBITS))(v.registr()); \
  return {.r = GREX_CAT(vmovl_, GREX_ISUFFIX(SRCKIND, SRCBITS))(low)};
#define GREX_CVT_f32_f64(...) \
  return VectorFor<f32, 2>{vcombine_f32(vcvt_f32_f64(v.r), make_undefined<float32x2_t>())};

#define GREX_CVT_f32_i64(...) return convert(convert(v, type_tag<f64>), type_tag<f32>);
#define GREX_CVT_x64_f32(DSTKIND, DSTBITS, ...) \
  return {convert(convert(v, type_tag<f64>), type_tag<DSTKIND##DSTBITS>)};

#define GREX_CVT(DSTKIND, DSTBITS, SRCKIND, SRCBITS, SIZE, INTRINSIC, ...) \
  inline VectorFor<DSTKIND##DSTBITS, SIZE> convert(VectorFor<SRCKIND##SRCBITS, SIZE> v, \
                                                   TypeTag<DSTKIND##DSTBITS>) { \
    INTRINSIC(DSTKIND, DSTBITS, SRCKIND, SRCBITS, SIZE __VA_OPT__(, ) __VA_ARGS__) \
  }

// f64
GREX_CVT(f, 64, i, 64, 2, GREX_CVT_VCVTQ)
GREX_CVT(f, 64, u, 64, 2, GREX_CVT_VCVTQ)
GREX_CVT(f, 64, f, 32, 2, GREX_CVT_VCVTQ_LOW)
// f32
GREX_CVT(f, 32, f, 64, 2, GREX_CVT_f32_f64)
GREX_CVT(f, 32, i, 32, 4, GREX_CVT_VCVTQ)
GREX_CVT(f, 32, u, 32, 4, GREX_CVT_VCVTQ)
// i64
GREX_CVT(i, 64, f, 64, 2, GREX_CVT_VCVTQ)
GREX_CVT(i, 64, i, 32, 2, GREX_CVT_INTx2)
// u64
GREX_CVT(u, 64, f, 64, 2, GREX_CVT_VCVTQ)
GREX_CVT(u, 64, u, 32, 2, GREX_CVT_INTx2)

// f32
GREX_CVT(f, 32, i, 64, 2, GREX_CVT_f32_i64)
GREX_CVT(f, 32, u, 64, 2, GREX_CVT_f32_i64)
// i64
GREX_CVT(i, 64, f, 32, 2, GREX_CVT_x64_f32)
// i64
GREX_CVT(u, 64, f, 32, 2, GREX_CVT_x64_f32)

// Trivial no-op cases
// Source and destination type identical
template<Vectorizable T, std::size_t tSize>
inline Vector<T, tSize> convert(Vector<T, tSize> v, TypeTag<T> /*tag*/) {
  return v;
}
// Integers with the same number of bits
template<IntVectorizable TDst, IntVectorizable TSrc, std::size_t tSize>
requires(sizeof(TDst) == sizeof(TSrc))
inline Vector<TDst, tSize> convert(Vector<TSrc, tSize> v, TypeTag<TDst> /*tag*/) {
  return {.r = v.r};
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_CONVERT_HPP
