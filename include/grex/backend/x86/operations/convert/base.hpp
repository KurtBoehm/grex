// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_CONVERT_BASE_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_CONVERT_BASE_HPP

#include <cstddef>
#include <type_traits>

#include <immintrin.h>

#include "grex/backend/base.hpp"
#include "grex/backend/choosers.hpp"
#include "grex/backend/macros/base.hpp"
#include "grex/backend/macros/math.hpp"
#include "grex/backend/x86/instruction-sets.hpp"
#include "grex/backend/x86/macros/intrinsics.hpp"
#include "grex/backend/x86/macros/types.hpp"
#include "grex/backend/x86/operations/merge.hpp"
#include "grex/backend/x86/operations/split.hpp"
#include "grex/base.hpp"

#if GREX_X86_64_LEVEL < 4
#include "grex/backend/x86/operations/mask-convert.hpp"
#endif

namespace grex::backend {
// Conversion intrinsics
#define GREX_CVT_INTRINSIC_EPUI(DSTKIND, DSTBITS, SRCKIND, SRCBITS, SIZE, BITPREFIX, REGISTERBITS) \
  return GREX_VECTOR_TYPE(DSTKIND, DSTBITS, \
                          SIZE){GREX_CAT(BITPREFIX##_cvt, GREX_EPU_SUFFIX(SRCKIND, SRCBITS), _, \
                                         GREX_EPI_SUFFIX(DSTKIND, DSTBITS))(v.registr())};
#define GREX_CVT_INTRINSIC_EPU(DSTKIND, DSTBITS, SRCKIND, SRCBITS, SIZE, BITPREFIX, REGISTERBITS) \
  return GREX_VECTOR_TYPE(DSTKIND, DSTBITS, \
                          SIZE){GREX_CAT(BITPREFIX##_cvt, GREX_EPU_SUFFIX(SRCKIND, SRCBITS), _, \
                                         GREX_EPU_SUFFIX(DSTKIND, DSTBITS))(v.registr())};
#define GREX_CVT_INTRINSIC_EPI(DSTKIND, DSTBITS, SRCKIND, SRCBITS, SIZE, BITPREFIX, REGISTERBITS) \
  return GREX_VECTOR_TYPE(DSTKIND, DSTBITS, \
                          SIZE){GREX_CAT(BITPREFIX##_cvt, GREX_EPI_SUFFIX(SRCKIND, SRCBITS), _, \
                                         GREX_EPI_SUFFIX(DSTKIND, DSTBITS))(v.registr())};

#define GREX_CVTT_INTRINSIC_EPU(DSTKIND, DSTBITS, SRCKIND, SRCBITS, SIZE, BITPREFIX, REGISTERBITS) \
  return GREX_VECTOR_TYPE(DSTKIND, DSTBITS, \
                          SIZE){GREX_CAT(BITPREFIX##_cvtt, GREX_EPU_SUFFIX(SRCKIND, SRCBITS), _, \
                                         GREX_EPU_SUFFIX(DSTKIND, DSTBITS))(v.registr())};

// Floating-point → small integer (< 32 bits): first convert to i32, then narrow.
#define GREX_CVT_IMPL_F2SMALLI(DSTKIND, DSTBITS, SRCKIND, SRCBITS, SIZE, BITPREFIX, REGISTERBITS) \
  const auto vi32 = convert(v, type_tag<i32>).registr(); \
  return convert(GREX_VECTOR_TYPE(i, 32, SIZE){vi32}, type_tag<DSTKIND##DSTBITS>);

// Small integer (< 32 bits) → floating-point: first widen to i32, then convert to floating-point.
#define GREX_CVT_IMPL_SMALLI2F(DSTKIND, DSTBITS, SRCKIND, SRCBITS, SIZE, BITPREFIX, REGISTERBITS) \
  const GREX_VECTOR_TYPE(SRCKIND, SRCBITS, GREX_MAX(SIZE, 4)) full{v.registr()}; \
  const auto vi32 = convert(full, type_tag<i32>).r; \
  return convert(GREX_VECTOR_TYPE(i, 32, SIZE){vi32}, type_tag<DSTKIND##DSTBITS>);

// Base macros that dispatch to the (Dst, Src, Size) specialization
// Vector
#define GREX_CVT_IMPL(DSTKIND, DSTBITS, SRCKIND, SRCBITS, SIZE, ...) \
  GREX_CVT_IMPL_##DSTKIND##DSTBITS##_##SRCKIND##SRCBITS##_##SIZE( \
    DSTKIND, DSTBITS, SRCKIND, SRCBITS, SIZE __VA_OPT__(, ) __VA_ARGS__)
#define GREX_CVT(DSTKIND, DSTBITS, SRCKIND, SRCBITS, SIZE, ...) \
  inline GREX_VECTOR_TYPE(DSTKIND, DSTBITS, SIZE) \
    convert(GREX_VECTOR_TYPE(SRCKIND, SRCBITS, SIZE) v, TypeTag<DSTKIND##DSTBITS>) { \
    GREX_CVT_IMPL(DSTKIND, DSTBITS, SRCKIND, SRCBITS, SIZE __VA_OPT__(, ) __VA_ARGS__) \
  }
#define GREX_CVT_SUPER(DSTKIND, DSTBITS, SRCKIND, SRCBITS, SIZE, ...) \
  inline VectorFor<DSTKIND##DSTBITS, SIZE> convert(VectorFor<SRCKIND##SRCBITS, SIZE> v, \
                                                   TypeTag<DSTKIND##DSTBITS>) { \
    GREX_CVT_IMPL(DSTKIND, DSTBITS, SRCKIND, SRCBITS, SIZE __VA_OPT__(, ) __VA_ARGS__) \
  }
// Mask
#define GREX_CVTMSK_IMPL(DSTKIND, DSTBITS, SRCKIND, SRCBITS, SIZE, ...) \
  GREX_CVTMSK_IMPL_##DSTBITS##_##SRCBITS##_##SIZE(DSTKIND, DSTBITS, SRCKIND, SRCBITS, \
                                                  SIZE __VA_OPT__(, ) __VA_ARGS__)
#define GREX_CVTMSK(DSTKIND, DSTBITS, SRCKIND, SRCBITS, SIZE, ...) \
  inline GREX_MASK_TYPE(DSTKIND, DSTBITS, SIZE) \
    convert(GREX_MASK_TYPE(SRCKIND, SRCBITS, SIZE) m, TypeTag<DSTKIND##DSTBITS>) { \
    GREX_CVTMSK_IMPL(DSTKIND, DSTBITS, SRCKIND, SRCBITS, SIZE __VA_OPT__(, ) __VA_ARGS__) \
  }
#define GREX_CVTMSK_SUPER(DSTKIND, DSTBITS, SRCKIND, SRCBITS, SIZE, ...) \
  inline MaskFor<DSTKIND##DSTBITS, SIZE> convert(MaskFor<SRCKIND##SRCBITS, SIZE> m, \
                                                 TypeTag<DSTKIND##DSTBITS>) { \
    GREX_CVTMSK_IMPL(DSTKIND, DSTBITS, SRCKIND, SRCBITS, SIZE __VA_OPT__(, ) __VA_ARGS__) \
  }

#define GREX_CVT_DEF_ALL_BASE(MACRO, BITPREFIX, REGISTERBITS) \
  /* Integer widening: 2× element width (same signedness). */ \
  MACRO(i, 16, i, 8, GREX_DIVIDE(REGISTERBITS, 16), BITPREFIX, REGISTERBITS) \
  MACRO(u, 16, u, 8, GREX_DIVIDE(REGISTERBITS, 16), BITPREFIX, REGISTERBITS) \
  MACRO(i, 32, i, 16, GREX_DIVIDE(REGISTERBITS, 32), BITPREFIX, REGISTERBITS) \
  MACRO(u, 32, u, 16, GREX_DIVIDE(REGISTERBITS, 32), BITPREFIX, REGISTERBITS) \
  MACRO(i, 64, i, 32, GREX_DIVIDE(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  MACRO(u, 64, u, 32, GREX_DIVIDE(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  /* Integer widening: 4× element width (same signedness). */ \
  MACRO(i, 32, i, 8, GREX_DIVIDE(REGISTERBITS, 32), BITPREFIX, REGISTERBITS) \
  MACRO(u, 32, u, 8, GREX_DIVIDE(REGISTERBITS, 32), BITPREFIX, REGISTERBITS) \
  MACRO(i, 64, i, 16, GREX_DIVIDE(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  MACRO(u, 64, u, 16, GREX_DIVIDE(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  /* Integer widening: 8× element width (same signedness). */ \
  MACRO(i, 64, i, 8, GREX_DIVIDE(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  MACRO(u, 64, u, 8, GREX_DIVIDE(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  /* Unsigned narrowing: 1/2 element width. */ \
  MACRO(u, 8, u, 16, GREX_DIVIDE(REGISTERBITS, 16), BITPREFIX, REGISTERBITS) \
  MACRO(u, 16, u, 32, GREX_DIVIDE(REGISTERBITS, 32), BITPREFIX, REGISTERBITS) \
  MACRO(u, 32, u, 64, GREX_DIVIDE(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  /* Unsigned narrowing: 1/4 element width. */ \
  MACRO(u, 8, u, 32, GREX_DIVIDE(REGISTERBITS, 32), BITPREFIX, REGISTERBITS) \
  MACRO(u, 16, u, 64, GREX_DIVIDE(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  /* Unsigned narrowing: 1/8 element width. */ \
  MACRO(u, 8, u, 64, GREX_DIVIDE(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  /* Floating-point ↔ floating-point (same register width). */ \
  MACRO(f, 64, f, 32, GREX_DIVIDE(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  MACRO(f, 32, f, 64, GREX_DIVIDE(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  /* Integer → floating-point */ \
  /* integer → f64. */ \
  MACRO(f, 64, i, 64, GREX_DIVIDE(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  MACRO(f, 64, u, 64, GREX_DIVIDE(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  MACRO(f, 64, i, 32, GREX_DIVIDE(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  MACRO(f, 64, u, 32, GREX_DIVIDE(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  MACRO(f, 64, i, 16, GREX_DIVIDE(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  MACRO(f, 64, u, 16, GREX_DIVIDE(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  MACRO(f, 64, i, 8, GREX_DIVIDE(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  MACRO(f, 64, u, 8, GREX_DIVIDE(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  /* integer → f32. */ \
  MACRO(f, 32, i, 64, GREX_DIVIDE(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  MACRO(f, 32, u, 64, GREX_DIVIDE(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  MACRO(f, 32, i, 32, GREX_DIVIDE(REGISTERBITS, 32), BITPREFIX, REGISTERBITS) \
  MACRO(f, 32, u, 32, GREX_DIVIDE(REGISTERBITS, 32), BITPREFIX, REGISTERBITS) \
  MACRO(f, 32, i, 16, GREX_DIVIDE(REGISTERBITS, 32), BITPREFIX, REGISTERBITS) \
  MACRO(f, 32, u, 16, GREX_DIVIDE(REGISTERBITS, 32), BITPREFIX, REGISTERBITS) \
  MACRO(f, 32, i, 8, GREX_DIVIDE(REGISTERBITS, 32), BITPREFIX, REGISTERBITS) \
  MACRO(f, 32, u, 8, GREX_DIVIDE(REGISTERBITS, 32), BITPREFIX, REGISTERBITS) \
  /* Floating-point → integer */ \
  /* f64 → integer */ \
  MACRO(i, 64, f, 64, GREX_DIVIDE(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  MACRO(u, 64, f, 64, GREX_DIVIDE(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  MACRO(i, 32, f, 64, GREX_DIVIDE(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  MACRO(u, 32, f, 64, GREX_DIVIDE(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  MACRO(i, 16, f, 64, GREX_DIVIDE(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  MACRO(u, 16, f, 64, GREX_DIVIDE(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  MACRO(i, 8, f, 64, GREX_DIVIDE(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  MACRO(u, 8, f, 64, GREX_DIVIDE(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  /* f32 → integer */ \
  MACRO(i, 64, f, 32, GREX_DIVIDE(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  MACRO(u, 64, f, 32, GREX_DIVIDE(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  MACRO(i, 32, f, 32, GREX_DIVIDE(REGISTERBITS, 32), BITPREFIX, REGISTERBITS) \
  MACRO(u, 32, f, 32, GREX_DIVIDE(REGISTERBITS, 32), BITPREFIX, REGISTERBITS) \
  MACRO(i, 16, f, 32, GREX_DIVIDE(REGISTERBITS, 32), BITPREFIX, REGISTERBITS) \
  MACRO(u, 16, f, 32, GREX_DIVIDE(REGISTERBITS, 32), BITPREFIX, REGISTERBITS) \
  MACRO(i, 8, f, 32, GREX_DIVIDE(REGISTERBITS, 32), BITPREFIX, REGISTERBITS) \
  MACRO(u, 8, f, 32, GREX_DIVIDE(REGISTERBITS, 32), BITPREFIX, REGISTERBITS)
#define GREX_CVT_DEF_ALL(BITPREFIX, REGISTERBITS) \
  GREX_CVT_DEF_ALL_BASE(GREX_CVT, BITPREFIX, REGISTERBITS)
#define GREX_CVTMSK_DEF_ALL(BITPREFIX, REGISTERBITS) \
  GREX_CVT_DEF_ALL_BASE(GREX_CVTMSK, BITPREFIX, REGISTERBITS)

#if GREX_X86_64_LEVEL >= 4
// Baseline for compact masks:
// If converting to a super-native mask, process lower/upper halves recursively;
// otherwise, just reinterpret the underlying mask register.
template<AnyMask Mask, typename Dst>
requires(!AnySuperNativeMask<Mask>)
inline MaskFor<Dst, Mask::size> convert(Mask mask, TypeTag<Dst> tag) {
  using Out = MaskFor<Dst, Mask::size>;
  if constexpr (is_supernative<Dst, Mask::size>) {
    return Out{
      .lower = convert(split(mask, index_tag<0>), tag),
      .upper = convert(split(mask, index_tag<1>), tag),
    };
  } else {
    using OutRegister = Out::Register;
    return Out{OutRegister(mask.registr())};
  }
}
// Super-native mask → native/sub-native mask: convert each half and then merge.
template<AnySuperNativeMask Mask, typename Dst>
inline MaskFor<Dst, Mask::size> convert(Mask mask, TypeTag<Dst> tag) {
  return merge(convert(mask.lower, tag), convert(mask.upper, tag));
}
#else
// Baseline for broad masks:
// Reinterpret mask as a signed integer vector, convert that, then reinterpret as a mask
// of the desired element type.
template<AnyMask Mask, typename Dst>
inline auto convert(Mask mask, TypeTag<Dst> /*tag*/) {
  return vector2mask(convert(mask2vector(mask), type_tag<SignedInt<sizeof(Dst)>>), type_tag<Dst>);
}
#endif

// Native → super-native: Split native
template<Vectorizable Dst, Vectorizable Src, std::size_t N>
requires(is_supernative<Dst, N>)
inline VectorFor<Dst, N> convert(NativeVector<Src, N> v, TypeTag<Dst> tag) {
  return merge(convert(get_low(v), tag), convert(get_high(v), tag));
}

// Integer → larger integer (native) with different signedness:
// widen while preserving source signedness, then reinterpret to destination type.
template<IntVectorizable Dst, IntVector Src>
requires(sizeof(Dst) > sizeof(ValueOf<Src>) &&
         (std::is_signed_v<Dst> != std::is_signed_v<ValueOf<Src>>) && is_native<Dst, size_of<Src>>)
inline NativeVector<Dst, size_of<Src>> convert(Src v, TypeTag<Dst> /*tag*/) {
  return {.r = convert(v, type_tag<CopySignInt<ValueOf<Src>, sizeof(Dst)>>).r};
}

// Integer (native) → smaller integer where at least one type is signed:
// perform truncation via an unsigned path.
template<IntVectorizable Dst, IntVectorizable Src, std::size_t N>
requires((std::is_signed_v<Dst> || std::is_signed_v<Src>) && sizeof(Dst) < sizeof(Src))
inline VectorFor<Dst, N> convert(NativeVector<Src, N> v, TypeTag<Dst> /*tag*/) {
  const auto s = convert(NativeVector<UnsignedOf<Src>, N>{v.r}, type_tag<UnsignedOf<Dst>>);
  return VectorFor<Dst, N>{s.registr()};
}

// Sub-native to super-native, integer → any: convert to the smallest integer type for which
// the vector with N lanes is native and go from there
template<Vectorizable Dst, IntVectorizable Src, std::size_t N>
requires(is_supernative<Dst, N>)
inline VectorFor<Dst, N> convert(SubVector<Src, N> v, TypeTag<Dst> tag) {
  return convert(convert(v, type_tag<CopySignInt<Src, 16 / N>>), tag);
}

// Super-native → sub-native/native for integers:
// convert each half to the next-smaller integer type, merge halves,
// then continue converting to the final destination type.
template<IntVectorizable Dst, IntVector Half>
requires(!is_supernative<Dst, Half::size * 2>)
inline VectorFor<Dst, Half::size * 2> convert(SuperVector<Half> v, TypeTag<Dst> /*tag*/) {
  using Src = ValueOf<Half>;
  static constexpr std::size_t tmp_bytes = sizeof(Src) / 2;
  static constexpr bool tmp_signed = std::is_signed_v<Src>;
  using Tmp = std::conditional_t<tmp_signed, SignedInt<tmp_bytes>, UnsignedInt<tmp_bytes>>;
  const auto tmp = merge(convert(v.lower, type_tag<Tmp>), convert(v.upper, type_tag<Tmp>));
  return convert(tmp, type_tag<Dst>);
}
// Super-native → sub-native/native, floating-point → integer:
// cast to same-sized integer, then go from there.
// Excludes binary16 sources, as there is no conversion between 16-bit integers and binary16 without
// AVX512-FP.
template<IntVectorizable Dst, FloatVector Half>
requires(!is_supernative<Dst, Half::size * 2> && !Float16<ValueOf<Half>>)
inline VectorFor<Dst, Half::size * 2> convert(SuperVector<Half> v, TypeTag<Dst> tag) {
  return convert(convert(v, type_tag<CopySignInt<Dst, sizeof(ValueOf<Half>)>>), tag);
}
// Super-native → sub-native/native, integer → floating-point:
// cast to same-sized floating-point, then go from there.
template<FloatVectorizable Dst, IntVector Half>
requires(!is_supernative<Dst, Half::size * 2>)
inline VectorFor<Dst, Half::size * 2> convert(SuperVector<Half> v, TypeTag<Dst> tag) {
  return convert(convert(v, type_tag<Float<sizeof(ValueOf<Half>)>>), tag);
}
// Super-native → sub-native/native, floating-point → floating-point:
// split into halves
template<FloatVectorizable Dst, FloatVector Half>
requires(!is_supernative<Dst, Half::size * 2>)
inline VectorFor<Dst, Half::size * 2> convert(SuperVector<Half> v, TypeTag<Dst> tag) {
  return merge(convert(v.lower, tag), convert(v.upper, tag));
}

#if GREX_X86_64_LEVEL < 4
template<typename Dst, typename Half>
requires(is_supernative<Dst, Half::size * 2>)
inline MaskFor<Dst, Half::size * 2> convert(SuperMask<Half> v, TypeTag<Dst> /*tag*/) {
  return {.lower = convert(v.lower, type_tag<Dst>), .upper = convert(v.upper, type_tag<Dst>)};
}
#endif
} // namespace grex::backend

#include "grex/backend/shared/operations/convert.hpp" // IWYU pragma: export

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_CONVERT_BASE_HPP
