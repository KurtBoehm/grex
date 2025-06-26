// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_CONVERT_BASE_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_CONVERT_BASE_HPP

#include <concepts>
#include <cstddef>
#include <type_traits>

#include <immintrin.h>

#include "grex/backend/choosers.hpp"
#include "grex/backend/defs.hpp"
#include "grex/backend/x86/helpers.hpp"
#include "grex/backend/x86/instruction-sets.hpp"
#include "grex/backend/x86/operations/mask-convert.hpp"
#include "grex/backend/x86/operations/merge.hpp"
#include "grex/backend/x86/operations/split.hpp"
#include "grex/base/defs.hpp"

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

// Floating-point → integer with less than 32 bits: Convert to i32 and go from there
#define GREX_CVT_IMPL_F2SMALLI(DSTKIND, DSTBITS, SRCKIND, SRCBITS, SIZE, BITPREFIX, REGISTERBITS) \
  const auto vi32 = convert(v, type_tag<i32>).registr(); \
  return convert(GREX_VECTOR_TYPE(i, 32, SIZE){vi32}, type_tag<DSTKIND##DSTBITS>);

// Integer with less than 32 bits → floating-point: Convert to i32 and go from there
#define GREX_CVT_IMPL_SMALLI2F(DSTKIND, DSTBITS, SRCKIND, SRCBITS, SIZE, BITPREFIX, REGISTERBITS) \
  const GREX_VECTOR_TYPE(SRCKIND, SRCBITS, GREX_MAX(SIZE, 4)) full{v.registr()}; \
  const auto vi32 = convert(full, type_tag<i32>).r; \
  return convert(GREX_VECTOR_TYPE(i, 32, SIZE){vi32}, type_tag<DSTKIND##DSTBITS>);

// Base macros
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
  /* Double integer size */ \
  MACRO(i, 16, i, 8, GREX_ELEMENTS(REGISTERBITS, 16), BITPREFIX, REGISTERBITS) \
  MACRO(u, 16, u, 8, GREX_ELEMENTS(REGISTERBITS, 16), BITPREFIX, REGISTERBITS) \
  MACRO(i, 32, i, 16, GREX_ELEMENTS(REGISTERBITS, 32), BITPREFIX, REGISTERBITS) \
  MACRO(u, 32, u, 16, GREX_ELEMENTS(REGISTERBITS, 32), BITPREFIX, REGISTERBITS) \
  MACRO(i, 64, i, 32, GREX_ELEMENTS(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  MACRO(u, 64, u, 32, GREX_ELEMENTS(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  /* Quadruple size */ \
  MACRO(i, 32, i, 8, GREX_ELEMENTS(REGISTERBITS, 32), BITPREFIX, REGISTERBITS) \
  MACRO(u, 32, u, 8, GREX_ELEMENTS(REGISTERBITS, 32), BITPREFIX, REGISTERBITS) \
  MACRO(i, 64, i, 16, GREX_ELEMENTS(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  MACRO(u, 64, u, 16, GREX_ELEMENTS(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  /* Octuple size */ \
  MACRO(i, 64, i, 8, GREX_ELEMENTS(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  MACRO(u, 64, u, 8, GREX_ELEMENTS(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  /* Halve integer size */ \
  MACRO(u, 8, u, 16, GREX_ELEMENTS(REGISTERBITS, 16), BITPREFIX, REGISTERBITS) \
  MACRO(u, 16, u, 32, GREX_ELEMENTS(REGISTERBITS, 32), BITPREFIX, REGISTERBITS) \
  MACRO(u, 32, u, 64, GREX_ELEMENTS(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  /* Quarter integer size */ \
  MACRO(u, 8, u, 32, GREX_ELEMENTS(REGISTERBITS, 32), BITPREFIX, REGISTERBITS) \
  MACRO(u, 16, u, 64, GREX_ELEMENTS(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  /* Divide integer size by eight */ \
  MACRO(u, 8, u, 64, GREX_ELEMENTS(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  /* Floating-point conversions */ \
  MACRO(f, 64, f, 32, GREX_ELEMENTS(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  MACRO(f, 32, f, 64, GREX_ELEMENTS(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  /* Integer → floating-point */ \
  /* f64 */ \
  MACRO(f, 64, i, 64, GREX_ELEMENTS(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  MACRO(f, 64, u, 64, GREX_ELEMENTS(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  MACRO(f, 64, i, 32, GREX_ELEMENTS(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  MACRO(f, 64, u, 32, GREX_ELEMENTS(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  MACRO(f, 64, i, 16, GREX_ELEMENTS(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  MACRO(f, 64, u, 16, GREX_ELEMENTS(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  MACRO(f, 64, i, 8, GREX_ELEMENTS(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  MACRO(f, 64, u, 8, GREX_ELEMENTS(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  /* f32 */ \
  MACRO(f, 32, i, 64, GREX_ELEMENTS(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  MACRO(f, 32, u, 64, GREX_ELEMENTS(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  MACRO(f, 32, i, 32, GREX_ELEMENTS(REGISTERBITS, 32), BITPREFIX, REGISTERBITS) \
  MACRO(f, 32, u, 32, GREX_ELEMENTS(REGISTERBITS, 32), BITPREFIX, REGISTERBITS) \
  MACRO(f, 32, i, 16, GREX_ELEMENTS(REGISTERBITS, 32), BITPREFIX, REGISTERBITS) \
  MACRO(f, 32, u, 16, GREX_ELEMENTS(REGISTERBITS, 32), BITPREFIX, REGISTERBITS) \
  MACRO(f, 32, i, 8, GREX_ELEMENTS(REGISTERBITS, 32), BITPREFIX, REGISTERBITS) \
  MACRO(f, 32, u, 8, GREX_ELEMENTS(REGISTERBITS, 32), BITPREFIX, REGISTERBITS) \
  /* Floating-point → integer */ \
  /* f64 */ \
  MACRO(i, 64, f, 64, GREX_ELEMENTS(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  MACRO(u, 64, f, 64, GREX_ELEMENTS(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  MACRO(i, 32, f, 64, GREX_ELEMENTS(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  MACRO(u, 32, f, 64, GREX_ELEMENTS(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  MACRO(i, 16, f, 64, GREX_ELEMENTS(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  MACRO(u, 16, f, 64, GREX_ELEMENTS(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  MACRO(i, 8, f, 64, GREX_ELEMENTS(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  MACRO(u, 8, f, 64, GREX_ELEMENTS(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  /* f32 */ \
  MACRO(i, 64, f, 32, GREX_ELEMENTS(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  MACRO(u, 64, f, 32, GREX_ELEMENTS(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  MACRO(i, 32, f, 32, GREX_ELEMENTS(REGISTERBITS, 32), BITPREFIX, REGISTERBITS) \
  MACRO(u, 32, f, 32, GREX_ELEMENTS(REGISTERBITS, 32), BITPREFIX, REGISTERBITS) \
  MACRO(i, 16, f, 32, GREX_ELEMENTS(REGISTERBITS, 32), BITPREFIX, REGISTERBITS) \
  MACRO(u, 16, f, 32, GREX_ELEMENTS(REGISTERBITS, 32), BITPREFIX, REGISTERBITS) \
  MACRO(i, 8, f, 32, GREX_ELEMENTS(REGISTERBITS, 32), BITPREFIX, REGISTERBITS) \
  MACRO(u, 8, f, 32, GREX_ELEMENTS(REGISTERBITS, 32), BITPREFIX, REGISTERBITS)
#define GREX_CVT_DEF_ALL(BITPREFIX, REGISTERBITS) \
  GREX_CVT_DEF_ALL_BASE(GREX_CVT, BITPREFIX, REGISTERBITS)
#define GREX_CVTMSK_DEF_ALL(BITPREFIX, REGISTERBITS) \
  GREX_CVT_DEF_ALL_BASE(GREX_CVTMSK, BITPREFIX, REGISTERBITS)

// Trivial no-op cases
// Source and destination type identical
template<Vectorizable T, std::size_t tSize>
inline Vector<T, tSize> convert(Vector<T, tSize> v, TypeTag<T> /*tag*/) {
  return v;
}
// Integers with the same number of bits
template<Vectorizable TDst, Vectorizable TSrc, std::size_t tSize>
requires(std::integral<TDst> && std::integral<TSrc> && sizeof(TDst) == sizeof(TSrc))
inline Vector<TDst, tSize> convert(Vector<TSrc, tSize> v, TypeTag<TDst> /*tag*/) {
  return {.r = v.r};
}

#if GREX_X86_64_LEVEL >= 4
// Baseline for compact masks: Reinterpret
template<AnyMask TMask, typename TDst>
requires(!AnySuperNativeMask<TMask>)
inline MaskFor<TDst, TMask::size> convert(TMask mask, TypeTag<TDst> tag) {
  using Out = MaskFor<TDst, TMask::size>;
  if constexpr (is_supernative<TDst, TMask::size>) {
    return Out{
      .lower = convert(split(mask, index_tag<0>), tag),
      .upper = convert(split(mask, index_tag<1>), tag),
    };
  } else {
    using OutRegister = Out::Register;
    return Out{OutRegister(mask.registr())};
  }
}
// Super-native to native/sub-native
template<AnySuperNativeMask TMask, typename TDst>
inline MaskFor<TDst, TMask::size> convert(TMask mask, TypeTag<TDst> tag) {
  return merge(convert(mask.lower, tag), convert(mask.upper, tag));
}
#else
// Baseline for broad masks: Convert as signed integer
template<AnyMask TMask, typename TDst>
inline auto convert(TMask mask, TypeTag<TDst> /*tag*/) {
  return vector2mask(convert(mask2vector(mask), type_tag<SignedInt<sizeof(TDst)>>), type_tag<TDst>);
}
#endif

// Integer to larger integer with different signedness: Increase size while retaining signedness,
// main condition: mixed signedness, increasing size
template<typename TDst, typename TSrc>
concept ConvertIntegralUp = std::integral<TDst> && std::integral<TSrc> &&
                            (std::unsigned_integral<TDst> != std::unsigned_integral<TSrc>) &&
                            sizeof(TDst) > sizeof(TSrc);
// native → native
template<Vectorizable TDst, Vectorizable TSrc, std::size_t tSize>
requires(ConvertIntegralUp<TDst, TSrc> && is_native<TDst, tSize>)
inline Vector<TDst, tSize> convert(Vector<TSrc, tSize> v, TypeTag<TDst> /*tag*/) {
  using Tmp = std::conditional_t<std::unsigned_integral<TSrc>, UnsignedOf<TDst>, SignedOf<TDst>>;
  return {.r = convert(v, type_tag<Tmp>).r};
}
// sub-native → native (sub-native → sub-native: covered by the base case)
template<Vectorizable TDst, Vectorizable TSrc, std::size_t tPart, std::size_t tSize>
requires(ConvertIntegralUp<TDst, TSrc> && is_native<TDst, tPart>)
inline VectorFor<TDst, tPart> convert(SubVector<TSrc, tPart, tSize> v, TypeTag<TDst> /*tag*/) {
  using Temp = std::conditional_t<std::unsigned_integral<TSrc>, UnsignedOf<TDst>, SignedOf<TDst>>;
  return {.r = convert(v, type_tag<Temp>).r};
}

// Integer to smaller integer where one is signed: Truncation → use unsigned version
// main condition: at least one signed, decreasing size
template<typename TDst, typename TSrc>
concept ConvertIntegralDown =
  std::integral<TDst> && std::integral<TSrc> &&
  (!std::unsigned_integral<TDst> || !std::unsigned_integral<TSrc>) && sizeof(TDst) < sizeof(TSrc);
// native → sub-native/native
template<Vectorizable TDst, Vectorizable TSrc, std::size_t tSize>
requires(ConvertIntegralDown<TDst, TSrc> && !is_supernative<TDst, tSize>)
inline VectorFor<TDst, tSize> convert(Vector<TSrc, tSize> v, TypeTag<TDst> /*tag*/) {
  const auto s = convert(Vector<UnsignedOf<TSrc>, tSize>{v.r}, type_tag<UnsignedOf<TDst>>);
  return VectorFor<TDst, tSize>{s.registr()};
}
// sub-native → sub-native: Covered by the base case

// Conversion between sub-native vectors: Increase size so that the larger is/both are native
template<Vectorizable TDst, Vectorizable TSrc, std::size_t tPart, std::size_t tSize>
requires(is_subnative<TDst, tPart>)
inline VectorFor<TDst, tPart> convert(SubVector<TSrc, tPart, tSize> v, TypeTag<TDst> /*tag*/) {
  using Out = VectorFor<TDst, tPart>;
  constexpr std::size_t work_size = std::min(tSize, Out::Full::size);
  static_assert(work_size > tPart);
  const auto s = convert(VectorFor<TSrc, work_size>{v.registr()}, type_tag<TDst>);
  return Out{s.registr()};
}

// Conversion between super-native vectors/masks: Work on the halves separately
template<typename TDst, typename THalf>
requires(is_supernative<TDst, THalf::size * 2>)
inline VectorFor<TDst, THalf::size * 2> convert(SuperVector<THalf> v, TypeTag<TDst> /*tag*/) {
  return {.lower = convert(v.lower, type_tag<TDst>), .upper = convert(v.upper, type_tag<TDst>)};
}
#if GREX_X86_64_LEVEL < 4
template<typename TDst, typename THalf>
requires(is_supernative<TDst, THalf::size * 2>)
inline MaskFor<TDst, THalf::size * 2> convert(SuperMask<THalf> v, TypeTag<TDst> /*tag*/) {
  return {.lower = convert(v.lower, type_tag<TDst>), .upper = convert(v.upper, type_tag<TDst>)};
}
#endif

// Conversion between super-native and non-super-native
// native → super-native: Split into halves and convert separately
// TODO Separate implementations for four-fold and eight-fold super-native vectors?!
template<typename TDst, typename TSrc, std::size_t tSize>
requires(is_supernative<TDst, tSize>)
inline VectorFor<TDst, tSize> convert(Vector<TSrc, tSize> v, TypeTag<TDst> /*tag*/) {
  return {
    .lower = convert(split(v, index_tag<0>), type_tag<TDst>),
    .upper = convert(split(v, index_tag<1>), type_tag<TDst>),
  };
}
// sub-native → super-native: Split into halves and convert separately
// TODO Separate implementations for four-fold and eight-fold super-native vectors?!
template<typename TDst, typename TSrc, std::size_t tPart, std::size_t tSize>
requires(is_supernative<TDst, tPart>)
inline VectorFor<TDst, tPart> convert(SubVector<TSrc, tPart, tSize> v, TypeTag<TDst> /*tag*/) {
  return {
    .lower = convert(split(v, index_tag<0>), type_tag<TDst>),
    .upper = convert(split(v, index_tag<1>), type_tag<TDst>),
  };
}
// Super-native → sub-native/native for non-integers: Convert halves separately and merge
// TODO Separate implementations for four-fold and eight-fold super-native vectors?
template<typename TDst, typename THalf>
requires((!std::integral<TDst> || !std::integral<typename THalf::Value>) &&
         !is_supernative<TDst, THalf::size * 2>)
inline VectorFor<TDst, THalf::size * 2> convert(SuperVector<THalf> v, TypeTag<TDst> /*tag*/) {
  return merge(convert(v.lower, type_tag<TDst>), convert(v.upper, type_tag<TDst>));
}
// Super-native → sub-native/native for integers: Convert to next-smaller integer type, merge,
// and continue
// TODO Separate implementations for four-fold and eight-fold super-native vectors?
template<typename TDst, typename THalf>
requires(std::integral<TDst> && std::integral<typename THalf::Value> &&
         !is_supernative<TDst, THalf::size * 2>)
inline VectorFor<TDst, THalf::size * 2> convert(SuperVector<THalf> v, TypeTag<TDst> /*tag*/) {
  using Src = THalf::Value;
  static constexpr std::size_t tmp_bytes = sizeof(Src) / 2;
  static constexpr bool tmp_signed = std::signed_integral<Src>;
  using Tmp = std::conditional_t<tmp_signed, SignedInt<tmp_bytes>, UnsignedInt<tmp_bytes>>;
  const auto tmp = merge(convert(v.lower, type_tag<Tmp>), convert(v.upper, type_tag<Tmp>));
  return convert(tmp, type_tag<TDst>);
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_CONVERT_BASE_HPP
