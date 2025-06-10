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

#define GREX_CVT_DEF_ALL(BITPREFIX, REGISTERBITS) \
  /* Double integer size */ \
  GREX_CVT(i, 16, i, 8, GREX_ELEMENTS(REGISTERBITS, 16), BITPREFIX, REGISTERBITS) \
  GREX_CVT(u, 16, u, 8, GREX_ELEMENTS(REGISTERBITS, 16), BITPREFIX, REGISTERBITS) \
  GREX_CVT(i, 32, i, 16, GREX_ELEMENTS(REGISTERBITS, 32), BITPREFIX, REGISTERBITS) \
  GREX_CVT(u, 32, u, 16, GREX_ELEMENTS(REGISTERBITS, 32), BITPREFIX, REGISTERBITS) \
  GREX_CVT(i, 64, i, 32, GREX_ELEMENTS(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  GREX_CVT(u, 64, u, 32, GREX_ELEMENTS(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  /* Quadruple size */ \
  GREX_CVT(i, 32, i, 8, GREX_ELEMENTS(REGISTERBITS, 32), BITPREFIX, REGISTERBITS) \
  GREX_CVT(u, 32, u, 8, GREX_ELEMENTS(REGISTERBITS, 32), BITPREFIX, REGISTERBITS) \
  GREX_CVT(i, 64, i, 16, GREX_ELEMENTS(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  GREX_CVT(u, 64, u, 16, GREX_ELEMENTS(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  /* Octuple size */ \
  GREX_CVT(i, 64, i, 8, GREX_ELEMENTS(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  GREX_CVT(u, 64, u, 8, GREX_ELEMENTS(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  /* Halve integer size */ \
  GREX_CVT(u, 8, u, 16, GREX_ELEMENTS(REGISTERBITS, 16), BITPREFIX, REGISTERBITS) \
  GREX_CVT(u, 16, u, 32, GREX_ELEMENTS(REGISTERBITS, 32), BITPREFIX, REGISTERBITS) \
  GREX_CVT(u, 32, u, 64, GREX_ELEMENTS(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  /* Quarter integer size */ \
  GREX_CVT(u, 8, u, 32, GREX_ELEMENTS(REGISTERBITS, 32), BITPREFIX, REGISTERBITS) \
  GREX_CVT(u, 16, u, 64, GREX_ELEMENTS(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  /* Divide integer size by eight */ \
  GREX_CVT(u, 8, u, 64, GREX_ELEMENTS(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  /* Floating-point conversions */ \
  GREX_CVT(f, 64, f, 32, GREX_ELEMENTS(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  GREX_CVT(f, 32, f, 64, GREX_ELEMENTS(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  /* Integer → floating-point */ \
  /* f64 */ \
  GREX_CVT(f, 64, i, 64, GREX_ELEMENTS(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  GREX_CVT(f, 64, u, 64, GREX_ELEMENTS(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  GREX_CVT(f, 64, i, 32, GREX_ELEMENTS(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  GREX_CVT(f, 64, u, 32, GREX_ELEMENTS(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  GREX_CVT(f, 64, i, 16, GREX_ELEMENTS(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  GREX_CVT(f, 64, u, 16, GREX_ELEMENTS(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  GREX_CVT(f, 64, i, 8, GREX_ELEMENTS(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  GREX_CVT(f, 64, u, 8, GREX_ELEMENTS(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  /* f32 */ \
  GREX_CVT(f, 32, i, 64, GREX_ELEMENTS(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  GREX_CVT(f, 32, u, 64, GREX_ELEMENTS(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  GREX_CVT(f, 32, i, 32, GREX_ELEMENTS(REGISTERBITS, 32), BITPREFIX, REGISTERBITS) \
  GREX_CVT(f, 32, u, 32, GREX_ELEMENTS(REGISTERBITS, 32), BITPREFIX, REGISTERBITS) \
  GREX_CVT(f, 32, i, 16, GREX_ELEMENTS(REGISTERBITS, 32), BITPREFIX, REGISTERBITS) \
  GREX_CVT(f, 32, u, 16, GREX_ELEMENTS(REGISTERBITS, 32), BITPREFIX, REGISTERBITS) \
  GREX_CVT(f, 32, i, 8, GREX_ELEMENTS(REGISTERBITS, 32), BITPREFIX, REGISTERBITS) \
  GREX_CVT(f, 32, u, 8, GREX_ELEMENTS(REGISTERBITS, 32), BITPREFIX, REGISTERBITS) \
  /* Floating-point → integer */ \
  /* f64 */ \
  GREX_CVT(i, 64, f, 64, GREX_ELEMENTS(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  GREX_CVT(u, 64, f, 64, GREX_ELEMENTS(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  GREX_CVT(i, 32, f, 64, GREX_ELEMENTS(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  GREX_CVT(u, 32, f, 64, GREX_ELEMENTS(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  GREX_CVT(i, 16, f, 64, GREX_ELEMENTS(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  GREX_CVT(u, 16, f, 64, GREX_ELEMENTS(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  GREX_CVT(i, 8, f, 64, GREX_ELEMENTS(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  GREX_CVT(u, 8, f, 64, GREX_ELEMENTS(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  /* f32 */ \
  GREX_CVT(i, 64, f, 32, GREX_ELEMENTS(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  GREX_CVT(u, 64, f, 32, GREX_ELEMENTS(REGISTERBITS, 64), BITPREFIX, REGISTERBITS) \
  GREX_CVT(i, 32, f, 32, GREX_ELEMENTS(REGISTERBITS, 32), BITPREFIX, REGISTERBITS) \
  GREX_CVT(u, 32, f, 32, GREX_ELEMENTS(REGISTERBITS, 32), BITPREFIX, REGISTERBITS) \
  GREX_CVT(i, 16, f, 32, GREX_ELEMENTS(REGISTERBITS, 32), BITPREFIX, REGISTERBITS) \
  GREX_CVT(u, 16, f, 32, GREX_ELEMENTS(REGISTERBITS, 32), BITPREFIX, REGISTERBITS) \
  GREX_CVT(i, 8, f, 32, GREX_ELEMENTS(REGISTERBITS, 32), BITPREFIX, REGISTERBITS) \
  GREX_CVT(u, 8, f, 32, GREX_ELEMENTS(REGISTERBITS, 32), BITPREFIX, REGISTERBITS)

// The same type: no-op
template<Vectorizable T, std::size_t tSize>
inline Vector<T, tSize> convert(Vector<T, tSize> v, TypeTag<T> /*tag*/) {
  return v;
}
// Integers with the same number of bits: No-op
template<Vectorizable TDst, Vectorizable TSrc, std::size_t tSize>
requires(std::integral<TDst> && std::integral<TSrc> && sizeof(TDst) == sizeof(TSrc))
inline Vector<TDst, tSize> convert(Vector<TSrc, tSize> v, TypeTag<TDst> /*tag*/) {
  return {.r = v.r};
}
// Converting to a larger integer with different signedness: Increase size while keeping signedness
template<Vectorizable TDst, Vectorizable TSrc, std::size_t tSize>
requires(std::integral<TDst> && std::integral<TSrc> &&
         (std::unsigned_integral<TDst> != std::unsigned_integral<TSrc>) && is_native<TDst, tSize> &&
         sizeof(TDst) > sizeof(TSrc))
inline Vector<TDst, tSize> convert(Vector<TSrc, tSize> v, TypeTag<TDst> /*tag*/) {
  using Temp = std::conditional_t<std::unsigned_integral<TSrc>, UnsignedOf<TDst>, SignedOf<TDst>>;
  const auto s = convert(v, type_tag<Temp>);
  return {.r = s.r};
}
template<Vectorizable TDst, Vectorizable TSrc, std::size_t tPart, std::size_t tSize>
requires(std::integral<TDst> && std::integral<TSrc> &&
         (std::unsigned_integral<TDst> != std::unsigned_integral<TSrc>) &&
         sizeof(TDst) > sizeof(TSrc))
inline VectorFor<TDst, tPart> convert(SubVector<TSrc, tPart, tSize> v, TypeTag<TDst> /*tag*/) {
  using Temp = std::conditional_t<std::unsigned_integral<TSrc>, UnsignedOf<TDst>, SignedOf<TDst>>;
  const auto s = convert(v, type_tag<Temp>);
  // The output can be native or super-native (sub-native to sub-native is handled elsewhere)
  if constexpr (is_supernative<TDst, tPart>) {
    // TODO Is this too restrictive?
    return {.lower = {.r = s.lower.r}, .upper = {.r = s.upper.r}};
  } else {
    return {.r = s.r};
  }
}
// Converting to a smaller integer: Truncation → ignore signedness
template<Vectorizable TDst, Vectorizable TSrc, std::size_t tSize>
requires(std::integral<TDst> && std::integral<TSrc> &&
         (!std::unsigned_integral<TDst> || !std::unsigned_integral<TSrc>) &&
         sizeof(TDst) < sizeof(TSrc))
inline VectorFor<TDst, tSize> convert(Vector<TSrc, tSize> v, TypeTag<TDst> /*tag*/) {
  const auto s = convert(Vector<UnsignedOf<TSrc>, tSize>{v.r}, type_tag<UnsignedOf<TDst>>);
  return VectorFor<TDst, tSize>{s.registr()};
}
template<Vectorizable TDst, Vectorizable TSrc, std::size_t tPart, std::size_t tSize>
requires(std::integral<TDst> && std::integral<TSrc> &&
         (!std::unsigned_integral<TDst> || !std::unsigned_integral<TSrc>) &&
         sizeof(TDst) < sizeof(TSrc))
inline VectorFor<TDst, tPart> convert(SubVector<TSrc, tPart, tSize> v, TypeTag<TDst> /*tag*/) {
  const auto s =
    convert(SubVector<UnsignedOf<TSrc>, tPart, tSize>{v.registr()}, type_tag<UnsignedOf<TDst>>);
  return VectorFor<TDst, tPart>{s.registr()};
}
// Baseline for converting sub-native vectors: Use the smaller of the two native sizes if the output
template<Vectorizable TDst, Vectorizable TSrc, std::size_t tPart, std::size_t tSize>
inline VectorFor<TDst, tPart> convert(SubVector<TSrc, tPart, tSize> v, TypeTag<TDst> /*tag*/) {
  using Out = VectorFor<TDst, tPart>;
  if constexpr (is_subnative<TDst, tPart>) {
    constexpr std::size_t work_size = std::min(tSize, Out::Full::size);
    static_assert(work_size >= tPart);
    const auto s = convert(VectorFor<TSrc, work_size>{v.registr()}, type_tag<TDst>);
    return Out{s.registr()};
  } else {
    static_assert(is_supernative<TDst, tPart>);
    return Out{
      .lower = convert(split(v, index_tag<0>), type_tag<TDst>),
      .upper = convert(split(v, index_tag<1>), type_tag<TDst>),
    };
  }
}

// Conversion between super-native vectors: Work on the halves separately
template<typename TDst, typename THalf>
requires(is_supernative<TDst, THalf::size * 2>)
inline VectorFor<TDst, THalf::size * 2> convert(SuperVector<THalf> v, TypeTag<TDst> /*tag*/) {
  return {.lower = convert(v.lower, type_tag<TDst>), .upper = convert(v.upper, type_tag<TDst>)};
}
// From native to super-native: Split into parts and convert separately
template<typename TDst, typename TSrc, std::size_t tSize>
requires(is_supernative<TDst, tSize>)
inline VectorFor<TDst, tSize> convert(Vector<TSrc, tSize> v, TypeTag<TDst> /*tag*/) {
  return {
    .lower = convert(split(v, index_tag<0>), type_tag<TDst>),
    .upper = convert(split(v, index_tag<1>), type_tag<TDst>),
  };
}
// From super-native to native: Convert parts separately and merge
template<typename TDst, typename THalf>
requires(!is_supernative<TDst, THalf::size * 2>)
inline VectorFor<TDst, THalf::size * 2> convert(SuperVector<THalf> v, TypeTag<TDst> /*tag*/) {
  return merge(convert(v.lower, type_tag<TDst>), convert(v.upper, type_tag<TDst>));
}
// TODO Separate implementations for four-fold and eight-fold super-native vectors?!
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_CONVERT_BASE_HPP
