// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_SET_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_SET_HPP

#include <array>
#include <bit>
#include <cstddef>

#include <immintrin.h>

#include "grex/backend/defs.hpp"
#include "grex/backend/x86/instruction-sets.hpp"
#include "grex/backend/x86/macros/base.hpp"
#include "grex/backend/x86/macros/conditional.hpp"
#include "grex/backend/x86/macros/for-each.hpp"
#include "grex/backend/x86/macros/intrinsics.hpp"
#include "grex/backend/x86/macros/repeat.hpp"
#include "grex/backend/x86/types.hpp"
#include "grex/base/defs.hpp"

namespace grex::backend {
// Define the very messy suffixes used by the set intrinsics
#define GREX_SET_EPI64_128 epi64x
#define GREX_SET_EPI64_256 epi64x
#define GREX_SET_EPI64_512 epi64
#define GREX_SET_EPI_8(REGISTERBITS) epi8
#define GREX_SET_EPI_16(REGISTERBITS) epi16
#define GREX_SET_EPI_32(REGISTERBITS) epi32
#define GREX_SET_EPI_64(REGISTERBITS) GREX_SET_EPI64_##REGISTERBITS
#define GREX_SET_EPI(BITS, REGISTERBITS) GREX_SET_EPI_##BITS(REGISTERBITS)
#define GREX_SET_SUFFIX_f(BITS, REGISTERBITS) GREX_FP_SUFFIX(BITS)
#define GREX_SET_SUFFIX_i(BITS, REGISTERBITS) GREX_SET_EPI(BITS, REGISTERBITS)
#define GREX_SET_SUFFIX_u(BITS, REGISTERBITS) GREX_SET_EPI(BITS, REGISTERBITS)
#define GREX_SET_SUFFIX(KIND, BITS, REGISTERBITS) GREX_SET_SUFFIX_##KIND(BITS, REGISTERBITS)
// Define the casts to the argument type of the intrinsics
#define GREX_SET_CAST_f(BITS, X) X
#define GREX_SET_CAST_i(BITS, X) X
#define GREX_SET_CAST_u(BITS, X) i##BITS(X)
#define GREX_SET_CAST(KIND, BITS, X) GREX_SET_CAST_##KIND(BITS, X)
// Helpers to define function arguments for the set-based operations
#define GREX_SET_ARG(CNT, IDX, TYPE) GREX_COMMA_IF(IDX) TYPE v##IDX
#define GREX_SET_VAL(CNT, IDX, KIND, BITS) GREX_SET_CAST(KIND, BITS, v##IDX) GREX_COMMA_IF(IDX)
#define GREX_SET_NEGVAL(CNT, IDX, BITS) -i##BITS(v##IDX) GREX_COMMA_IF(IDX)
// Define the messy undefined macros
#define GREX_UNDEF_BASE(KIND, BITS, BITPREFIX, REGISTERBITS) \
  GREX_CAT(BITPREFIX##_setzero_, GREX_SI_SUFFIX(KIND, BITS, REGISTERBITS))
#define GREX_UNDEF_I128 _mm_undefined_si128
#define GREX_UNDEF_I256 _mm256_undefined_si256
#define GREX_UNDEF_I512 _mm512_undefined_epi32
#define GREX_UNDEF_INT(KIND, BITS, BITPREFIX, REGISTERBITS) GREX_UNDEF_I##REGISTERBITS
#define GREX_UNDEF_f GREX_UNDEF_BASE
#define GREX_UNDEF_i GREX_UNDEF_INT
#define GREX_UNDEF_u GREX_UNDEF_INT
#define GREX_UNDEF(KIND, ...) GREX_UNDEF_##KIND(KIND, __VA_ARGS__)

#define GREX_CMASK_SET_OP(CNT, IDX, TYPE) \
  GREX_IF(IDX, |, GREX_EMPTY()) \
  GREX_IF(IDX, (TYPE(v##IDX) << IDX##U), TYPE(v##IDX))
#define GREX_CMASK_SET(SIZE, TYPE) GREX_REPEAT(SIZE, GREX_CMASK_SET_OP, TYPE)

#define GREX_ONEMASK_2 0x3
#define GREX_ONEMASK_4 0xF
#define GREX_ONEMASK_8 0xFF
#define GREX_ONEMASK_16 0xFFFF
#define GREX_ONEMASK_32 0xFFFFFFFF
#define GREX_ONEMASK_64 0xFFFFFFFFFFFFFFFF

// Define mask operations, which can be applied to compressed or broad masks
#if GREX_X86_64_LEVEL >= 4
#define GREX_SET_MASK(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  inline Mask<KIND##BITS, SIZE> zeros(TypeTag<Mask<KIND##BITS, SIZE>>) { \
    return {.r = GREX_MMASK(SIZE){0}}; \
  } \
  inline Mask<KIND##BITS, SIZE> ones(TypeTag<Mask<KIND##BITS, SIZE>>) { \
    return {.r = GREX_MMASK(SIZE)(GREX_ONEMASK_##SIZE)}; \
  } \
  inline Mask<KIND##BITS, SIZE> broadcast(bool value, TypeTag<Mask<KIND##BITS, SIZE>>) { \
    return {.r = GREX_MMASK(SIZE)(-GREX_CAT(i, GREX_MAX(SIZE, 8))(value))}; \
  } \
  inline Mask<KIND##BITS, SIZE> set(TypeTag<Mask<KIND##BITS, SIZE>>, \
                                    GREX_REPEAT(SIZE, GREX_SET_ARG, bool)) { \
    return {.r = GREX_MMASK(SIZE)(GREX_CMASK_SET(SIZE, GREX_CAT(u, GREX_MAX(SIZE, 8))))}; \
  }
#else
#define GREX_SET_MASK(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  inline Mask<KIND##BITS, SIZE> zeros(TypeTag<Mask<KIND##BITS, SIZE>>) { \
    return {.r = BITPREFIX##_setzero_si##REGISTERBITS()}; \
  } \
  inline Mask<KIND##BITS, SIZE> ones(TypeTag<Mask<KIND##BITS, SIZE>>) { \
    return {.r = BITPREFIX##_set1_epi32(-1)}; \
  } \
  inline Mask<KIND##BITS, SIZE> broadcast(bool value, TypeTag<Mask<KIND##BITS, SIZE>>) { \
    return {.r = GREX_CAT(BITPREFIX##_set1_, GREX_SET_EPI(BITS, REGISTERBITS))(-i##BITS(value))}; \
  } \
  inline Mask<KIND##BITS, SIZE> set(TypeTag<Mask<KIND##BITS, SIZE>>, \
                                    GREX_REPEAT(SIZE, GREX_SET_ARG, bool)) { \
    return {.r = GREX_CAT(BITPREFIX##_set_, GREX_SET_EPI(BITS, REGISTERBITS))( \
              GREX_RREPEAT(SIZE, GREX_SET_NEGVAL, BITS))}; \
  }
#endif

// Define vector operations
#define GREX_SET_VEC(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  inline Vector<KIND##BITS, SIZE> zeros(TypeTag<Vector<KIND##BITS, SIZE>>) { \
    return {.r = GREX_CAT(BITPREFIX##_setzero_, GREX_SI_SUFFIX(KIND, BITS, REGISTERBITS))()}; \
  } \
  inline Vector<KIND##BITS, SIZE> undefined(TypeTag<Vector<KIND##BITS, SIZE>>) { \
    return {.r = GREX_UNDEF(KIND, BITS, BITPREFIX, REGISTERBITS)()}; \
  } \
  inline Vector<KIND##BITS, SIZE> broadcast(KIND##BITS value, TypeTag<Vector<KIND##BITS, SIZE>>) { \
    return {.r = GREX_CAT(BITPREFIX##_set1_, GREX_SET_SUFFIX(KIND, BITS, REGISTERBITS))( \
              GREX_SET_CAST(KIND, BITS, value))}; \
  } \
  inline Vector<KIND##BITS, SIZE> set(TypeTag<Vector<KIND##BITS, SIZE>>, \
                                      GREX_REPEAT(SIZE, GREX_SET_ARG, KIND##BITS)) { \
    return {.r = GREX_CAT(BITPREFIX##_set_, GREX_SET_SUFFIX(KIND, BITS, REGISTERBITS))( \
              GREX_RREPEAT(SIZE, GREX_SET_VAL, KIND, BITS))}; \
  }

#define GREX_SET(...) GREX_SET_VEC(__VA_ARGS__) GREX_SET_MASK(__VA_ARGS__)
#define GREX_SET_ALL(REGISTERBITS, BITPREFIX) \
  GREX_FOREACH_TYPE(GREX_SET, REGISTERBITS, BITPREFIX, REGISTERBITS)
GREX_FOREACH_X86_64_LEVEL(GREX_SET_ALL)

template<Vectorizable T, std::size_t tSize>
inline Vector<T, tSize> indices(TypeTag<Vector<T, tSize>> /*tag*/) {
  return static_apply<tSize>(
    []<std::size_t... tIdxs>() { return set(type_tag<Vector<T, tSize>>, T(tIdxs)...); });
}

// SubVector
template<Vectorizable T, std::size_t tPart, std::size_t tSize, typename... Ts>
inline SubVector<T, tPart, tSize> zeros(TypeTag<SubVector<T, tPart, tSize>> /*tag*/) {
  return SubVector<T, tPart, tSize>{zeros(type_tag<Vector<T, tSize>>)};
}
template<Vectorizable T, std::size_t tPart, std::size_t tSize, typename... Ts>
inline SubVector<T, tPart, tSize> undefined(TypeTag<SubVector<T, tPart, tSize>> /*tag*/) {
  return SubVector<T, tPart, tSize>{undefined(type_tag<Vector<T, tSize>>)};
}
template<Vectorizable T, std::size_t tPart, std::size_t tSize, typename... Ts>
inline SubVector<T, tPart, tSize> broadcast(T value, TypeTag<SubVector<T, tPart, tSize>> /*tag*/) {
  return SubVector<T, tPart, tSize>{broadcast(value, type_tag<Vector<T, tSize>>)};
}
template<Vectorizable T, std::size_t tPart, std::size_t tSize, typename... Ts>
inline SubVector<T, tPart, tSize> set(TypeTag<SubVector<T, tPart, tSize>> /*tag*/, Ts... values) {
  std::array<T, tSize> elements{values...};
  const auto full = static_apply<tSize>([&]<std::size_t... tIdxs> {
    return set(type_tag<Vector<T, tSize>>, std::get<tIdxs>(elements)...);
  });
  return SubVector<T, tPart, tSize>{full};
}
template<Vectorizable T, std::size_t tPart, std::size_t tSize, typename... Ts>
inline SubVector<T, tPart, tSize> indices(TypeTag<SubVector<T, tPart, tSize>> /*tag*/) {
  return SubVector<T, tPart, tSize>{indices(type_tag<Vector<T, tSize>>)};
}

// SubMask
template<Vectorizable T, std::size_t tPart, std::size_t tSize>
inline SubMask<T, tPart, tSize> zeros(TypeTag<SubMask<T, tPart, tSize>> /*tag*/) {
  return SubMask<T, tPart, tSize>{zeros(type_tag<Mask<T, tSize>>)};
}
template<Vectorizable T, std::size_t tPart, std::size_t tSize>
inline SubMask<T, tPart, tSize> ones(TypeTag<SubMask<T, tPart, tSize>> /*tag*/) {
  return SubMask<T, tPart, tSize>{ones(type_tag<Mask<T, tSize>>)};
}
template<Vectorizable T, std::size_t tPart, std::size_t tSize, typename... Ts>
inline SubMask<T, tPart, tSize> broadcast(bool value, TypeTag<SubMask<T, tPart, tSize>> /*tag*/) {
  return SubMask<T, tPart, tSize>{broadcast(value, type_tag<Mask<T, tSize>>)};
}
template<Vectorizable T, std::size_t tPart, std::size_t tSize, typename... Ts>
inline SubMask<T, tPart, tSize> set(TypeTag<SubMask<T, tPart, tSize>> /*tag*/, Ts... values) {
  std::array<bool, tSize> buf{values...};
  const auto full = static_apply<tSize>(
    [&]<std::size_t... tIdxs>() { return set(type_tag<Mask<T, tSize>>, buf[tIdxs]...); });
  return SubMask<T, tPart, tSize>{full};
}

// SuperVector
template<typename THalf>
inline SuperVector<THalf> zeros(TypeTag<SuperVector<THalf>> /*tag*/) {
  const auto half = zeros(type_tag<THalf>);
  return {.lower = half, .upper = half};
}
template<typename THalf>
inline SuperVector<THalf> undefined(TypeTag<SuperVector<THalf>> /*tag*/) {
  const auto half = undefined(type_tag<THalf>);
  return {.lower = half, .upper = half};
}
template<typename THalf>
inline SuperVector<THalf> broadcast(typename THalf::Value value,
                                    TypeTag<SuperVector<THalf>> /*tag*/) {
  const auto half = broadcast(value, type_tag<THalf>);
  return {.lower = half, .upper = half};
}
template<typename THalf, typename... Ts>
requires(sizeof...(Ts) == 2 * THalf::size && std::has_single_bit(sizeof...(Ts)))
inline SuperVector<THalf> set(TypeTag<SuperVector<THalf>> /*tag*/, Ts... values) {
  constexpr std::size_t size = sizeof...(Ts);
  const std::array buf{values...};
  auto op = [&]<std::size_t... tIdxs>() { return set(type_tag<THalf>, std::get<tIdxs>(buf)...); };
  return {.lower = static_apply<0, size / 2>(op), .upper = static_apply<size / 2, size>(op)};
}
template<typename THalf>
inline SuperVector<THalf> indices(TypeTag<SuperVector<THalf>> /*tag*/) {
  using Vec = SuperVector<THalf>;
  using Value = Vec::Value;
  constexpr std::size_t size = Vec::size;
  auto op = []<std::size_t... tIdxs>() { return set(type_tag<THalf>, Value(tIdxs)...); };
  return {.lower = static_apply<0, size / 2>(op), .upper = static_apply<size / 2, size>(op)};
}

// SuperMask
template<typename THalf>
inline SuperMask<THalf> zeros(TypeTag<SuperMask<THalf>> /*tag*/) {
  const auto half = zeros(type_tag<THalf>);
  return {.lower = half, .upper = half};
}
template<typename THalf>
inline SuperMask<THalf> ones(TypeTag<SuperMask<THalf>> /*tag*/) {
  const auto half = ones(type_tag<THalf>);
  return {.lower = half, .upper = half};
}
template<typename THalf>
inline SuperMask<THalf> broadcast(bool value, TypeTag<SuperMask<THalf>> /*tag*/) {
  const auto half = broadcast(value, type_tag<THalf>);
  return {.lower = half, .upper = half};
}
template<typename THalf, typename... Ts>
requires(sizeof...(Ts) == 2 * THalf::size && std::has_single_bit(sizeof...(Ts)))
inline SuperMask<THalf> set(TypeTag<SuperMask<THalf>> /*tag*/, Ts... values) {
  constexpr std::size_t size = sizeof...(Ts);
  const std::array buf{values...};
  auto op = [&]<std::size_t... tIdxs>() { return set(type_tag<THalf>, std::get<tIdxs>(buf)...); };
  return {.lower = static_apply<0, size / 2>(op), .upper = static_apply<size / 2, size>(op)};
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_SET_HPP
