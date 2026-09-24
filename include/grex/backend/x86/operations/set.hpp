// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_SET_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_SET_HPP

#include <concepts>
#include <cstddef>

#include <immintrin.h>

#include "grex/backend/base.hpp"
#include "grex/backend/choosers.hpp"
#include "grex/backend/macros/base.hpp"
#include "grex/backend/macros/cast.hpp"
#include "grex/backend/macros/conditional.hpp"
#include "grex/backend/macros/for-each.hpp"
#include "grex/backend/macros/math.hpp"
#include "grex/backend/macros/repeat.hpp"
#include "grex/backend/macros/types.hpp"
#include "grex/backend/x86/instruction-sets.hpp"
#include "grex/backend/x86/macros/for-each.hpp"
#include "grex/backend/x86/macros/intrinsics.hpp"
#include "grex/backend/x86/operations/expand.hpp"
#include "grex/backend/x86/operations/merge.hpp"
#include "grex/backend/x86/sizes.hpp"
#include "grex/backend/x86/types.hpp"
#include "grex/base.hpp"

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

// Helpers to define function arguments for the set-based operations
#define GREX_SET_ARG(CNT, IDX, TYPE) GREX_COMMA_IF(IDX) TYPE v##IDX
#define GREX_SET_VAR(CNT, IDX) GREX_COMMA_IF(IDX) v##IDX
#define GREX_SET_VAL(CNT, IDX, KIND, BITS) GREX_SIGNED_CAST(KIND, BITS, v##IDX) GREX_COMMA_IF(IDX)
#define GREX_SET_NEGVAL(CNT, IDX, BITS) GREX_COMMA_IF(IDX) GREX_OPCAST(i, BITS, -i##BITS(v##IDX))

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
  inline NativeMask<KIND##BITS, SIZE> zeros(TypeTag<NativeMask<KIND##BITS, SIZE>>) { \
    return {.r = 0}; \
  } \
  inline NativeMask<KIND##BITS, SIZE> ones(TypeTag<NativeMask<KIND##BITS, SIZE>>) { \
    return {.r = GREX_CAT(GREX_ONEMASK_, SIZE)}; \
  } \
  inline NativeMask<KIND##BITS, SIZE> broadcast(bool value, \
                                                TypeTag<NativeMask<KIND##BITS, SIZE>>) { \
    return {.r = GREX_MMASK(SIZE)(-GREX_CAT(i, GREX_MAX(SIZE, 8))(value))}; \
  } \
  inline NativeMask<KIND##BITS, SIZE> set(TypeTag<NativeMask<KIND##BITS, SIZE>>, \
                                          GREX_REPEAT(SIZE, GREX_SET_ARG, bool)) { \
    return {.r = GREX_MMASK_CAST(SIZE, GREX_CMASK_SET(SIZE, GREX_CAT(u, GREX_MAX(SIZE, 8))))}; \
  }

#define GREX_SUBSET_MASK(KIND, BITS, PART, SIZE) \
  inline SubMask<KIND##BITS, PART> set(TypeTag<SubMask<KIND##BITS, PART>>, \
                                       GREX_REPEAT(PART, GREX_SET_ARG, bool)) { \
    const auto r = GREX_MMASK_CAST(SIZE, GREX_CMASK_SET(PART, GREX_CAT(u, GREX_MAX(SIZE, 8)))); \
    return SubMask<KIND##BITS, PART>{r}; \
  }
GREX_FOREACH_SUB_EXT(GREX_SUBSET_MASK)
#else
#define GREX_SET_MASK(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  inline NativeMask<KIND##BITS, SIZE> zeros(TypeTag<NativeMask<KIND##BITS, SIZE>>) { \
    return {.r = BITPREFIX##_setzero_si##REGISTERBITS()}; \
  } \
  inline NativeMask<KIND##BITS, SIZE> ones(TypeTag<NativeMask<KIND##BITS, SIZE>>) { \
    return {.r = BITPREFIX##_set1_epi32(-1)}; \
  } \
  inline NativeMask<KIND##BITS, SIZE> broadcast(bool value, \
                                                TypeTag<NativeMask<KIND##BITS, SIZE>>) { \
    const i##BITS entry = GREX_OPCAST(i, BITS, -i##BITS(value)); \
    return {.r = GREX_CAT(BITPREFIX##_set1_, GREX_SET_EPI(BITS, REGISTERBITS))(entry)}; \
  } \
  inline NativeMask<KIND##BITS, SIZE> set(TypeTag<NativeMask<KIND##BITS, SIZE>>, \
                                          GREX_REPEAT(SIZE, GREX_SET_ARG, bool)) { \
    using V = NativeVector<i##BITS, SIZE>; \
    return {.r = set(type_tag<V>, GREX_REPEAT(SIZE, GREX_SET_NEGVAL, BITS)).r}; \
  }

template<Vectorizable T, std::size_t N, typename... Ts>
inline SubMask<T, N> set(TypeTag<SubMask<T, N>> /*tag*/, Ts... values) {
  using SV = SignedInt<sizeof(T)>;
  const auto r = set(type_tag<SubVector<SV, N>>, -SV(values)...).registr();
  return SubMask<T, N>{r};
}
#endif

// Define vector operations
#define GREX_SET_VEC(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  inline NativeVector<KIND##BITS, SIZE> zeros(TypeTag<NativeVector<KIND##BITS, SIZE>>) { \
    return {.r = GREX_CAT(BITPREFIX##_setzero_, GREX_SI_SUFFIX(KIND, BITS, REGISTERBITS))()}; \
  } \
  inline NativeVector<KIND##BITS, SIZE> undefined(TypeTag<NativeVector<KIND##BITS, SIZE>>) { \
    return {.r = GREX_UNDEF(KIND, BITS, BITPREFIX, REGISTERBITS)()}; \
  } \
  inline NativeVector<KIND##BITS, SIZE> broadcast(KIND##BITS value, \
                                                  TypeTag<NativeVector<KIND##BITS, SIZE>>) { \
    return {.r = GREX_CAT(BITPREFIX##_set1_, GREX_SET_SUFFIX(KIND, BITS, REGISTERBITS))( \
              GREX_SIGNED_CAST(KIND, BITS, value))}; \
  } \
  inline NativeVector<KIND##BITS, SIZE> set(TypeTag<NativeVector<KIND##BITS, SIZE>>, \
                                            GREX_REPEAT(SIZE, GREX_SET_ARG, KIND##BITS)) { \
    return {.r = GREX_CAT(BITPREFIX##_set_, GREX_SET_SUFFIX(KIND, BITS, REGISTERBITS))( \
              GREX_RREPEAT(SIZE, GREX_SET_VAL, KIND, BITS))}; \
  }

// Binary16 vectors share `u16`’s register, making the `ph` set intrinsics a poor fit, but binary16
// masks are plain bit patterns and are covered by the generic definitions above.
#define GREX_SET_ALL(REGISTERBITS, BITPREFIX) \
  GREX_FOREACH_TYPE(GREX_SET_VEC, REGISTERBITS, BITPREFIX, REGISTERBITS) \
  GREX_FOREACH_TYPE_EXT(GREX_SET_MASK, REGISTERBITS, BITPREFIX, REGISTERBITS)
GREX_FOREACH_X86_64_LEVEL(GREX_SET_ALL)

//==================================================================================================
// Sub-native set
//==================================================================================================

//--------------------------------------------------------------------------------------------------
// Two values, the base case of the unpack tree below
//--------------------------------------------------------------------------------------------------

// The first value is expanded into the lowest lane and the second one is interleaved
// (floating-point, 8-bit and 32-bit integers on level 1) or inserted (otherwise).
#define GREX_SUBSET2_HEAD(KIND, BITS) \
  inline SubVector<KIND##BITS, 2> set(TypeTag<SubVector<KIND##BITS, 2>>, KIND##BITS v0, \
                                      KIND##BITS v1)
#define GREX_SUBSET2_EXPAND(KIND, BITS, IDX) \
  expand_any(v##IDX, index_tag<min_native_size<KIND##BITS>>).r
#define GREX_SUBSET2_INSERT(KIND, BITS) \
  GREX_SUBSET2_HEAD(KIND, BITS) { \
    const auto vv0 = GREX_SUBSET2_EXPAND(KIND, BITS, 0); \
    const auto vv1 = GREX_KINDCAST_SINGLE(KIND, i, BITS, v1); \
    return SubVector<KIND##BITS, 2>{_mm_insert_epi##BITS(vv0, vv1, 1)}; \
  }
// The expansions are bound to variables to fix their order: They are opaque to the compiler, so
// leaving the order to the call, which GCC evaluates right-to-left, costs a register copy.
#define GREX_SUBSET2_UNPACK(KIND, BITS) \
  GREX_SUBSET2_HEAD(KIND, BITS) { \
    const auto vv0 = GREX_SUBSET2_EXPAND(KIND, BITS, 0); \
    const auto vv1 = GREX_SUBSET2_EXPAND(KIND, BITS, 1); \
    return SubVector<KIND##BITS, 2>{ \
      GREX_CAT(_mm_unpacklo_, GREX_EPI_SUFFIX(GREX_REGKIND(KIND, BITS), BITS))(vv0, vv1)}; \
  }
#define GREX_SUBSET2_INT(BITS, IMPL) \
  IMPL(i, BITS) \
  IMPL(u, BITS)

GREX_SUBSET2_INT(16, GREX_SUBSET2_INSERT)
#if GREX_X86_64_LEVEL >= 2
GREX_SUBSET2_INT(8, GREX_SUBSET2_INSERT)
GREX_SUBSET2_INT(32, GREX_SUBSET2_INSERT)
#else
GREX_SUBSET2_INT(8, GREX_SUBSET2_UNPACK)
GREX_SUBSET2_INT(32, GREX_SUBSET2_UNPACK)
#endif
GREX_SUBSET2_UNPACK(f, 16)
GREX_SUBSET2_UNPACK(f, 32)

//--------------------------------------------------------------------------------------------------
// More than two values, which are combined by an unpack tree
//--------------------------------------------------------------------------------------------------

/** The `I`-th value of a pack, which is resolved entirely at compile time. */
template<std::size_t I, typename T, typename... Ts>
GREX_ALWAYS_INLINE inline T pack_value(T value, Ts... rest) {
  if constexpr (I == 0) {
    return value;
  } else {
    return pack_value<I - 1>(rest...);
  }
}

/**
 * The `N` values starting at index `Begin`, combined into a vector by an unpack tree: The two
 * halves are built recursively and interleaved by `merge`, which uses a `punpckl` instruction up to
 * the native size and `vinserti128`/`vinserti64x4` beyond it.
 */
template<std::size_t Begin, std::size_t N, Vectorizable T, std::same_as<T>... Ts>
requires(N >= 2 && sizeof...(Ts) + 1 >= Begin + N)
inline VectorFor<T, N> set_part(T value, Ts... rest) {
  if constexpr (N == 2) {
    return set(type_tag<SubVector<T, 2>>, pack_value<Begin>(value, rest...),
               pack_value<Begin + 1>(value, rest...));
  } else {
    constexpr std::size_t half = N / 2;
    return merge(set_part<Begin, half>(value, rest...),
                 set_part<Begin + half, half>(value, rest...));
  }
}

#define GREX_SUBSET_TREE(KIND, BITS, PART) \
  inline SubVector<KIND##BITS, PART> set(TypeTag<SubVector<KIND##BITS, PART>>, \
                                         GREX_REPEAT(PART, GREX_SET_ARG, KIND##BITS)) { \
    return set_part<0, PART>(GREX_REPEAT(PART, GREX_SET_VAR)); \
  }
#define GREX_SUBSET_TREE_INT(BITS, PART) \
  GREX_SUBSET_TREE(i, BITS, PART) \
  GREX_SUBSET_TREE(u, BITS, PART)

GREX_SUBSET_TREE_INT(16, 4)
GREX_SUBSET_TREE_INT(8, 4)
GREX_SUBSET_TREE_INT(8, 8)
GREX_SUBSET_TREE(f, 16, 4)

//==================================================================================================
// Binary16, whose values the ABI passes in vector registers
//==================================================================================================

#define GREX_SET_F16(REGISTERBITS, BITPREFIX) \
  inline NativeVector<f16, GREX_DIVIDE(REGISTERBITS, 16)> set( \
    TypeTag<NativeVector<f16, GREX_DIVIDE(REGISTERBITS, 16)>>, \
    GREX_REPEAT(GREX_DIVIDE(REGISTERBITS, 16), GREX_SET_ARG, f16)) { \
    return set_part<0, GREX_DIVIDE(REGISTERBITS, 16)>( \
      GREX_REPEAT(GREX_DIVIDE(REGISTERBITS, 16), GREX_SET_VAR)); \
  }
GREX_FOREACH_X86_64_LEVEL(GREX_SET_F16)

#if GREX_X86_64_LEVEL >= 3
#define GREX_BROADCAST_F16(REGISTERBITS, BITPREFIX) \
  inline NativeVector<f16, GREX_DIVIDE(REGISTERBITS, 16)> broadcast( \
    f16 value, TypeTag<NativeVector<f16, GREX_DIVIDE(REGISTERBITS, 16)>>) { \
    return {.r = BITPREFIX##_broadcastw_epi16(expand_any(value, index_tag<8>).r)}; \
  }
GREX_FOREACH_X86_64_LEVEL(GREX_BROADCAST_F16)
#else
// Without AVX2, there is no `vpbroadcastw` reading from a vector register, so the lowest lane is
// splatted across the low 64 bits, which are then duplicated into the upper half.
inline NativeVector<f16, 8> broadcast(f16 value, TypeTag<NativeVector<f16, 8>> /*tag*/) {
  const __m128i low = _mm_shufflelo_epi16(expand_any(value, index_tag<8>).r, 0);
  return {.r = _mm_unpacklo_epi64(low, low)};
}
#endif
} // namespace grex::backend

#include "grex/backend/shared/operations/set.hpp" // IWYU pragma: export

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_SET_HPP
