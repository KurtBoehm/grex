// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_SET_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_SET_HPP

#include <immintrin.h>

#include "thesauros/types.hpp"

#include "grex/backend/x86/helpers.hpp"
#include "grex/backend/x86/instruction-sets.hpp"
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
#define GREX_SET_SUFFIX_f(BITS, REGISTERBITS) GREX_FP_SUFFIX(f##BITS)
#define GREX_SET_SUFFIX_i(BITS, REGISTERBITS) GREX_SET_EPI_##BITS(REGISTERBITS)
#define GREX_SET_SUFFIX_u(BITS, REGISTERBITS) GREX_SET_EPI_##BITS(REGISTERBITS)
#define GREX_SET_SUFFIX(KIND, BITS, REGISTERBITS) GREX_SET_SUFFIX_##KIND(BITS, REGISTERBITS)
// Helpers to define function arguments for the set-based operations
#define GREX_SET_ARG(CNT, IDX, TYPE) BOOST_PP_COMMA_IF(IDX) TYPE v##IDX
#define GREX_SET_VAL(CNT, IDX, SIZE) \
  BOOST_PP_COMMA_IF(IDX) BOOST_PP_CAT(v, BOOST_PP_SUB(SIZE, BOOST_PP_INC(IDX)))
#define GREX_SET_NEGVAL_IMPLI(CNT, IDX, BITS, SIZE) \
  BOOST_PP_COMMA_IF(IDX) - i##BITS(BOOST_PP_CAT(v, BOOST_PP_SUB(SIZE, BOOST_PP_INC(IDX))))
#define GREX_SET_NEGVAL_IMPL(CNT, IDX, PAIR) GREX_SET_NEGVAL_IMPLI(CNT, IDX, PAIR)
#define GREX_SET_NEGVAL(CNT, IDX, PAIR) GREX_SET_NEGVAL_IMPL(CNT, IDX, BOOST_PP_REMOVE_PARENS(PAIR))

#define GREX_CMASK_SET_OP(CNT, IDX, TYPE) \
  BOOST_PP_IF(IDX, |, BOOST_PP_EMPTY()) \
  BOOST_PP_IF(IDX, (TYPE(v##IDX) << IDX##U), TYPE(v##IDX))
#define GREX_CMASK_SET(SIZE, TYPE) BOOST_PP_REPEAT(SIZE, GREX_CMASK_SET_OP, TYPE)

// Define mask operations, which can be applied to compressed or broad masks
#if GREX_X86_64_LEVEL >= 4
#define GREX_MASK_SET_IMPL(KIND, BITS, SIZE, BITPREFIX, SUFFIX, REGISTERBITS, MMASKSIZE) \
  inline Mask<KIND##BITS, SIZE> zero(thes::TypeTag<Mask<KIND##BITS, SIZE>>) { \
    return {.r = GREX_SIZEMMASK(SIZE){0}}; \
  } \
  inline Mask<KIND##BITS, SIZE> broadcast(bool value, thes::TypeTag<Mask<KIND##BITS, SIZE>>) { \
    return {.r = GREX_SIZEMMASK(SIZE)(-i##MMASKSIZE(value))}; \
  } \
  inline Mask<KIND##BITS, SIZE> set(BOOST_PP_REPEAT(SIZE, GREX_SET_ARG, bool), \
                                    thes::TypeTag<Mask<KIND##BITS, SIZE>>) { \
    return {.r = GREX_SIZEMMASK(SIZE)(GREX_CMASK_SET(SIZE, u##MMASKSIZE))}; \
  }
#else
#define GREX_MASK_SET_IMPL(KIND, BITS, SIZE, BITPREFIX, SUFFIX, REGISTERBITS, MMASKSIZE) \
  inline Mask<KIND##BITS, SIZE> zero(thes::TypeTag<Mask<KIND##BITS, SIZE>>) { \
    return {.r = BITPREFIX##_setzero_si##REGISTERBITS()}; \
  } \
  inline Mask<KIND##BITS, SIZE> broadcast(bool value, thes::TypeTag<Mask<KIND##BITS, SIZE>>) { \
    return {.r = BITPREFIX##_set1_##SUFFIX(-i##BITS(value))}; \
  } \
  inline Mask<KIND##BITS, SIZE> set(BOOST_PP_REPEAT(SIZE, GREX_SET_ARG, bool), \
                                    thes::TypeTag<Mask<KIND##BITS, SIZE>>) { \
    return {.r = BITPREFIX##_set_##SUFFIX(BOOST_PP_REPEAT(SIZE, GREX_SET_NEGVAL, (BITS, SIZE)))}; \
  }
#endif
#define GREX_MASK_SET(KIND, BITS, SIZE, BITPREFIX, SUFFIX, REGISTERBITS, MMASKSIZE) \
  GREX_MASK_SET_IMPL(KIND, BITS, SIZE, BITPREFIX, SUFFIX, REGISTERBITS, MMASKSIZE)

// Define the setzero-based operations
#define GREX_ZERO(ELEMENT, SIZE, BITPREFIX, SUFFIX, REGISTERBITS) \
  inline Vector<ELEMENT, SIZE> zero(thes::TypeTag<Vector<ELEMENT, SIZE>>) { \
    return {.r = BITPREFIX##_setzero_##SUFFIX()}; \
  }

// Define the set1- and set-based operations
#define GREX_SET_IMPL(ELEMENT, SIZE, BITPREFIX, SUFFIX, ARGS, VALS) \
  inline Vector<ELEMENT, SIZE> set(ARGS, thes::TypeTag<Vector<ELEMENT, SIZE>>) { \
    return {.r = BITPREFIX##_set_##SUFFIX(VALS)}; \
  }
#define GREX_VSET(ELEMENT, SIZE, BITPREFIX, SUFFIX, REGISTERBITS) \
  inline Vector<ELEMENT, SIZE> broadcast(ELEMENT value, thes::TypeTag<Vector<ELEMENT, SIZE>>) { \
    return {.r = BITPREFIX##_set1_##SUFFIX(value)}; \
  } \
  GREX_SET_IMPL(ELEMENT, SIZE, BITPREFIX, SUFFIX, BOOST_PP_REPEAT(SIZE, GREX_SET_ARG, ELEMENT), \
                BOOST_PP_REPEAT(SIZE, GREX_SET_VAL, SIZE))

// Define all set operations
#define GREX_ASET_IMPL(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  GREX_APPLY(GREX_ZERO, KIND##BITS, SIZE, BITPREFIX, GREX_SI_SUFFIX(KIND, BITS, REGISTERBITS), \
             REGISTERBITS) \
  GREX_APPLY(GREX_VSET, KIND##BITS, SIZE, BITPREFIX, GREX_SET_SUFFIX(KIND, BITS, REGISTERBITS), \
             REGISTERBITS) \
  GREX_APPLY(GREX_MASK_SET, KIND, BITS, SIZE, BITPREFIX, GREX_SET_EPI_##BITS(REGISTERBITS), \
             REGISTERBITS, GREX_MMASKSIZE(SIZE))
#define GREX_ASET(REGISTERBITS, PREFIX) \
  GREX_FOREACH_TYPE(GREX_ASET_IMPL, REGISTERBITS, PREFIX, REGISTERBITS)

GREX_FOREACH_X86_64_LEVEL(GREX_ASET)
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_SET_HPP
