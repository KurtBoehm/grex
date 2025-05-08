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
#include "grex/backend/x86/types.hpp"

namespace grex::backend {
// Define the setzero-based operation
#define GREX_ZERO(ELEMENT, SIZE, PREFIX, SUFFIX) \
  inline Vector<ELEMENT, SIZE> zero(thes::TypeTag<ELEMENT>, thes::IndexTag<SIZE>) { \
    return {_##PREFIX##_setzero_##SUFFIX()}; \
  }

// Helpers to define the set-based operation
#define GREX_SET_ARG(CNT, IDX, TYPE) BOOST_PP_COMMA_IF(IDX) TYPE v##IDX
#define GREX_SET_VAL_NAME_IMPL(IDX) v##IDX
#define GREX_SET_VAL_NAME(IDX) GREX_SET_VAL_NAME_IMPL(IDX)
#define GREX_SET_VAL(CNT, IDX, SIZE) \
  BOOST_PP_COMMA_IF(IDX) GREX_SET_VAL_NAME(BOOST_PP_SUB(SIZE, BOOST_PP_INC(IDX)))
#define GREX_SET_IMPL(ELEMENT, SIZE, PREFIX, SUFFIX, ARGS, VALS) \
  inline Vector<ELEMENT, SIZE> set(ARGS) { \
    return {_##PREFIX##_set_##SUFFIX(VALS)}; \
  }
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

// Define the set1- and set-based operations
#define GREX_VSET(ELEMENT, SIZE, PREFIX, SUFFIX) \
  inline Vector<ELEMENT, SIZE> broadcast(ELEMENT value, thes::IndexTag<SIZE>) { \
    return {_##PREFIX##_set1_##SUFFIX(value)}; \
  } \
  GREX_SET_IMPL(ELEMENT, SIZE, PREFIX, SUFFIX, BOOST_PP_REPEAT(SIZE, GREX_SET_ARG, ELEMENT), \
                BOOST_PP_REPEAT(SIZE, GREX_SET_VAL, SIZE))

// Define all set operations
#define GREX_ASET_IMPL(KIND, BITS, SIZE, PREFIX, REGISTERBITS) \
  GREX_APPLY(GREX_ZERO, KIND##BITS, SIZE, PREFIX, GREX_SI_SUFFIX(KIND, BITS, REGISTERBITS)) \
  GREX_APPLY(GREX_VSET, KIND##BITS, SIZE, PREFIX, GREX_SET_SUFFIX(KIND, BITS, REGISTERBITS))
#define GREX_ASET(REGISTERBITS, PREFIX) \
  GREX_FOREACH_TYPE(GREX_ASET_IMPL, REGISTERBITS, PREFIX, REGISTERBITS)

GREX_FOREACH_X86_64_LEVEL(GREX_ASET)
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_SET_HPP
