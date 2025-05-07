// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_SET_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_SET_HPP

#include <immintrin.h>

#include "thesauros/types.hpp"

#include "grex/backend/x86/operations/helpers.hpp"
#include "grex/backend/x86/types.hpp"
#include "grex/base/defs.hpp"

namespace grex::backend {
// Define the setzero-based operation
#define GREX_DEF_ZERO(ELEMENT, SIZE, PREFIX, SUFFIX) \
  inline Vector<ELEMENT, SIZE> zero(thes::TypeTag<ELEMENT>, thes::IndexTag<SIZE>) { \
    return {_##PREFIX##_setzero_##SUFFIX()}; \
  }

// Helpers to define the set-based operation
#define GREX_DEF_SET_ARG(CNT, IDX, TYPE) BOOST_PP_COMMA_IF(IDX) TYPE v##IDX
#define GREX_DEF_SET_VAL_NAME_IMPL(IDX) v##IDX
#define GREX_DEF_SET_VAL_NAME(IDX) GREX_DEF_SET_VAL_NAME_IMPL(IDX)
#define GREX_DEF_SET_VAL(CNT, IDX, SIZE) \
  BOOST_PP_COMMA_IF(IDX) GREX_DEF_SET_VAL_NAME(BOOST_PP_SUB(SIZE, BOOST_PP_INC(IDX)))
#define GREX_DEF_SET_IMPL(ELEMENT, SIZE, PREFIX, SUFFIX, ARGS, VALS) \
  inline Vector<ELEMENT, SIZE> set(ARGS) { \
    return {_##PREFIX##_set_##SUFFIX(VALS)}; \
  }
#define GREX_SET_EPI64_128 epi64x
#define GREX_SET_EPI64_256 epi64x
#define GREX_SET_EPI64_512 epi64

// Define the set1- and set-based operations
#define GREX_DEF_VSET(ELEMENT, SIZE, PREFIX, SUFFIX) \
  inline Vector<ELEMENT, SIZE> broadcast(ELEMENT value, thes::IndexTag<SIZE>) { \
    return {_##PREFIX##_set1_##SUFFIX(value)}; \
  } \
  GREX_DEF_SET_IMPL(ELEMENT, SIZE, PREFIX, SUFFIX, \
                    BOOST_PP_REPEAT(SIZE, GREX_DEF_SET_ARG, ELEMENT), \
                    BOOST_PP_REPEAT(SIZE, GREX_DEF_SET_VAL, SIZE))

// Define all set operations for floating-point types
#define GREX_DEF_ASET_FP_IMPL(ELEMENT, SIZE, PREFIX, SUFFIX) \
  GREX_DEF_ZERO(ELEMENT, SIZE, PREFIX, SUFFIX) \
  GREX_DEF_VSET(ELEMENT, SIZE, PREFIX, SUFFIX)
#define GREX_DEF_ASET_FP(REGISTERBITS, PREFIX) \
  GREX_DEF_ASET_FP_IMPL(f32, GREX_ELEMENTS(REGISTERBITS, 32), PREFIX, ps) \
  GREX_DEF_ASET_FP_IMPL(f64, GREX_ELEMENTS(REGISTERBITS, 64), PREFIX, pd)
// Define all set operations for integer types
#define GREX_DEF_ASET_INT(REGISTERBITS, PREFIX) \
  GREX_DEFINE_SI_OPERATION(GREX_DEF_ZERO, REGISTERBITS, PREFIX, si##REGISTERBITS) \
  GREX_DEF_VSET(u8, GREX_ELEMENTS(REGISTERBITS, 8), PREFIX, epi8) \
  GREX_DEF_VSET(i8, GREX_ELEMENTS(REGISTERBITS, 8), PREFIX, epi8) \
  GREX_DEF_VSET(u16, GREX_ELEMENTS(REGISTERBITS, 16), PREFIX, epi16) \
  GREX_DEF_VSET(i16, GREX_ELEMENTS(REGISTERBITS, 16), PREFIX, epi16) \
  GREX_DEF_VSET(u32, GREX_ELEMENTS(REGISTERBITS, 32), PREFIX, epi32) \
  GREX_DEF_VSET(i32, GREX_ELEMENTS(REGISTERBITS, 32), PREFIX, epi32) \
  GREX_APPLY(GREX_DEF_VSET, u64, GREX_ELEMENTS(REGISTERBITS, 64), PREFIX, \
             GREX_SET_EPI64_##REGISTERBITS) \
  GREX_APPLY(GREX_DEF_VSET, i64, GREX_ELEMENTS(REGISTERBITS, 64), PREFIX, \
             GREX_SET_EPI64_##REGISTERBITS)

GREX_DEFINE_INSTR_SET(GREX_DEF_ASET_FP, GREX_DEF_ASET_INT)

#undef GREX_DEF_ZERO
#undef GREX_DEF_SET_ARG
#undef GREX_DEF_SET_VAL_NAME_IMPL
#undef GREX_DEF_SET_VAL_NAME
#undef GREX_DEF_SET_VAL
#undef GREX_DEF_SET_IMPL
#undef GREX_SET_EPI64_128
#undef GREX_SET_EPI64_256
#undef GREX_SET_EPI64_512
#undef GREX_DEF_VSET
#undef GREX_DEF_ASET_FP_IMPL
#undef GREX_DEF_ASET_FP
#undef GREX_DEF_ASET_INT
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_SET_HPP
