// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_SET_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_SET_HPP

#include <immintrin.h>

#include "thesauros/types.hpp"

#include "grex/backend/x86/instruction-sets.hpp"
#include "grex/backend/x86/operations/helpers.hpp"
#include "grex/backend/x86/types.hpp"
#include "grex/base/defs.hpp"

namespace grex::backend {
#define GREX_DEF_ZERO(ELEMENT, SIZE, PREFIX, SUFFIX) \
  inline Vector<ELEMENT, SIZE> zero(thes::TypeTag<ELEMENT>, thes::IndexTag<SIZE>) { \
    return {_##PREFIX##_setzero_##SUFFIX()}; \
  }
#define GREX_DEF_BROADCAST(ELEMENT, SIZE, MACRO) \
  inline Vector<ELEMENT, SIZE> broadcast(ELEMENT value, thes::IndexTag<SIZE>) { \
    return {MACRO(value)}; \
  }
#define GREX_DEF_SSET_FP_IMPL(ELEMENT, SIZE, PREFIX, SUFFIX) \
  GREX_DEF_ZERO(ELEMENT, SIZE, PREFIX, SUFFIX) \
  GREX_DEF_BROADCAST(ELEMENT, SIZE, _##PREFIX##_set1_##SUFFIX)
#define GREX_DEF_SSET_FP(REGISTERBITS, PREFIX) \
  GREX_DEF_SSET_FP_IMPL(f32, REGISTERBITS / 32, PREFIX, ps) \
  GREX_DEF_SSET_FP_IMPL(f64, REGISTERBITS / 64, PREFIX, pd)
#define GREX_DEF_SSET_INT(REGISTERBITS, PREFIX, SUFFIX64) \
  GREX_DEFINE_SI_OPERATION(GREX_DEF_ZERO, REGISTERBITS, PREFIX, si##REGISTERBITS) \
  GREX_DEF_BROADCAST(u8, (REGISTERBITS) / 8, _##PREFIX##_set1_epi8) \
  GREX_DEF_BROADCAST(i8, (REGISTERBITS) / 8, _##PREFIX##_set1_epi8) \
  GREX_DEF_BROADCAST(u16, (REGISTERBITS) / 16, _##PREFIX##_set1_epi16) \
  GREX_DEF_BROADCAST(i16, (REGISTERBITS) / 16, _##PREFIX##_set1_epi16) \
  GREX_DEF_BROADCAST(u32, (REGISTERBITS) / 32, _##PREFIX##_set1_epi32) \
  GREX_DEF_BROADCAST(i32, (REGISTERBITS) / 32, _##PREFIX##_set1_epi32) \
  GREX_DEF_BROADCAST(u64, (REGISTERBITS) / 64, _##PREFIX##_set1_epi64##SUFFIX64) \
  GREX_DEF_BROADCAST(i64, (REGISTERBITS) / 64, _##PREFIX##_set1_epi64##SUFFIX64)

GREX_DEF_SSET_FP(128, mm)
GREX_DEF_SSET_INT(128, mm, x)
#if GREX_INSTRUCTION_SET >= GREX_AVX
GREX_DEF_SSET_FP(256, mm256)
#endif
#if GREX_INSTRUCTION_SET >= GREX_AVX2
GREX_DEF_SSET_INT(256, mm256, x)
#endif
#if GREX_INSTRUCTION_SET >= GREX_AVX512_1
GREX_DEF_SSET_FP(512, mm512)
#endif
#if GREX_INSTRUCTION_SET >= GREX_AVX512_2
GREX_DEF_SSET_INT(512, mm512, )
#endif

#undef GREX_DEF_ZERO
#undef GREX_DEF_BROADCAST
#undef GREX_DEF_SSET_FP
#undef GREX_DEF_SSET_INT
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_SET_HPP
