// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_ARITHMETIC_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_ARITHMETIC_HPP

#include "grex/backend/x86/instruction-sets.hpp"
#include "grex/backend/x86/operations/helpers.hpp"
#include "grex/backend/x86/types.hpp"
#include "grex/base/defs.hpp"

namespace grex::backend {
#define GREX_DEF_BIN_OP_BASE(NAME, INTRINSIC, ELEMENT, SIZE) \
  inline Vector<ELEMENT, SIZE> NAME(Vector<ELEMENT, SIZE> a, Vector<ELEMENT, SIZE> b) { \
    return {INTRINSIC(a.r, b.r)}; \
  }
#define GREX_DEF_BIN_OP_BASE_INT(PREFIX, BITS, SIZE, NAME, OP) \
  GREX_DEF_BIN_OP_BASE(NAME, OP##_epi##BITS, PREFIX##BITS, SIZE)

#define GREX_DEF_BIN_OP_FP(NAME, OP, REGISTERBITS) \
  GREX_DEF_BIN_OP_BASE(NAME, OP##_ps, f32, GREX_ELEMENTS(REGISTERBITS, 32)) \
  GREX_DEF_BIN_OP_BASE(NAME, OP##_pd, f64, GREX_ELEMENTS(REGISTERBITS, 64))
#define GREX_DEF_BIN_OP_INT(NAME, OP, REGISTERBITS) \
  GREX_DEFINE_SI_OPERATION_EXT(GREX_DEF_BIN_OP_BASE_INT, REGISTERBITS, NAME, OP)

GREX_DEF_BIN_OP_FP(add, _mm_add, 128)
GREX_DEF_BIN_OP_FP(subtract, _mm_sub, 128)
GREX_DEF_BIN_OP_INT(add, _mm_add, 128)
GREX_DEF_BIN_OP_INT(subtract, _mm_sub, 128)

#if GREX_INSTRUCTION_SET >= GREX_AVX
GREX_DEF_BIN_OP_FP(add, _mm256_add, 256)
GREX_DEF_BIN_OP_FP(subtract, _mm256_sub, 256)
#endif
#if GREX_INSTRUCTION_SET >= GREX_AVX2
GREX_DEF_BIN_OP_INT(add, _mm256_add, 256)
GREX_DEF_BIN_OP_INT(subtract, _mm256_sub, 256)
#endif

#if GREX_INSTRUCTION_SET >= GREX_AVX512_1
GREX_DEF_BIN_OP_FP(add, _mm512_add, 512)
GREX_DEF_BIN_OP_FP(subtract, _mm512_sub, 512)
#endif
#if GREX_INSTRUCTION_SET >= GREX_AVX512_2
GREX_DEF_BIN_OP_INT(add, _mm512_add, 512)
GREX_DEF_BIN_OP_INT(subtract, _mm512_sub, 512)
#endif

#undef GREX_DEF_BIN_OP_BASE
#undef GREX_DEF_BIN_OP_BASE_INT
#undef GREX_DEF_BIN_OP_FP
#undef GREX_DEF_BIN_OP_INT
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_ARITHMETIC_HPP
