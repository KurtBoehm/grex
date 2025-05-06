// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_ARITHMETIC_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_ARITHMETIC_HPP

#include "grex/backend/x86/instruction-sets.hpp"
#include "grex/backend/x86/types.hpp"

namespace grex::backend {
#define DEF_BIN_OP_BASE(NAME, INTRINSIC, PREFIX, BITS, SIZE) \
  inline Vector<PREFIX##BITS, SIZE> NAME(Vector<PREFIX##BITS, SIZE> a, \
                                         Vector<PREFIX##BITS, SIZE> b) { \
    return {INTRINSIC(a.r, b.r)}; \
  }
#define DEF_BIN_OP_FP(NAME, OP, REGISTERBITS) \
  DEF_BIN_OP_BASE(NAME, OP##_ps, f, 32, (REGISTERBITS) / 32) \
  DEF_BIN_OP_BASE(NAME, OP##_pd, f, 64, (REGISTERBITS) / 64)
#define DEF_BIN_OP_INT(NAME, OP, REGISTERBITS) \
  DEF_BIN_OP_BASE(NAME, OP##_epi8, u, 8, (REGISTERBITS) / 8) \
  DEF_BIN_OP_BASE(NAME, OP##_epi8, i, 8, (REGISTERBITS) / 8) \
  DEF_BIN_OP_BASE(NAME, OP##_epi16, u, 16, (REGISTERBITS) / 16) \
  DEF_BIN_OP_BASE(NAME, OP##_epi16, i, 16, (REGISTERBITS) / 16) \
  DEF_BIN_OP_BASE(NAME, OP##_epi32, u, 32, (REGISTERBITS) / 32) \
  DEF_BIN_OP_BASE(NAME, OP##_epi32, i, 32, (REGISTERBITS) / 32) \
  DEF_BIN_OP_BASE(NAME, OP##_epi64, u, 64, (REGISTERBITS) / 64) \
  DEF_BIN_OP_BASE(NAME, OP##_epi64, i, 64, (REGISTERBITS) / 64)

DEF_BIN_OP_FP(add, _mm_add, 128)
DEF_BIN_OP_FP(subtract, _mm_sub, 128)
DEF_BIN_OP_INT(add, _mm_add, 128)
DEF_BIN_OP_INT(subtract, _mm_sub, 128)

#if GREX_INSTRUCTION_SET >= GREX_AVX
DEF_BIN_OP_FP(add, _mm256_add, 256)
DEF_BIN_OP_FP(subtract, _mm256_sub, 256)
#endif
#if GREX_INSTRUCTION_SET >= GREX_AVX2
DEF_BIN_OP_INT(add, _mm256_add, 256)
DEF_BIN_OP_INT(subtract, _mm256_sub, 256)
#endif

#if GREX_INSTRUCTION_SET >= GREX_AVX512_1
DEF_BIN_OP_FP(add, _mm512_add, 512)
DEF_BIN_OP_FP(subtract, _mm512_sub, 512)
#endif
#if GREX_INSTRUCTION_SET >= GREX_AVX512_2
DEF_BIN_OP_INT(add, _mm512_add, 512)
DEF_BIN_OP_INT(subtract, _mm512_sub, 512)
#endif
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_ARITHMETIC_HPP
