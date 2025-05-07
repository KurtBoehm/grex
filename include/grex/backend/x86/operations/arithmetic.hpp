// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_ARITHMETIC_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_ARITHMETIC_HPP

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

// Define for floating-point types
#define GREX_DEF_BIN_OP_FP(NAME, OP, REGISTERBITS) \
  GREX_DEF_BIN_OP_BASE(NAME, OP##_ps, f32, GREX_ELEMENTS(REGISTERBITS, 32)) \
  GREX_DEF_BIN_OP_BASE(NAME, OP##_pd, f64, GREX_ELEMENTS(REGISTERBITS, 64))
#define GREX_DEF_OPS_FP(REGISTERBITS, PREFIX) \
  GREX_DEF_BIN_OP_FP(add, _##PREFIX##_add, REGISTERBITS) \
  GREX_DEF_BIN_OP_FP(subtract, _##PREFIX##_sub, REGISTERBITS)

// Define for integer types
#define GREX_DEF_BIN_OP_INT(NAME, OP, REGISTERBITS) \
  GREX_DEFINE_SI_OPERATION_EXT(GREX_DEF_BIN_OP_BASE_INT, REGISTERBITS, NAME, OP)
#define GREX_DEF_OPS_INT(REGISTERBITS, PREFIX) \
  GREX_DEF_BIN_OP_INT(add, _##PREFIX##_add, REGISTERBITS) \
  GREX_DEF_BIN_OP_INT(subtract, _##PREFIX##_sub, REGISTERBITS)

GREX_DEFINE_INSTR_SET(GREX_DEF_OPS_FP, GREX_DEF_OPS_INT)

#undef GREX_DEF_BIN_OP_BASE
#undef GREX_DEF_BIN_OP_BASE_INT
#undef GREX_DEF_BIN_OP_FP
#undef GREX_DEF_BIN_OP_INT
#undef GREX_DEF_OPS_FP
#undef GREX_DEF_OPS_INT
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_ARITHMETIC_HPP
