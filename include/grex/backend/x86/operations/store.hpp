// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_STORE_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_STORE_HPP

#include <immintrin.h>

#include "grex/backend/x86/operations/helpers.hpp"
#include "grex/backend/x86/types.hpp"
#include "grex/base/defs.hpp"

namespace grex::backend {
#define GREX_DEF_STORE_BASE(NAME, ELEMENT, SIZE, MACRO, CAST) \
  inline void NAME(ELEMENT* dst, Vector<ELEMENT, SIZE> src) { \
    MACRO(CAST(dst), src.r); \
  }
#define GREX_DEF_STORE(ELEMENT, SIZE, PREFIX, SUFFIX, CAST) \
  GREX_DEF_STORE_BASE(store, ELEMENT, SIZE, _##PREFIX##_storeu_##SUFFIX, CAST) \
  GREX_DEF_STORE_BASE(store_aligned, ELEMENT, SIZE, _##PREFIX##_store_##SUFFIX, CAST)
#define GREX_DEF_STORE_FP(REGISTERBITS, PREFIX) \
  GREX_DEF_STORE(f32, GREX_ELEMENTS(REGISTERBITS, 32), PREFIX, ps, GREX_IDENTITY) \
  GREX_DEF_STORE(f64, GREX_ELEMENTS(REGISTERBITS, 64), PREFIX, pd, GREX_IDENTITY)
#define GREX_DEF_STORE_INT(REGISTERBITS, PREFIX) \
  GREX_DEFINE_SI_OPERATION(GREX_DEF_STORE, REGISTERBITS, PREFIX, si##REGISTERBITS, \
                           reinterpret_cast<__m##REGISTERBITS##i*>)

GREX_DEFINE_INSTR_SET(GREX_DEF_STORE_FP, GREX_DEF_STORE_INT)
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_STORE_HPP
