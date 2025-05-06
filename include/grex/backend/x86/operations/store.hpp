// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_STORE_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_STORE_HPP

#include <immintrin.h>

#include "grex/backend/x86/instruction-sets.hpp"
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
  GREX_DEF_STORE(f32, (REGISTERBITS) / 32, PREFIX, ps, GREX_IDENTITY) \
  GREX_DEF_STORE(f64, (REGISTERBITS) / 64, PREFIX, pd, GREX_IDENTITY)
#define GREX_DEF_STORE_INT(REGISTERBITS, PREFIX) \
  GREX_DEFINE_SI_OPERATION(GREX_DEF_STORE, REGISTERBITS, PREFIX, si##REGISTERBITS, \
                           reinterpret_cast<__m##REGISTERBITS##i*>)

GREX_DEF_STORE_FP(128, mm)
GREX_DEF_STORE_INT(128, mm)
#if GREX_INSTRUCTION_SET >= GREX_AVX
GREX_DEF_STORE_FP(256, mm256)
#endif
#if GREX_INSTRUCTION_SET >= GREX_AVX2
GREX_DEF_STORE_INT(256, mm256)
#endif
#if GREX_INSTRUCTION_SET >= GREX_AVX512_1
GREX_DEF_STORE_FP(512, mm512)
#endif
#if GREX_INSTRUCTION_SET >= GREX_AVX512_2
GREX_DEF_STORE_INT(512, mm512)
#endif
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_STORE_HPP
