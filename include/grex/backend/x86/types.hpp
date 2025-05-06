// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_TYPES_HPP
#define INCLUDE_GREX_BACKEND_X86_TYPES_HPP

#include <immintrin.h>

#include "grex/backend/defs.hpp"
#include "grex/backend/x86/instruction-sets.hpp"
#include "grex/base/defs.hpp"

namespace grex::backend {
// Mask definition macros

#define DEF_MASK_BROAD(ELEMENT, SIZE, REGISTER) \
  template<> \
  struct Mask<ELEMENT, SIZE> { \
    REGISTER r; \
  }; \
  using b##ELEMENT##x##SIZE = Mask<ELEMENT, SIZE>
#define DEF_MASK_COMPACT(PREFIX, BITS, SIZE, REGISTER) \
  template<> \
  struct Mask<PREFIX##BITS, SIZE> { \
    __mmask##BITS r; \
  }; \
  using b##PREFIX##BITS##x##SIZE = Mask<PREFIX##BITS, SIZE>

#if GREX_INSTRUCTION_SET >= GREX_AVX512_1
#define DEF_MASK_FP(PREFIX, BITS, SIZE, REGISTER) DEF_MASK_COMPACT(PREFIX, BITS, SIZE, REGISTER)
#else
#define DEF_MASK_FP(PREFIX, BITS, SIZE, REGISTER) DEF_MASK_BROAD(PREFIX##BITS, SIZE, REGISTER)
#endif

#if GREX_INSTRUCTION_SET >= GREX_AVX512_2
#define DEF_MASK_INT(PREFIX, BITS, SIZE, REGISTER) DEF_MASK_COMPACT(PREFIX, BITS, SIZE, REGISTER)
#else
#define DEF_MASK_INT(PREFIX, BITS, SIZE, REGISTER) DEF_MASK_BROAD(PREFIX##BITS, SIZE, REGISTER)
#endif

// Combined vector and mask definition macros

#define DEF_FP(PREFIX, BITS, SIZE, REGISTER) \
  template<> \
  struct Vector<PREFIX##BITS, SIZE> { \
    REGISTER r; \
  }; \
  using PREFIX##BITS##x##SIZE = Vector<PREFIX##BITS, SIZE>; \
  DEF_MASK_FP(PREFIX, BITS, SIZE, REGISTER)

#define DEF_INT_BASE(BITS) \
  struct Vector##BITS##i { \
    __m##BITS##i r; \
  }

#define DEF_INT(PREFIX, BITS, SIZE, REGISTERBITS) \
  template<> \
  struct Vector<PREFIX##BITS, SIZE> : public Vector##REGISTERBITS##i {}; \
  using PREFIX##BITS##x##SIZE = Vector<PREFIX##BITS, SIZE>; \
  DEF_MASK_INT(PREFIX, BITS, SIZE, __m##REGISTERBITS##i)

// Define 128 bit types (always available)

DEF_FP(f, 32, 4, __m128);
DEF_FP(f, 64, 2, __m128d);
DEF_INT_BASE(128);
DEF_INT(u, 8, 16, 128);
DEF_INT(i, 8, 16, 128);
DEF_INT(u, 16, 8, 128);
DEF_INT(i, 16, 8, 128);
DEF_INT(u, 32, 4, 128);
DEF_INT(i, 32, 4, 128);
DEF_INT(u, 64, 2, 128);
DEF_INT(i, 64, 2, 128);

// Define 256 bit types (available with AVX/AVX2)

#if GREX_INSTRUCTION_SET >= GREX_AVX
DEF_FP(f, 32, 8, __m256);
DEF_FP(f, 64, 4, __m256d);
#endif
#if GREX_INSTRUCTION_SET >= GREX_AVX2
DEF_INT_BASE(256);
DEF_INT(u, 8, 32, 256);
DEF_INT(i, 8, 32, 256);
DEF_INT(u, 16, 16, 256);
DEF_INT(i, 16, 16, 256);
DEF_INT(u, 32, 8, 256);
DEF_INT(i, 32, 8, 256);
DEF_INT(u, 64, 4, 256);
DEF_INT(i, 64, 4, 256);
#endif

// Define 512 bit types (available with AVX-512)

#if GREX_INSTRUCTION_SET >= GREX_AVX512_1
DEF_FP(f, 32, 16, __m512);
DEF_FP(f, 64, 8, __m512d);
#endif
#if GREX_INSTRUCTION_SET >= GREX_AVX512_2
DEF_INT_BASE(512);
DEF_INT(u, 8, 64, 512);
DEF_INT(i, 8, 64, 512);
DEF_INT(u, 16, 32, 512);
DEF_INT(i, 16, 32, 512);
DEF_INT(u, 32, 16, 512);
DEF_INT(i, 32, 16, 512);
DEF_INT(u, 64, 8, 512);
DEF_INT(i, 64, 8, 512);
#endif

#undef DEF_MASK_BROAD
#undef DEF_MASK_COMPACT
#undef DEF_MASK_FP
#undef DEF_MASK_INT
#undef DEF_FP
#undef DEF_INT_BASE
#undef DEF_INT
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_TYPES_HPP
