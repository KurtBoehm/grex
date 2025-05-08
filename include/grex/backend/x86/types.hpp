// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_TYPES_HPP
#define INCLUDE_GREX_BACKEND_X86_TYPES_HPP

#include <immintrin.h>

#include "grex/backend/defs.hpp"
#include "grex/backend/x86/helpers.hpp"
#include "grex/backend/x86/instruction-sets.hpp"

namespace grex::backend {
// Mask definition macros
#define GREX_TYPES_MASK_BROAD(ELEMENT, SIZE, REGISTER) \
  template<> \
  struct Mask<ELEMENT, SIZE> { \
    REGISTER r; \
  }; \
  using b##ELEMENT##x##SIZE = Mask<ELEMENT, SIZE>
#define GREX_TYPES_MASK_COMPACT(PREFIX, BITS, SIZE, REGISTER) \
  template<> \
  struct Mask<PREFIX##BITS, SIZE> { \
    __mmask##BITS r; \
  }; \
  using b##PREFIX##BITS##x##SIZE = Mask<PREFIX##BITS, SIZE>
#if GREX_X86_64_LEVEL >= 4
#define GREX_TYPES_MASK GREX_TYPES_MASK_COMPACT
#else
#define GREX_TYPES_MASK(PREFIX, BITS, SIZE, REGISTER) \
  GREX_TYPES_MASK_BROAD(PREFIX##BITS, SIZE, REGISTER)
#endif

// Combined vector and mask definition macros
#define GREX_TYPES_IMPL(PREFIX, BITS, SIZE, REGISTER) \
  template<> \
  struct Vector<PREFIX##BITS, SIZE> { \
    REGISTER r; \
  }; \
  using PREFIX##BITS##x##SIZE = Vector<PREFIX##BITS, SIZE>; \
  GREX_TYPES_MASK(PREFIX, BITS, SIZE, REGISTER);
#define GREX_TYPES(PREFIX, BITS, SIZE, REGISTERBITS) \
  GREX_TYPES_IMPL(PREFIX, BITS, SIZE, GREX_BITREGISTER(REGISTERBITS))

#define GREX_TYPES_ALL(REGISTERBITS, BITPREFIX) \
  GREX_FOREACH_TYPE(GREX_TYPES, REGISTERBITS, REGISTERBITS)

GREX_FOREACH_X86_64_LEVEL(GREX_TYPES_ALL)
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_TYPES_HPP
