// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_TYPES_HPP
#define INCLUDE_GREX_BACKEND_X86_TYPES_HPP

#include <cstddef>

#include <immintrin.h>

#include "grex/backend/base.hpp"
#include "grex/backend/macros/base.hpp"
#include "grex/backend/macros/for-each.hpp"
#include "grex/backend/x86/instruction-sets.hpp"
#include "grex/backend/x86/macros/for-each.hpp"
#include "grex/backend/x86/macros/intrinsics.hpp"

namespace grex::backend {
// Mask definition macros
#define GREX_TYPES_MASK_BROAD(KIND, SIZE, REGISTERBITS) \
  template<> \
  struct Mask<KIND, SIZE> { \
    using Register = __m##REGISTERBITS##i; \
    using VectorValue = KIND; \
    static constexpr std::size_t size = SIZE; \
    static constexpr std::size_t rbits = REGISTERBITS; \
\
    Register r; \
\
    Register registr() const { \
      return r; \
    } \
  }; \
  using b##KIND##x##SIZE = Mask<KIND, SIZE>
#define GREX_TYPES_MASK_COMPACT(KIND, BITS, SIZE, REGISTERBITS) \
  template<> \
  struct Mask<KIND##BITS, SIZE> { \
    using Register = GREX_MMASK(SIZE); \
    using VectorValue = KIND##BITS; \
    static constexpr std::size_t size = SIZE; \
    static constexpr std::size_t rbits = REGISTERBITS; \
\
    Register r; \
\
    Register registr() const { \
      return r; \
    } \
  }; \
  using b##KIND##BITS##x##SIZE = Mask<KIND##BITS, SIZE>
#if GREX_X86_64_LEVEL >= 4
#define GREX_TYPES_MASK GREX_TYPES_MASK_COMPACT
#else
#define GREX_TYPES_MASK(KIND, BITS, SIZE, REGISTERBITS) \
  GREX_TYPES_MASK_BROAD(KIND##BITS, SIZE, REGISTERBITS)
#endif

// Combined vector and mask definition macros
#define GREX_TYPES_IMPL(KIND, BITS, SIZE, REGISTERBITS) \
  template<> \
  struct Vector<KIND##BITS, SIZE> { \
    using Register = GREX_CAT(__m##REGISTERBITS, GREX_REGISTER_SUFFIX(KIND, BITS)); \
    using Value = KIND##BITS; \
    static constexpr std::size_t size = SIZE; \
\
    Register r; \
\
    Vector native() const { \
      return *this; \
    } \
    Register registr() const { \
      return r; \
    } \
  }; \
  using KIND##BITS##x##SIZE = Vector<KIND##BITS, SIZE>; \
  GREX_TYPES_MASK(KIND, BITS, SIZE, REGISTERBITS);
#define GREX_TYPES(KIND, BITS, SIZE, REGISTERBITS) GREX_TYPES_IMPL(KIND, BITS, SIZE, REGISTERBITS)

#define GREX_TYPES_ALL(REGISTERBITS, BITPREFIX) \
  GREX_FOREACH_TYPE(GREX_TYPES, REGISTERBITS, REGISTERBITS)

GREX_FOREACH_X86_64_LEVEL(GREX_TYPES_ALL)
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_TYPES_HPP
