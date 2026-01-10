// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_NEON_TYPES_HPP
#define INCLUDE_GREX_BACKEND_NEON_TYPES_HPP

#include <arm_neon.h>

#include "grex/backend/base.hpp"
#include "grex/backend/defs.hpp" // IWYU pragma: keep
#include "grex/backend/macros/for-each.hpp"
#include "grex/backend/neon/macros/types.hpp"
#include "grex/base.hpp" // IWYU pragma: keep

namespace grex::backend {
#define GREX_TYPES_I(KIND, BITS, SIZE) \
  template<> \
  struct Mask<KIND##BITS, SIZE> { \
    using Register = GREX_REGISTER(u, BITS, SIZE); \
    using VectorValue = KIND##BITS; \
    static constexpr std::size_t size = SIZE; \
    static constexpr std::size_t bytes = sizeof(VectorValue) * size; \
\
    Register r; \
\
    [[nodiscard]] Register registr() const { \
      return r; \
    } \
  }; \
  using b##KIND##BITS##x##SIZE = Mask<KIND##BITS, SIZE>; \
  template<> \
  struct Vector<KIND##BITS, SIZE> { \
    using Register = GREX_REGISTER(KIND, BITS, SIZE); \
    using Value = KIND##BITS; \
    static constexpr std::size_t size = SIZE; \
    static constexpr std::size_t bytes = sizeof(Value) * size; \
\
    Register r; \
\
    [[nodiscard]] Register registr() const { \
      return r; \
    } \
  }; \
  using KIND##BITS##x##SIZE = Vector<KIND##BITS, SIZE>;
#define GREX_TYPES(KIND, BITS, SIZE) GREX_TYPES_I(KIND, BITS, SIZE)

GREX_FOREACH_TYPE(GREX_TYPES, 128)
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_NEON_TYPES_HPP
