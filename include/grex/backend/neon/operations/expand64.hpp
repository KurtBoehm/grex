// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_NEON_OPERATIONS_EXPAND64_HPP
#define INCLUDE_GREX_BACKEND_NEON_OPERATIONS_EXPAND64_HPP

#include <cstddef>

#include <arm_neon.h>

#include "grex/backend/defs.hpp" // IWYU pragma: keep
#include "grex/backend/macros/math.hpp"
#include "grex/backend/neon/macros/types.hpp"
#include "grex/base.hpp"

namespace grex::backend {
#if GREX_GCC
#define GREX_EXPAND_64(KIND, BITS, SIZE) \
  inline GREX_REGISTER(KIND, BITS, GREX_MULTIPLY(SIZE, 2)) \
    expand64(GREX_REGISTER(KIND, BITS, SIZE) v) { \
    GREX_REGISTER(KIND, BITS, GREX_MULTIPLY(SIZE, 2)) retval; \
    asm("" : "=w"(retval) : "0"(v)); /*NOLINT*/ \
    return retval; \
  }
#elif GREX_CLANG
#define GREX_EXPAND_64(KIND, BITS, SIZE) \
  inline GREX_REGISTER(KIND, BITS, GREX_MULTIPLY(SIZE, 2)) \
    expand64(GREX_REGISTER(KIND, BITS, SIZE) v) { \
    return static_apply<GREX_MULTIPLY(SIZE, 2)>([&]<std::size_t... tI>() { \
      return __builtin_shufflevector(v, v, ((tI < SIZE) ? int{tI} : -1)...); \
    }); \
  }
#endif

GREX_EXPAND_64(f, 64, 1) // NOLINT
GREX_EXPAND_64(i, 64, 1) // NOLINT
GREX_EXPAND_64(u, 64, 1) // NOLINT
GREX_EXPAND_64(f, 32, 2) // NOLINT
GREX_EXPAND_64(i, 32, 2) // NOLINT
GREX_EXPAND_64(u, 32, 2) // NOLINT
GREX_EXPAND_64(f, 16, 4) // NOLINT
GREX_EXPAND_64(i, 16, 4) // NOLINT
GREX_EXPAND_64(u, 16, 4) // NOLINT
GREX_EXPAND_64(i, 8, 8) // NOLINT
GREX_EXPAND_64(u, 8, 8) // NOLINT
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_EXPAND64_HPP
