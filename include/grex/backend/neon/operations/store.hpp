// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_NEON_OPERATIONS_STORE_HPP
#define INCLUDE_GREX_BACKEND_NEON_OPERATIONS_STORE_HPP

#include <arm_neon.h>

#include "grex/backend/macros/base.hpp"
#include "grex/backend/neon/macros/types.hpp"
#include "grex/backend/neon/types.hpp"

namespace grex::backend {
#define GREX_STORE(KIND, BITS, SIZE) \
  inline void store(KIND##BITS* dst, Vector<KIND##BITS, SIZE> src) { \
    GREX_CAT(vst1q_, GREX_ISUFFIX(KIND, BITS))(dst, src.r); \
  } \
  /* This is not actually aligned, but who cares */ \
  inline void store_aligned(KIND##BITS* dst, Vector<KIND##BITS, SIZE> src) { \
    GREX_CAT(vst1q_, GREX_ISUFFIX(KIND, BITS))(dst, src.r); \
  }

GREX_FOREACH_TYPE(GREX_STORE, 128)
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_STORE_HPP
