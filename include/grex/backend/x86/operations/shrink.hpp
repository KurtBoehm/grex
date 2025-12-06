// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_SHRINK_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_SHRINK_HPP

#include <cstddef>

#include "grex/backend/defs.hpp"
#include "grex/backend/macros/base.hpp"
#include "grex/backend/macros/for-each.hpp"
#include "grex/backend/macros/math.hpp"
#include "grex/backend/x86/instruction-sets.hpp"
#include "grex/backend/x86/macros/intrinsics.hpp"
#include "grex/backend/x86/types.hpp"
#include "grex/base.hpp"

// Shared definitions
#include "grex/backend/shared/operations/shrink.hpp" // IWYU pragma: export

namespace grex::backend {
#define GREX_SHRINK_INTRINSIC(KIND, BITS, DSTSIZE, SRCSIZE) \
  inline Vector<KIND##BITS, DSTSIZE> shrink(Vector<KIND##BITS, SRCSIZE> v, \
                                            IndexTag<DSTSIZE> /*dst_size*/) { \
    return {.r = GREX_CAT(GREX_BITPREFIX(GREX_MULTIPLY(BITS, SRCSIZE)), _cast, \
                          GREX_SIR_SUFFIX(KIND, BITS, GREX_MULTIPLY(BITS, SRCSIZE)), _, \
                          GREX_SIR_SUFFIX(KIND, BITS, GREX_MULTIPLY(BITS, DSTSIZE)))(v.r)}; \
  }
#if GREX_X86_64_LEVEL >= 3
#define GREX_SHRINK256(KIND, BITS, SIZE) \
  GREX_SHRINK_INTRINSIC(KIND, BITS, SIZE, GREX_MULTIPLY(SIZE, 2))
GREX_FOREACH_TYPE(GREX_SHRINK256, 128)
#endif
#if GREX_X86_64_LEVEL >= 4
#define GREX_SHRINK512(KIND, BITS, SIZE) \
  GREX_SHRINK_INTRINSIC(KIND, BITS, SIZE, GREX_MULTIPLY(SIZE, 4)) \
  GREX_SHRINK_INTRINSIC(KIND, BITS, GREX_MULTIPLY(SIZE, 2), GREX_MULTIPLY(SIZE, 4))
GREX_FOREACH_TYPE(GREX_SHRINK512, 128)
#endif
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_SHRINK_HPP
