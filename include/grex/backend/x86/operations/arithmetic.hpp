// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_ARITHMETIC_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_ARITHMETIC_HPP

#include "grex/backend/x86/helpers.hpp"
#include "grex/backend/x86/types.hpp"

namespace grex::backend {
#define GREX_BINARITH_OP_IMPL(NAME, OP, KINDSUFFIX, ELEMENT, SIZE) \
  inline Vector<ELEMENT, SIZE> NAME(Vector<ELEMENT, SIZE> a, Vector<ELEMENT, SIZE> b) { \
    return {OP##_##KINDSUFFIX(a.r, b.r)}; \
  }
#define GREX_BINARITH_OP_BASE(KIND, BITS, SIZE, NAME, OP) \
  GREX_APPLY(GREX_BINARITH_OP_IMPL, NAME, OP, GREX_EPI_SUFFIX(KIND, BITS), KIND##BITS, SIZE)
#define GREX_BINARITH_OPS_ALL(REGISTERBITS, KINDPREFIX) \
  GREX_FOREACH_TYPE(GREX_BINARITH_OP_BASE, REGISTERBITS, add, _##KINDPREFIX##_add) \
  GREX_FOREACH_TYPE(GREX_BINARITH_OP_BASE, REGISTERBITS, subtract, _##KINDPREFIX##_sub)

GREX_FOREACH_X86_64_LEVEL(GREX_BINARITH_OPS_ALL)
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_ARITHMETIC_HPP
