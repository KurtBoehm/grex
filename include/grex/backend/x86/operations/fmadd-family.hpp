// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_FMADD_FAMILY_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_FMADD_FAMILY_HPP

#include <immintrin.h>

#include "grex/backend/x86/helpers.hpp"
#include "grex/backend/x86/instruction-sets.hpp"
#include "grex/backend/x86/operations/arithmetic.hpp" // IWYU pragma: keep
#include "grex/backend/x86/types.hpp"
#include "grex/base/defs.hpp" // IWYU pragma: keep

namespace grex::backend {
#if GREX_X86_64_LEVEL >= 3
#define GREX_FMADDF_CALL(NAME, BITPREFIX, KINDSUFFIX, ELEMENT, SIZE) \
  {.r = BITPREFIX##_##NAME##_##KINDSUFFIX(a.r, b.r, c.r)}
#else
#define GREX_FMADDF_CALL_fmadd add(multiply(a, b), c)
#define GREX_FMADDF_CALL_fmsub subtract(multiply(a, b), c)
#define GREX_FMADDF_CALL_fnmadd subtract(c, multiply(a, b))
#define GREX_FMADDF_CALL_fnmsub negate(add(multiply(a, b), c))
#define GREX_FMADDF_CALL(NAME, ...) GREX_FMADDF_CALL_##NAME
#endif

#define GREX_FMADDF_IMPL(NAME, ELEMENT, SIZE, BITPREFIX, KINDSUFFIX) \
  inline Vector<ELEMENT, SIZE> NAME(Vector<ELEMENT, SIZE> a, Vector<ELEMENT, SIZE> b, \
                                    Vector<ELEMENT, SIZE> c) { \
    return GREX_FMADDF_CALL(NAME, BITPREFIX, KINDSUFFIX, ELEMENT, SIZE); \
  }
#define GREX_FMADDF(KIND, BITS, SIZE, BITPREFIX, NAME) \
  GREX_APPLY(GREX_FMADDF_IMPL, NAME, KIND##BITS, SIZE, BITPREFIX, GREX_EPI_SUFFIX(KIND, BITS))
#define GREX_FMADDF_ALL(REGISTERBITS, BITPREFIX, NAME) \
  GREX_FOREACH_FP_TYPE(GREX_FMADDF, REGISTERBITS, BITPREFIX, NAME)

GREX_FOREACH_X86_64_LEVEL(GREX_FMADDF_ALL, fmadd)
GREX_FOREACH_X86_64_LEVEL(GREX_FMADDF_ALL, fmsub)
GREX_FOREACH_X86_64_LEVEL(GREX_FMADDF_ALL, fnmadd)
GREX_FOREACH_X86_64_LEVEL(GREX_FMADDF_ALL, fnmsub)
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_FMADD_FAMILY_HPP
