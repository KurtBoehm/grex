// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_FMADD_FAMILY_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_FMADD_FAMILY_HPP

#include <immintrin.h>

#include "grex/backend/defs.hpp"
#include "grex/backend/macros/for-each.hpp"
#include "grex/backend/x86/instruction-sets.hpp"
#include "grex/backend/x86/macros/for-each.hpp"
#include "grex/backend/x86/macros/types.hpp"
#include "grex/backend/x86/types.hpp"
#include "grex/base/defs.hpp" // IWYU pragma: keep

#if GREX_X86_64_LEVEL >= 3
#include "grex/backend/macros/base.hpp"
#include "grex/backend/x86/macros/intrinsics.hpp"
#include "grex/backend/x86/operations/expand-scalar.hpp"
#else
#include "grex/backend/x86/operations/arithmetic.hpp"
#endif

namespace grex::backend {
#if GREX_X86_64_LEVEL >= 3
#define GREX_FMADDF_CALL(NAME, KIND, BITS, BITPREFIX) \
  {.r = GREX_CAT(BITPREFIX##_##NAME##_, GREX_EPI_SUFFIX(KIND, BITS))(a.r, b.r, c.r)}
#define GREX_FMADDS_CALL(NAME, KIND, BITS, SIZE) \
  const auto va = expand_any(a, index_tag<SIZE>).r; \
  const auto vb = expand_any(b, index_tag<SIZE>).r; \
  const auto vc = expand_any(c, index_tag<SIZE>).r; \
  const auto vout = GREX_CAT(_mm_##NAME##_s, GREX_FP_LETTER(BITS))(va, vb, vc); \
  return {.value = GREX_CAT(_mm_cvts, GREX_FP_LETTER(BITS), _f##BITS)(vout)};

inline constexpr bool has_fma = true;
#else
#define GREX_FMADDF_CALL_fmadd add(multiply(a, b), c)
#define GREX_FMADDF_CALL_fmsub subtract(multiply(a, b), c)
#define GREX_FMADDF_CALL_fnmadd subtract(c, multiply(a, b))
#define GREX_FMADDF_CALL_fnmsub negate(add(multiply(a, b), c))
#define GREX_FMADDF_CALL(NAME, ...) GREX_FMADDF_CALL_##NAME
#define GREX_FMADDS_CALL_fmadd return {.value = (a.value * b.value) + c.value};
#define GREX_FMADDS_CALL_fmsub return {.value = (a.value * b.value) - c.value};
#define GREX_FMADDS_CALL_fnmadd return {.value = c.value - (a.value * b.value)};
#define GREX_FMADDS_CALL_fnmsub return {.value = -(a.value * b.value + c.value)};
#define GREX_FMADDS_CALL(NAME, ...) GREX_FMADDS_CALL_##NAME

inline constexpr bool has_fma = false;
#endif

#define GREX_FMADDF(KIND, BITS, SIZE, BITPREFIX, NAME) \
  inline Vector<KIND##BITS, SIZE> NAME(Vector<KIND##BITS, SIZE> a, Vector<KIND##BITS, SIZE> b, \
                                       Vector<KIND##BITS, SIZE> c) { \
    return GREX_FMADDF_CALL(NAME, KIND, BITS, BITPREFIX); \
  }
#define GREX_FMADDF_ALL(REGISTERBITS, BITPREFIX, NAME) \
  GREX_FOREACH_FP_TYPE(GREX_FMADDF, REGISTERBITS, BITPREFIX, NAME)

GREX_FOREACH_X86_64_LEVEL(GREX_FMADDF_ALL, fmadd)
GREX_FOREACH_X86_64_LEVEL(GREX_FMADDF_ALL, fmsub)
GREX_FOREACH_X86_64_LEVEL(GREX_FMADDF_ALL, fnmadd)
GREX_FOREACH_X86_64_LEVEL(GREX_FMADDF_ALL, fnmsub)

GREX_SUBVECTOR_TERNARY(fmadd)
GREX_SUBVECTOR_TERNARY(fmsub)
GREX_SUBVECTOR_TERNARY(fnmadd)
GREX_SUBVECTOR_TERNARY(fnmsub)

GREX_SUPERVECTOR_TERNARY(fmadd)
GREX_SUPERVECTOR_TERNARY(fmsub)
GREX_SUPERVECTOR_TERNARY(fnmadd)
GREX_SUPERVECTOR_TERNARY(fnmsub)

#define GREX_FMADDS(KIND, BITS, SIZE, NAME) \
  inline Scalar<KIND##BITS> NAME(Scalar<KIND##BITS> a, Scalar<KIND##BITS> b, \
                                 Scalar<KIND##BITS> c) { \
    GREX_FMADDS_CALL(NAME, KIND, BITS, SIZE) \
  }
GREX_FOREACH_FP_TYPE(GREX_FMADDS, 128, fmadd)
GREX_FOREACH_FP_TYPE(GREX_FMADDS, 128, fmsub)
GREX_FOREACH_FP_TYPE(GREX_FMADDS, 128, fnmadd)
GREX_FOREACH_FP_TYPE(GREX_FMADDS, 128, fnmsub)
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_FMADD_FAMILY_HPP
