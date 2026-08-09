// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_ARITHMETIC_MASK_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_ARITHMETIC_MASK_HPP

#include <immintrin.h>

#include "grex/backend/base.hpp"
#include "grex/backend/defs.hpp"
#include "grex/backend/macros/base.hpp"
#include "grex/backend/macros/for-each.hpp"
#include "grex/backend/macros/math.hpp"
#include "grex/backend/x86/instruction-sets.hpp"
#include "grex/backend/x86/macros/for-each.hpp"
#include "grex/backend/x86/macros/intrinsics.hpp"
#include "grex/backend/x86/operations/arithmetic.hpp"
#include "grex/backend/x86/operations/blend.hpp"
#include "grex/backend/x86/operations/f16.hpp"
#include "grex/backend/x86/types.hpp"
#include "grex/base.hpp"

namespace grex::backend {
// Wrapper for all variants.
#define GREX_MASKARITH_WRAP(KIND, BITS, SIZE, NAME, OP, IMPL) \
  inline NativeVector<KIND##BITS, SIZE> mask_##NAME(NativeMask<KIND##BITS, SIZE> m, \
                                                    NativeVector<KIND##BITS, SIZE> a, \
                                                    NativeVector<KIND##BITS, SIZE> b) { \
    IMPL(KIND, BITS, SIZE, NAME, OP) \
  }

// AVX-512 (AVX512-FP for binary16): Use intrinsics.
#define GREX_MASKARITH_AVX512(KIND, BITS, SIZE, NAME, OP) \
  const auto ar = from_stored<KIND##BITS>(a.r); \
  const auto br = from_stored<KIND##BITS>(b.r); \
  const auto r = GREX_CAT(OP##_, GREX_EPI_SUFFIX(KIND, BITS))(ar, m.r, ar, br); \
  return {.r = to_stored<KIND##BITS>(r)};

// Addition/subtraction fallback: zero-blend and apply the basic operator.
#define GREX_MASKADDSUB_FALLBACK(KIND, BITS, SIZE, NAME, OP) return NAME(a, blend_zero(m, b));
// Multiplication/division fallback: blend the basic operator with `a`.
#define GREX_MASKMULDIV_FALLBACK(KIND, BITS, SIZE, NAME, OP) return blend(m, a, NAME(a, b));

#if GREX_X86_64_LEVEL >= 4
#define GREX_MASKADDSUB_IMPL GREX_MASKARITH_AVX512
#define GREX_MASKMULDIV_IMPL GREX_MASKARITH_AVX512
#else
#define GREX_MASKADDSUB_IMPL GREX_MASKADDSUB_FALLBACK
#define GREX_MASKMULDIV_IMPL GREX_MASKMULDIV_FALLBACK
#endif

// Multiplication: no instruction for 8 bit integers, always use fallback.
#define GREX_MASKMUL_IMPL_INT8 GREX_MASKMULDIV_FALLBACK
#define GREX_MASKMUL_IMPL_INT16 GREX_MASKMULDIV_IMPL
#define GREX_MASKMUL_IMPL_INT32 GREX_MASKMULDIV_IMPL
#define GREX_MASKMUL_IMPL_INT64 GREX_MASKMULDIV_IMPL
#define GREX_MASKMUL_IMPL_f(BITS) GREX_MASKMULDIV_IMPL
#define GREX_MASKMUL_IMPL_i(BITS) GREX_MASKMUL_IMPL_INT##BITS
#define GREX_MASKMUL_IMPL_u(BITS) GREX_MASKMUL_IMPL_INT##BITS
#define GREX_MASKMUL_IMPL(KIND, BITS) GREX_MASKMUL_IMPL_##KIND(BITS)

// Multiplication: truncated integer multiplication just like for primitives.
#define GREX_MASKMUL_INTRIN_f(BITPREFIX) BITPREFIX##_mask_mul
#define GREX_MASKMUL_INTRIN_i(BITPREFIX) BITPREFIX##_mask_mullo
#define GREX_MASKMUL_INTRIN_u(BITPREFIX) BITPREFIX##_mask_mullo
#define GREX_MASKMUL_INTRIN(KIND, BITPREFIX) GREX_MASKMUL_INTRIN_##KIND(BITPREFIX)

// Addition/subtraction.
#define GREX_MASKADDSUB_MAIN(...) GREX_MASKARITH_WRAP(__VA_ARGS__, GREX_MASKADDSUB_IMPL)
#define GREX_MASKADDSUB_ALL(REGISTERBITS, BITPREFIX) \
  GREX_FOREACH_TYPE(GREX_MASKADDSUB_MAIN, REGISTERBITS, add, BITPREFIX##_mask_add) \
  GREX_FOREACH_TYPE(GREX_MASKADDSUB_MAIN, REGISTERBITS, subtract, BITPREFIX##_mask_sub)

// Multiplication.
#define GREX_MASKMUL(KIND, BITS, SIZE, BITPREFIX) \
  GREX_MASKARITH_WRAP(KIND, BITS, SIZE, multiply, GREX_MASKMUL_INTRIN(KIND, BITPREFIX), \
                      GREX_MASKMUL_IMPL(KIND, BITS))
#define GREX_MASKMUL_ALL(REGISTERBITS, BITPREFIX) \
  GREX_FOREACH_TYPE(GREX_MASKMUL, REGISTERBITS, BITPREFIX)

// Division.
#define GREX_MASKDIV(KIND, BITS, SIZE, BITPREFIX) \
  GREX_MASKARITH_WRAP(KIND, BITS, SIZE, divide, BITPREFIX##_mask_div, GREX_MASKMULDIV_IMPL)
#define GREX_MASKDIV_ALL(REGISTERBITS, BITPREFIX) \
  GREX_FOREACH_FP_TYPE(GREX_MASKDIV, REGISTERBITS, BITPREFIX)

GREX_FOREACH_X86_64_LEVEL(GREX_MASKADDSUB_ALL)
GREX_FOREACH_X86_64_LEVEL(GREX_MASKMUL_ALL)
GREX_FOREACH_X86_64_LEVEL(GREX_MASKDIV_ALL)

// Binary16 without AVX512-FP16: use the same fallback as above.
#if GREX_F16_NATIVE_ARITHMETIC
#define GREX_MASKADDSUB_F16 GREX_MASKARITH_AVX512
#define GREX_MASKMULDIV_F16 GREX_MASKARITH_AVX512
#else
#define GREX_MASKADDSUB_F16 GREX_MASKADDSUB_FALLBACK
#define GREX_MASKMULDIV_F16 GREX_MASKMULDIV_FALLBACK
#endif

#define GREX_MASKF16_ALL_BASE(BITPREFIX, SIZE) \
  GREX_MASKARITH_WRAP(f, 16, SIZE, add, BITPREFIX##_mask_add, GREX_MASKADDSUB_F16) \
  GREX_MASKARITH_WRAP(f, 16, SIZE, subtract, BITPREFIX##_mask_sub, GREX_MASKADDSUB_F16) \
  GREX_MASKARITH_WRAP(f, 16, SIZE, multiply, BITPREFIX##_mask_mul, GREX_MASKMULDIV_F16) \
  GREX_MASKARITH_WRAP(f, 16, SIZE, divide, BITPREFIX##_mask_div, GREX_MASKMULDIV_F16)
#define GREX_MASKF16_ALL(REGISTERBITS, BITPREFIX) \
  GREX_MASKF16_ALL_BASE(BITPREFIX, GREX_DIVIDE(REGISTERBITS, 16))
GREX_FOREACH_X86_64_LEVEL(GREX_MASKF16_ALL)
} // namespace grex::backend

#include "grex/backend/shared/operations/arithmetic-mask.hpp" // IWYU pragma: export

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_ARITHMETIC_MASK_HPP
