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
//--------------------------------------------------------------------------------------------------
// Implementations, which all leave the value of `a` in the masked-off lanes
//--------------------------------------------------------------------------------------------------

// Wrapper for all variants, which differ only in the body produced by `IMPL`.
#define GREX_MASKARITH_WRAP(KIND, BITS, SIZE, NAME, OP, IMPL) \
  inline NativeVector<KIND##BITS, SIZE> mask_##NAME(NativeMask<KIND##BITS, SIZE> m, \
                                                    NativeVector<KIND##BITS, SIZE> a, \
                                                    NativeVector<KIND##BITS, SIZE> b) { \
    IMPL(KIND, BITS, SIZE, NAME, OP) \
  }

// AVX-512 (AVX512-FP16 for binary16): Use the masked intrinsics, passing `a` through where the
// mask is false.
#define GREX_MASKARITH_AVX512(KIND, BITS, SIZE, NAME, OP) \
  const auto ar = from_stored<KIND##BITS>(a.r); \
  const auto br = from_stored<KIND##BITS>(b.r); \
  const auto r = GREX_CAT(OP##_, GREX_EPI_SUFFIX(KIND, BITS))(ar, m.r, ar, br); \
  return {.r = to_stored<KIND##BITS>(r)};

// Zeroing fallback: Zero the masked-off lanes of `b` and apply the operation unconditionally, which
// relies on zero being a neutral element that reproduces `a` exactly.
// This holds for `a - (+0)` for every floating-point value, negative zero included, and for both
// integer operations, where there is no signed zero. It fails for floating-point addition, since
// `(-0) + (+0)` is `+0`, and multiplication and division have no zero-patterned neutral element.
#define GREX_MASKARITH_ZERO_FALLBACK(KIND, BITS, SIZE, NAME, OP) return NAME(a, blend_zero(m, b));
// Blend fallback: Carry the operation out unconditionally and blend the result with `a`, which
// reproduces the masked-off lanes of `a` bit for bit, but costs three instructions rather than one
// on x86-64-v1, which has no blend instruction.
#define GREX_MASKARITH_BLEND_FALLBACK(KIND, BITS, SIZE, NAME, OP) return blend(m, a, NAME(a, b));

//--------------------------------------------------------------------------------------------------
// Case distinctions between the implementations
//--------------------------------------------------------------------------------------------------

// Addition needs the blend for floating-point values only; see the fallbacks above.
#if GREX_X86_64_LEVEL >= 4
#define GREX_MASKADD_IMPL_f GREX_MASKARITH_AVX512
#define GREX_MASKADD_IMPL_i GREX_MASKARITH_AVX512
#define GREX_MASKADD_IMPL_u GREX_MASKARITH_AVX512
#define GREX_MASKSUB_IMPL GREX_MASKARITH_AVX512
#define GREX_MASKMULDIV_IMPL GREX_MASKARITH_AVX512
#else
#define GREX_MASKADD_IMPL_f GREX_MASKARITH_BLEND_FALLBACK
#define GREX_MASKADD_IMPL_i GREX_MASKARITH_ZERO_FALLBACK
#define GREX_MASKADD_IMPL_u GREX_MASKARITH_ZERO_FALLBACK
#define GREX_MASKSUB_IMPL GREX_MASKARITH_ZERO_FALLBACK
#define GREX_MASKMULDIV_IMPL GREX_MASKARITH_BLEND_FALLBACK
#endif
#define GREX_MASKADD_IMPL(KIND) GREX_MASKADD_IMPL_##KIND

// Multiplication: there is no masked 8-bit multiplication, since even the unmasked one is emulated.
#define GREX_MASKMUL_IMPL_INT8 GREX_MASKARITH_BLEND_FALLBACK
#define GREX_MASKMUL_IMPL_INT16 GREX_MASKMULDIV_IMPL
#define GREX_MASKMUL_IMPL_INT32 GREX_MASKMULDIV_IMPL
#define GREX_MASKMUL_IMPL_INT64 GREX_MASKMULDIV_IMPL
#define GREX_MASKMUL_IMPL_f(BITS) GREX_MASKMULDIV_IMPL
#define GREX_MASKMUL_IMPL_i(BITS) GREX_MASKMUL_IMPL_INT##BITS
#define GREX_MASKMUL_IMPL_u(BITS) GREX_MASKMUL_IMPL_INT##BITS
#define GREX_MASKMUL_IMPL(KIND, BITS) GREX_MASKMUL_IMPL_##KIND(BITS)

// Multiplication: integers use the truncating `mullo` variant, just like the unmasked operation.
#define GREX_MASKMUL_INTRIN_f(BITPREFIX) BITPREFIX##_mask_mul
#define GREX_MASKMUL_INTRIN_i(BITPREFIX) BITPREFIX##_mask_mullo
#define GREX_MASKMUL_INTRIN_u(BITPREFIX) BITPREFIX##_mask_mullo
#define GREX_MASKMUL_INTRIN(KIND, BITPREFIX) GREX_MASKMUL_INTRIN_##KIND(BITPREFIX)

//--------------------------------------------------------------------------------------------------
// Instantiation for every type and register size
//--------------------------------------------------------------------------------------------------

// Addition and subtraction.
#define GREX_MASKADD_MAIN(KIND, BITS, SIZE, NAME, OP) \
  GREX_MASKARITH_WRAP(KIND, BITS, SIZE, NAME, OP, GREX_MASKADD_IMPL(KIND))
#define GREX_MASKSUB_MAIN(...) GREX_MASKARITH_WRAP(__VA_ARGS__, GREX_MASKSUB_IMPL)
#define GREX_MASKADDSUB_ALL(REGISTERBITS, BITPREFIX) \
  GREX_FOREACH_TYPE(GREX_MASKADD_MAIN, REGISTERBITS, add, BITPREFIX##_mask_add) \
  GREX_FOREACH_TYPE(GREX_MASKSUB_MAIN, REGISTERBITS, subtract, BITPREFIX##_mask_sub)

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

//--------------------------------------------------------------------------------------------------
// Binary16
//--------------------------------------------------------------------------------------------------

// The masked intrinsics are used with AVX512-FP16 and the same fallbacks as above otherwise, which
// inherit the binary32 emulation from the unmasked operation.
#if GREX_F16_NATIVE_ARITHMETIC
#define GREX_MASKADD_F16 GREX_MASKARITH_AVX512
#define GREX_MASKSUB_F16 GREX_MASKARITH_AVX512
#define GREX_MASKMULDIV_F16 GREX_MASKARITH_AVX512
#else
#define GREX_MASKADD_F16 GREX_MASKARITH_BLEND_FALLBACK
#define GREX_MASKSUB_F16 GREX_MASKARITH_ZERO_FALLBACK
#define GREX_MASKMULDIV_F16 GREX_MASKARITH_BLEND_FALLBACK
#endif

#define GREX_MASKF16_ALL_BASE(BITPREFIX, SIZE) \
  GREX_MASKARITH_WRAP(f, 16, SIZE, add, BITPREFIX##_mask_add, GREX_MASKADD_F16) \
  GREX_MASKARITH_WRAP(f, 16, SIZE, subtract, BITPREFIX##_mask_sub, GREX_MASKSUB_F16) \
  GREX_MASKARITH_WRAP(f, 16, SIZE, multiply, BITPREFIX##_mask_mul, GREX_MASKMULDIV_F16) \
  GREX_MASKARITH_WRAP(f, 16, SIZE, divide, BITPREFIX##_mask_div, GREX_MASKMULDIV_F16)
#define GREX_MASKF16_ALL(REGISTERBITS, BITPREFIX) \
  GREX_MASKF16_ALL_BASE(BITPREFIX, GREX_DIVIDE(REGISTERBITS, 16))
GREX_FOREACH_X86_64_LEVEL(GREX_MASKF16_ALL)
} // namespace grex::backend

#include "grex/backend/shared/operations/arithmetic-mask.hpp" // IWYU pragma: export

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_ARITHMETIC_MASK_HPP
