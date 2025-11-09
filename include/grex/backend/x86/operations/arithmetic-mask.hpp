// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_ARITHMETIC_MASK_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_ARITHMETIC_MASK_HPP

#include <cstddef>

#include <immintrin.h>

#include "grex/backend/defs.hpp"
#include "grex/backend/macros/base.hpp"
#include "grex/backend/macros/for-each.hpp"
#include "grex/backend/x86/instruction-sets.hpp"
#include "grex/backend/x86/macros/for-each.hpp"
#include "grex/backend/x86/macros/intrinsics.hpp"
#include "grex/backend/x86/operations/arithmetic.hpp"
#include "grex/backend/x86/operations/blend.hpp"
#include "grex/backend/x86/types.hpp"
#include "grex/base/defs.hpp"

namespace grex::backend {
// AVX-512: Use intrinsics
#define GREX_MASKARITH_AVX512(KIND, BITS, SIZE, NAME, OP) \
  return {.r = GREX_CAT(OP##_, GREX_EPI_SUFFIX(KIND, BITS))(a.r, m.r, a.r, b.r)};

// Addition and subtraction
// Otherwise: Zero-blend and then apply the basic operator
#define GREX_MASKADDSUB_FALLBACK(KIND, BITS, SIZE, NAME, OP) return NAME(a, blend_zero(m, b));

#if GREX_X86_64_LEVEL >= 4
#define GREX_MASKADDSUB_IMPL GREX_MASKARITH_AVX512
#else
#define GREX_MASKADDSUB_IMPL GREX_MASKADDSUB_FALLBACK
#endif

#define GREX_MASKADDSUB_BASE(KIND, BITS, SIZE, NAME, OP) \
  inline Vector<KIND##BITS, SIZE> mask_##NAME( \
    Mask<KIND##BITS, SIZE> m, Vector<KIND##BITS, SIZE> a, Vector<KIND##BITS, SIZE> b) { \
    GREX_MASKADDSUB_IMPL(KIND, BITS, SIZE, NAME, OP) \
  }

#define GREX_MASKADDSUB_ALL(REGISTERBITS, BITPREFIX) \
  GREX_FOREACH_TYPE(GREX_MASKADDSUB_BASE, REGISTERBITS, add, BITPREFIX##_mask_add) \
  GREX_FOREACH_TYPE(GREX_MASKADDSUB_BASE, REGISTERBITS, subtract, BITPREFIX##_mask_sub)

// Multiplication and division
// Fallback: Blend the basic operator with “a”
#define GREX_MASKMULDIV_FALLBACK(KIND, BITS, SIZE, NAME, OP) return blend(m, a, NAME(a, b));

#if GREX_X86_64_LEVEL >= 4
#define GREX_MASKMULDIV_IMPL GREX_MASKARITH_AVX512
#else
#define GREX_MASKMULDIV_IMPL GREX_MASKMULDIV_FALLBACK
#endif

// Multiplication: No instruction for 8 bit integers, always use fallback
#define GREX_MASKMUL_INT8 GREX_MASKMULDIV_FALLBACK
#define GREX_MASKMUL_INT16 GREX_MASKMULDIV_IMPL
#define GREX_MASKMUL_INT32 GREX_MASKMULDIV_IMPL
#define GREX_MASKMUL_INT64 GREX_MASKMULDIV_IMPL

#define GREX_MASKMUL_f(KIND, BITS, SIZE, BITPREFIX) \
  GREX_MASKMULDIV_IMPL(KIND, BITS, SIZE, multiply, BITPREFIX##_mask_mul)
#define GREX_MASKMUL_i(KIND, BITS, SIZE, BITPREFIX) \
  GREX_MASKMUL_INT##BITS(KIND, BITS, SIZE, multiply, BITPREFIX##_mask_mullo)
#define GREX_MASKMUL_u(KIND, BITS, SIZE, BITPREFIX) \
  GREX_MASKMUL_INT##BITS(KIND, BITS, SIZE, multiply, BITPREFIX##_mask_mullo)
#define GREX_MASKMUL(KIND, BITS, SIZE, BITPREFIX) \
  inline Vector<KIND##BITS, SIZE> mask_multiply( \
    Mask<KIND##BITS, SIZE> m, Vector<KIND##BITS, SIZE> a, Vector<KIND##BITS, SIZE> b) { \
    GREX_MASKMUL_##KIND(KIND, BITS, SIZE, BITPREFIX) \
  }
#define GREX_MASKMUL_ALL(REGISTERBITS, BITPREFIX) \
  GREX_FOREACH_TYPE(GREX_MASKMUL, REGISTERBITS, BITPREFIX)

// Division
#define GREX_MASKDIV(KIND, BITS, SIZE, BITPREFIX) \
  inline Vector<KIND##BITS, SIZE> mask_divide( \
    Mask<KIND##BITS, SIZE> m, Vector<KIND##BITS, SIZE> a, Vector<KIND##BITS, SIZE> b) { \
    GREX_MASKMULDIV_IMPL(KIND, BITS, SIZE, divide, BITPREFIX##_mask_div) \
  }
#define GREX_MASKDIV_ALL(REGISTERBITS, BITPREFIX) \
  GREX_FOREACH_FP_TYPE(GREX_MASKDIV, REGISTERBITS, BITPREFIX)

GREX_FOREACH_X86_64_LEVEL(GREX_MASKADDSUB_ALL)
GREX_FOREACH_X86_64_LEVEL(GREX_MASKMUL_ALL)
GREX_FOREACH_X86_64_LEVEL(GREX_MASKDIV_ALL)

#define GREX_MASKARITH_SUB(NAME) \
  template<Vectorizable T, std::size_t tPart, std::size_t tSize> \
  inline SubVector<T, tPart, tSize> NAME(SubMask<T, tPart, tSize> m, SubVector<T, tPart, tSize> a, \
                                         SubVector<T, tPart, tSize> b) { \
    return SubVector<T, tPart, tSize>{NAME(m.full, a.full, b.full)}; \
  }
GREX_MASKARITH_SUB(mask_add)
GREX_MASKARITH_SUB(mask_subtract)
GREX_MASKARITH_SUB(mask_multiply)
GREX_MASKARITH_SUB(mask_divide)

#define GREX_MASKARITH_SUPER(NAME) \
  template<typename TVecHalf, typename TMaskHalf> \
  inline SuperVector<TVecHalf> NAME(SuperMask<TMaskHalf> m, SuperVector<TVecHalf> a, \
                                    SuperVector<TVecHalf> b) { \
    return { \
      .lower = NAME(m.lower, a.lower, b.lower), \
      .upper = NAME(m.upper, a.upper, b.upper), \
    }; \
  }
GREX_MASKARITH_SUPER(mask_add)
GREX_MASKARITH_SUPER(mask_subtract)
GREX_MASKARITH_SUPER(mask_multiply)
GREX_MASKARITH_SUPER(mask_divide)
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_ARITHMETIC_MASK_HPP
