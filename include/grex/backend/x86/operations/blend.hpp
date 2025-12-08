// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_BLEND_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_BLEND_HPP

#include <cstddef>

#include <immintrin.h>

#include "grex/backend/base.hpp"
#include "grex/backend/macros/base.hpp"
#include "grex/backend/macros/for-each.hpp"
#include "grex/backend/x86/instruction-sets.hpp"
#include "grex/backend/x86/macros/for-each.hpp"
#include "grex/backend/x86/macros/intrinsics.hpp"
#include "grex/backend/x86/types.hpp"
#include "grex/base.hpp"

namespace grex::backend {
#define GREX_BLEND_CAST_f(BITS, REGISTERBITS, X) \
  GREX_CAT(GREX_BITPREFIX(REGISTERBITS), _castsi##REGISTERBITS##_, GREX_FP_SUFFIX(BITS))(X)
#define GREX_BLEND_CAST_i(BITS, REGISTERBITS, X) X
#define GREX_BLEND_CAST_u(BITS, REGISTERBITS, X) X
#define GREX_BLEND_CAST(KIND, BITS, REGISTERBITS, X) GREX_BLEND_CAST_##KIND(BITS, REGISTERBITS, X)

#define GREX_BLEND_SIOP(KIND, BITS, BITPREFIX, REGISTERBITS, NAME) \
  GREX_CAT(BITPREFIX##_##NAME##_, GREX_SI_SUFFIX(KIND, BITS, REGISTERBITS))
#define GREX_BLEND_MBINOP(KIND, BITS, BITPREFIX, REGISTERBITS, NAME, MASK, VEC) \
  GREX_BLEND_SIOP(KIND, BITS, BITPREFIX, REGISTERBITS, NAME) \
  (GREX_BLEND_CAST(KIND, BITS, REGISTERBITS, MASK), VEC)

#define GREX_BLENDZ_AVX512(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  inline Vector<KIND##BITS, SIZE> blend_zero(Mask<KIND##BITS, SIZE> m, \
                                             Vector<KIND##BITS, SIZE> v1) { \
    return {.r = GREX_CAT(BITPREFIX##_maskz_mov_, GREX_EPI_SUFFIX(KIND, BITS))(m.r, v1.r)}; \
  }
#define GREX_BLENDZ_BASE(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  inline Vector<KIND##BITS, SIZE> blend_zero(Mask<KIND##BITS, SIZE> m, \
                                             Vector<KIND##BITS, SIZE> v1) { \
    return {.r = GREX_BLEND_MBINOP(KIND, BITS, BITPREFIX, REGISTERBITS, and, m.r, v1.r)}; \
  }

#define GREX_BLEND_AVX512(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  inline Vector<KIND##BITS, SIZE> blend(Mask<KIND##BITS, SIZE> m, Vector<KIND##BITS, SIZE> v0, \
                                        Vector<KIND##BITS, SIZE> v1) { \
    return {.r = GREX_CAT(BITPREFIX##_mask_mov_, GREX_EPI_SUFFIX(KIND, BITS))(v0.r, m.r, v1.r)}; \
  }
#define GREX_BLEND_SSE4(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  inline Vector<KIND##BITS, SIZE> blend(Mask<KIND##BITS, SIZE> m, Vector<KIND##BITS, SIZE> v0, \
                                        Vector<KIND##BITS, SIZE> v1) { \
    return {.r = GREX_CAT(BITPREFIX##_blendv_, GREX_EPI8_SUFFIX(KIND, BITS))( \
              v0.r, v1.r, GREX_BLEND_CAST(KIND, BITS, REGISTERBITS, m.r))}; \
  }
#define GREX_BLEND_BASE(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  inline Vector<KIND##BITS, SIZE> blend(Mask<KIND##BITS, SIZE> m, Vector<KIND##BITS, SIZE> v0, \
                                        Vector<KIND##BITS, SIZE> v1) { \
    return {.r = GREX_BLEND_SIOP(KIND, BITS, BITPREFIX, REGISTERBITS, or)( \
              GREX_BLEND_MBINOP(KIND, BITS, BITPREFIX, REGISTERBITS, andnot, m.r, v0.r), \
              GREX_BLEND_MBINOP(KIND, BITS, BITPREFIX, REGISTERBITS, and, m.r, v1.r))}; \
  }

#if GREX_X86_64_LEVEL >= 4
#define GREX_BLENDZ GREX_BLENDZ_AVX512
#else
#define GREX_BLENDZ GREX_BLENDZ_BASE
#endif

#if GREX_X86_64_LEVEL >= 4
#define GREX_BLEND GREX_BLEND_AVX512
#elif GREX_X86_64_LEVEL >= 2
#define GREX_BLEND GREX_BLEND_SSE4
#else
#define GREX_BLEND GREX_BLEND_BASE
#endif

#define GREX_BLEND_ALL(REGISTERBITS, BITPREFIX, MACRO) \
  GREX_FOREACH_TYPE(MACRO, REGISTERBITS, BITPREFIX, REGISTERBITS)
GREX_FOREACH_X86_64_LEVEL(GREX_BLEND_ALL, GREX_BLENDZ)
GREX_FOREACH_X86_64_LEVEL(GREX_BLEND_ALL, GREX_BLEND)
} // namespace grex::backend

#include "grex/backend/shared/operations/blend.hpp"

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_BLEND_HPP
