// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_BLEND_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_BLEND_HPP

#include <cstddef>

#include <boost/preprocessor.hpp>
#include <immintrin.h>

#include "grex/backend/defs.hpp"
#include "grex/backend/x86/helpers.hpp"
#include "grex/backend/x86/instruction-sets.hpp"
#include "grex/backend/x86/types.hpp"
#include "grex/base/defs.hpp" // IWYU pragma: keep

namespace grex::backend {
#define GREX_BLEND_SIOP(KIND, BITS, BITPREFIX, REGISTERBITS, NAME) \
  GREX_CAT(BITPREFIX##_##NAME##_, GREX_SI_SUFFIX(KIND, BITS, REGISTERBITS))
#define GREX_BLEND_MBINOP(KIND, BITS, BITPREFIX, REGISTERBITS, NAME, MASK, VEC) \
  GREX_BLEND_SIOP(KIND, BITS, BITPREFIX, REGISTERBITS, NAME) \
  (GREX_BROADMASK_CONVERT(KIND, BITS, REGISTERBITS, MASK), VEC)

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
              v0.r, v1.r, GREX_BROADMASK_CONVERT(KIND, BITS, REGISTERBITS, m.r))}; \
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

template<Vectorizable T, std::size_t tPart, std::size_t tSize>
inline SubVector<T, tPart, tSize> blend_zero(SubMask<T, tPart, tSize> m,
                                             SubVector<T, tPart, tSize> v1) {
  return {.full = blend_zero(m.full, v1.full)};
}
template<Vectorizable T, std::size_t tPart, std::size_t tSize>
inline SubVector<T, tPart, tSize> blend(SubMask<T, tPart, tSize> m, SubVector<T, tPart, tSize> v0,
                                        SubVector<T, tPart, tSize> v1) {
  return {.full = blend(m.full, v0.full, v1.full)};
}

template<typename TVecHalf, typename TMaskHalf>
inline SuperVector<TVecHalf> blend_zero(SuperMask<TMaskHalf> m, SuperVector<TVecHalf> v1) {
  return {.lower = blend_zero(m.lower, v1.lower), .upper = blend_zero(m.upper, v1.upper)};
}
template<typename TVecHalf, typename TMaskHalf>
inline SuperVector<TVecHalf> blend(SuperMask<TMaskHalf> m, SuperVector<TVecHalf> v0,
                                   SuperVector<TVecHalf> v1) {
  return {.lower = blend(m.lower, v0.lower, v1.lower), .upper = blend(m.upper, v0.upper, v1.upper)};
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_BLEND_HPP
