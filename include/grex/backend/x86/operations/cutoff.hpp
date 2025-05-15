// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_CUTOFF_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_CUTOFF_HPP

#include <cstddef>

#include <boost/preprocessor.hpp>
#include <immintrin.h>

#include "thesauros/types/type-tag.hpp" // IWYU pragma: keep

#include "grex/backend/x86/helpers.hpp"
#include "grex/backend/x86/instruction-sets.hpp"
#include "grex/backend/x86/operations/blend.hpp"
#include "grex/backend/x86/operations/set.hpp"
#include "grex/backend/x86/types.hpp"
#include "grex/base/defs.hpp" // IWYU pragma: keep

namespace grex::backend {
// Since the largest vector size is 64, signed comparisons can be used even with i8
// to determine the cutoff mask
#if GREX_X86_64_LEVEL >= 4
#define GREX_CUTOFF_MASK_IMPL(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  return {.r = BITPREFIX##_cmp_epi##BITS##_mask( \
            indices(thes::type_tag<Vector<i##BITS, SIZE>>).r, \
            broadcast(i, thes::type_tag<Vector<i##BITS, SIZE>>).r, 1)};
#else
#define GREX_CUTOFF_MASK_CMPGT(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  return {.r = BITPREFIX##_cmpgt_epi##BITS(broadcast(i, thes::type_tag<Vector<i##BITS, SIZE>>).r, \
                                           indices(thes::type_tag<Vector<i##BITS, SIZE>>).r)};
#define GREX_CUTOFF_MASK_INT8 GREX_CUTOFF_MASK_CMPGT
#define GREX_CUTOFF_MASK_INT16 GREX_CUTOFF_MASK_CMPGT
#define GREX_CUTOFF_MASK_INT32 GREX_CUTOFF_MASK_CMPGT
#define GREX_CUTOFF_MASK_INT64_128 _mm_set_epi32(1, 1, 0, 0)
#define GREX_CUTOFF_MASK_INT64_256 _mm256_set_epi32(3, 3, 2, 2, 1, 1, 0, 0)
#define GREX_CUTOFF_MASK_INT64(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  /* perform a 32 bit comparison */ \
  return {.r = BITPREFIX##_cmpgt_epi32(BITPREFIX##_set1_epi32(i), \
                                       GREX_CUTOFF_MASK_INT64_##REGISTERBITS)};
#define GREX_CUTOFF_MASK_IMPL_f GREX_CUTOFF_MASK_CMPGT
#define GREX_CUTOFF_MASK_IMPL_i(KIND, BITS, ...) GREX_CUTOFF_MASK_INT##BITS(KIND, BITS, __VA_ARGS__)
#define GREX_CUTOFF_MASK_IMPL_u(KIND, BITS, ...) GREX_CUTOFF_MASK_INT##BITS(KIND, BITS, __VA_ARGS__)
#define GREX_CUTOFF_MASK_IMPL(KIND, ...) GREX_CUTOFF_MASK_IMPL_##KIND(KIND, __VA_ARGS__)
#endif

#define GREX_CUTOFF_MASK(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  inline Mask<KIND##BITS, SIZE> cutoff_mask(std::size_t i, \
                                            thes::TypeTag<Mask<KIND##BITS, SIZE>>) { \
    GREX_CUTOFF_MASK_IMPL(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  }
#define GREX_CUTOFF_MASK_ALL(REGISTERBITS, BITPREFIX) \
  GREX_FOREACH_TYPE(GREX_CUTOFF_MASK, REGISTERBITS, BITPREFIX, REGISTERBITS)

#define GREX_CUTOFF_VEC(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  inline Vector<KIND##BITS, SIZE> cutoff(std::size_t i, Vector<KIND##BITS, SIZE> v) { \
    return blend_zero(cutoff_mask(i, thes::type_tag<Mask<KIND##BITS, SIZE>>), v); \
  }
#define GREX_CUTOFF_VEC_ALL(REGISTERBITS, BITPREFIX) \
  GREX_FOREACH_TYPE(GREX_CUTOFF_VEC, REGISTERBITS, BITPREFIX, REGISTERBITS)

GREX_FOREACH_X86_64_LEVEL(GREX_CUTOFF_MASK_ALL)
GREX_FOREACH_X86_64_LEVEL(GREX_CUTOFF_VEC_ALL)
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_CUTOFF_HPP
