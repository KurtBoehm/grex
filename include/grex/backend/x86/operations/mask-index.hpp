// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_MASK_INDEX_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_MASK_INDEX_HPP

#include <cstddef>

#include <boost/preprocessor.hpp>
#include <immintrin.h>

#include "grex/backend/defs.hpp"
#include "grex/backend/x86/helpers.hpp"
#include "grex/backend/x86/instruction-sets.hpp"
#include "grex/backend/x86/operations/blend.hpp"
#include "grex/backend/x86/operations/set.hpp"
#include "grex/backend/x86/types.hpp"
#include "grex/base/defs.hpp"

namespace grex::backend {
// Since the largest vector size is 64, signed comparisons can be used even with i8
#if GREX_X86_64_LEVEL >= 4
#define GREX_CUTOFF_MASK_2 __mmask8(~(u8(-1) << i))
#define GREX_CUTOFF_MASK_4 __mmask8(~(u8(-1) << i))
#define GREX_CUTOFF_MASK_8 __mmask8(~(u16(-1) << i))
#define GREX_CUTOFF_MASK_16 __mmask16(~(u32(-1) << i))
#define GREX_CUTOFF_MASK_32 __mmask32(~(u64(-1) << i))
#define GREX_CUTOFF_MASK_64 __mmask64((i < 64) ? ~(u64(-1) << i) : u64(-1))
#define GREX_CUTOFF_MASK_IMPL(KIND, BITS, SIZE, ...) GREX_CUTOFF_MASK_##SIZE
#define GREX_SINGLE_MASK_IMPL(KIND, BITS, SIZE, ...) GREX_SIZEMMASK(SIZE)(u64{1} << i)
#else
#define GREX_INDEX_MASK_CMPGT(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS, CMP) \
  BITPREFIX##_cmp##CMP##_epi##BITS(broadcast(i##BITS(i), type_tag<Vector<i##BITS, SIZE>>).r, \
                                   indices(type_tag<Vector<i##BITS, SIZE>>).r)
#define GREX_INDEX_MASK_8 GREX_INDEX_MASK_CMPGT
#define GREX_INDEX_MASK_16 GREX_INDEX_MASK_CMPGT
#define GREX_INDEX_MASK_32 GREX_INDEX_MASK_CMPGT
#define GREX_INDEX_MASK_64_128 _mm_set_epi32(1, 1, 0, 0)
#define GREX_INDEX_MASK_64_256 _mm256_set_epi32(3, 3, 2, 2, 1, 1, 0, 0)
#define GREX_INDEX_MASK_64(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS, CMP) \
  /* perform a 32 bit comparison */ \
  BITPREFIX##_cmp##CMP##_epi32(BITPREFIX##_set1_epi32(i32(i)), GREX_INDEX_MASK_64_##REGISTERBITS)
#define GREX_CUTOFF_MASK_IMPL(KIND, BITS, ...) GREX_INDEX_MASK_##BITS(KIND, BITS, __VA_ARGS__, gt)
#define GREX_SINGLE_MASK_IMPL(KIND, BITS, ...) GREX_INDEX_MASK_##BITS(KIND, BITS, __VA_ARGS__, eq)
#endif

#define GREX_INDEX_MASK(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  inline Mask<KIND##BITS, SIZE> cutoff_mask(std::size_t i, TypeTag<Mask<KIND##BITS, SIZE>>) { \
    return {.r = GREX_CUTOFF_MASK_IMPL(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS)}; \
  } \
  inline Mask<KIND##BITS, SIZE> single_mask(std::size_t i, TypeTag<Mask<KIND##BITS, SIZE>>) { \
    return {.r = GREX_SINGLE_MASK_IMPL(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS)}; \
  }
#define GREX_INDEX_MASK_ALL(REGISTERBITS, BITPREFIX) \
  GREX_FOREACH_TYPE(GREX_INDEX_MASK, REGISTERBITS, BITPREFIX, REGISTERBITS)

#define GREX_CUTOFF_VEC(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  inline Vector<KIND##BITS, SIZE> cutoff(std::size_t i, Vector<KIND##BITS, SIZE> v) { \
    return blend_zero(cutoff_mask(i, type_tag<Mask<KIND##BITS, SIZE>>), v); \
  }
#define GREX_CUTOFF_VEC_ALL(REGISTERBITS, BITPREFIX) \
  GREX_FOREACH_TYPE(GREX_CUTOFF_VEC, REGISTERBITS, BITPREFIX, REGISTERBITS)
GREX_FOREACH_X86_64_LEVEL(GREX_INDEX_MASK_ALL)
GREX_FOREACH_X86_64_LEVEL(GREX_CUTOFF_VEC_ALL)

template<Vectorizable T, std::size_t tPart, std::size_t tSize>
inline SubMask<T, tPart, tSize> cutoff_mask(std::size_t i,
                                            TypeTag<SubMask<T, tPart, tSize>> /*tag*/) {
  return SubMask<T, tPart, tSize>{cutoff_mask(i, type_tag<Mask<T, tSize>>)};
}
template<Vectorizable T, std::size_t tPart, std::size_t tSize>
inline SubVector<T, tPart, tSize> cutoff(std::size_t i, SubVector<T, tPart, tSize> v) {
  return SubVector<T, tPart, tSize>{cutoff(i, v.full)};
}

template<typename THalf>
inline SuperMask<THalf> cutoff_mask(std::size_t i, TypeTag<SuperMask<THalf>> /*tag*/) {
  if (i <= THalf::size) {
    return {.lower = cutoff_mask(i, type_tag<THalf>), .upper = zeros(type_tag<THalf>)};
  }
  return {
    .lower = ones(type_tag<THalf>),
    .upper = cutoff_mask(i - THalf::size, type_tag<THalf>),
  };
}
template<typename THalf>
inline SuperVector<THalf> cutoff(std::size_t i, SuperVector<THalf> v) {
  if (i <= THalf::size) {
    return {.lower = cutoff(i, v.lower), .upper = zeros(type_tag<THalf>)};
  }
  return {.lower = v.lower, .upper = cutoff(i - THalf::size, v.upper)};
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_MASK_INDEX_HPP
