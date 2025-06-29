// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_INSERT_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_INSERT_HPP

#include <cstddef>

#include "grex/backend/defs.hpp"
#include "grex/backend/x86/instruction-sets.hpp"
#include "grex/backend/x86/macros/base.hpp"
#include "grex/backend/x86/macros/for-each.hpp"
#include "grex/backend/x86/macros/intrinsics.hpp"
#include "grex/backend/x86/macros/math.hpp"
#include "grex/backend/x86/types.hpp"
#include "grex/base/defs.hpp"

#if GREX_X86_64_LEVEL >= 4
#include <immintrin.h>
#else
#include "grex/backend/x86/operations/blend.hpp"
#include "grex/backend/x86/operations/mask-index.hpp"
#include "grex/backend/x86/operations/set.hpp"
#endif

namespace grex::backend {
// Vector insert:
// - AVX-512: Some sort of masked broadcast
// - Otherwise: A broadcast followed by blending
// AVX-512: Uncharactaristically, the naming of the intrinsics is a mess, leading to convoluted
// case distinctions
#define GREX_VEC_INSERT_AVX512_F32_4 _mm_mask_broadcastss_ps
#define GREX_VEC_INSERT_AVX512_F32_8 _mm256_mask_broadcastss_ps
#define GREX_VEC_INSERT_AVX512_F32_16 _mm512_mask_broadcastss_ps
#define GREX_VEC_INSERT_AVX512_F64_2 _mm_mask_movedup_pd
#define GREX_VEC_INSERT_AVX512_F64_4 _mm256_mask_broadcastsd_pd
#define GREX_VEC_INSERT_AVX512_F64_8 _mm512_mask_broadcastsd_pd
#define GREX_VEC_INSERT_AVX512_FP(KIND, BITS, SIZE, BITPREFIX) \
  GREX_VEC_INSERT_AVX512_F##BITS##_##SIZE( \
    v.r, GREX_MMASK(SIZE)(GREX_CAT(u, GREX_MAX(SIZE, 8)){1} << index), \
    GREX_CAT(_mm_set_s, GREX_FP_LETTER(BITS))(value))
#define GREX_VEC_INSERT_AVX512_INT(KIND, BITS, SIZE, BITPREFIX) \
  BITPREFIX##_mask_set1_epi##BITS( \
    v.r, GREX_MMASK(SIZE)(GREX_CAT(u, GREX_MAX(SIZE, 8)){1} << index), \
    GREX_KINDCAST_SINGLE(KIND, i, BITS, value))
#define GREX_VEC_INSERT_AVX512_f GREX_VEC_INSERT_AVX512_FP
#define GREX_VEC_INSERT_AVX512_i GREX_VEC_INSERT_AVX512_INT
#define GREX_VEC_INSERT_AVX512_u GREX_VEC_INSERT_AVX512_INT
#define GREX_VEC_INSERT_AVX512(KIND, BITS, SIZE, BITPREFIX) \
  inline Vector<KIND##BITS, SIZE> insert(Vector<KIND##BITS, SIZE> v, std::size_t index, \
                                         KIND##BITS value) { \
    return {.r = GREX_VEC_INSERT_AVX512_##KIND(KIND, BITS, SIZE, BITPREFIX)}; \
  }
// Fallback: Broadcast and blend
#define GREX_VEC_INSERT_FALLBACK(KIND, BITS, SIZE, BITPREFIX) \
  inline Vector<KIND##BITS, SIZE> insert(Vector<KIND##BITS, SIZE> v, std::size_t index, \
                                         KIND##BITS value) { \
    return blend(single_mask(index, type_tag<Mask<KIND##BITS, SIZE>>), v, \
                 broadcast(value, type_tag<Vector<KIND##BITS, SIZE>>)); \
  }

// Mask insert:
// - AVX-512: Use bit fiddling to set bit i to Boolean b based on the following formula:
//   (value & ~(u64{1} << bit_index)) + (T{bit_value} << bit_index)
// - Otherwise: Delegate to the corresponding integer insert instruction
// AVX-512
#define GREX_MASK_INSERT_AVX512(KIND, BITS, SIZE, BITPREFIX) \
  inline Mask<KIND##BITS, SIZE> insert(Mask<KIND##BITS, SIZE> m, std::size_t index, bool value) { \
    using Idx = GREX_CAT(u, GREX_MAX(SIZE, 8)); \
    return {.r = GREX_MMASK(SIZE)((m.r & ~(Idx{1} << index)) + (Idx{value} << index))}; \
  }
#define GREX_MASK_INSERT_FALLBACK(KIND, BITS, SIZE, BITPREFIX) \
  inline Mask<KIND##BITS, SIZE> insert(Mask<KIND##BITS, SIZE> m, std::size_t index, bool value) { \
    return {.r = insert(Vector<i##BITS, SIZE>{.r = m.r}, index, -i##BITS(value)).r}; \
  }

#if GREX_X86_64_LEVEL >= 4
#define GREX_VEC_INSERT GREX_VEC_INSERT_AVX512
#define GREX_MASK_INSERT GREX_MASK_INSERT_AVX512
#else
#define GREX_VEC_INSERT GREX_VEC_INSERT_FALLBACK
#define GREX_MASK_INSERT GREX_MASK_INSERT_FALLBACK
#endif

#define GREX_INSERT_ALL(REGISTERBITS, BITPREFIX) \
  GREX_FOREACH_TYPE(GREX_VEC_INSERT, REGISTERBITS, BITPREFIX) \
  GREX_FOREACH_TYPE(GREX_MASK_INSERT, REGISTERBITS, BITPREFIX)
GREX_FOREACH_X86_64_LEVEL(GREX_INSERT_ALL)

// SubVector/SubMask
template<Vectorizable T, std::size_t tPart, std::size_t tSize>
inline SubVector<T, tPart, tSize> insert(SubVector<T, tPart, tSize> v, std::size_t index, T value) {
  return SubVector<T, tPart, tSize>{insert(v.full, index, value)};
}
template<Vectorizable T, std::size_t tPart, std::size_t tSize>
inline SubMask<T, tPart, tSize> insert(SubMask<T, tPart, tSize> v, std::size_t index, bool value) {
  return SubMask<T, tPart, tSize>{insert(v.full, index, value)};
}

// SuperVector/SuperMask
template<typename THalf>
inline SuperVector<THalf> insert(SuperVector<THalf> v, std::size_t index,
                                 typename THalf::Value value) {
  if (index < THalf::size) {
    return {.lower = insert(v.lower, index, value), .upper = v.upper};
  }
  return {.lower = v.lower, .upper = insert(v.upper, index - THalf::size, value)};
}
template<typename THalf>
inline SuperMask<THalf> insert(SuperMask<THalf> m, std::size_t index, bool value) {
  if (index < THalf::size) {
    return {.lower = insert(m.lower, index, value), .upper = m.upper};
  }
  return {.lower = m.lower, .upper = insert(m.upper, index - THalf::size, value)};
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_INSERT_HPP
