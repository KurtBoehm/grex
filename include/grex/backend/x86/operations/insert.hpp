// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_INSERT_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_INSERT_HPP

#include <cstddef>

#include "grex/backend/base.hpp"
#include "grex/backend/macros/base.hpp"
#include "grex/backend/macros/cast.hpp"
#include "grex/backend/macros/for-each.hpp"
#include "grex/backend/macros/math.hpp"
#include "grex/backend/x86/instruction-sets.hpp"
#include "grex/backend/x86/macros/for-each.hpp"
#include "grex/backend/x86/macros/intrinsics.hpp"
#include "grex/backend/x86/operations/mask-index.hpp"
#include "grex/backend/x86/types.hpp"
#include "grex/base.hpp"

#if GREX_X86_64_LEVEL >= 4
#include <immintrin.h>

#include "grex/backend/x86/operations/expand.hpp"
#else
#include "grex/backend/x86/operations/blend.hpp"
#include "grex/backend/x86/operations/set.hpp"
#endif

namespace grex::backend {
// The mask selecting the single lane `index`, which both implementations below build on.
#define GREX_INSERT_MASK(KIND, BITS, SIZE) \
  single_mask(index, type_tag<NativeMask<KIND##BITS, SIZE>>)

//==================================================================================================
// Vector insert: A masked broadcast with AVX-512, a broadcast followed by a blend otherwise
//==================================================================================================

// Uncharacteristically, the naming of the masked broadcasts is a mess, which forces the case
// distinction below. The floating-point variants broadcast out of the lowest lane of a vector
// register, which `expand_any` fills without a detour through a general-purpose register, whereas
// the integer variants broadcast straight out of one, which is where the value already is.
#define GREX_VEC_INSERT_AVX512_F16_8 _mm_mask_broadcastw_epi16
#define GREX_VEC_INSERT_AVX512_F16_16 _mm256_mask_broadcastw_epi16
#define GREX_VEC_INSERT_AVX512_F16_32 _mm512_mask_broadcastw_epi16
#define GREX_VEC_INSERT_AVX512_F32_4 _mm_mask_broadcastss_ps
#define GREX_VEC_INSERT_AVX512_F32_8 _mm256_mask_broadcastss_ps
#define GREX_VEC_INSERT_AVX512_F32_16 _mm512_mask_broadcastss_ps
#define GREX_VEC_INSERT_AVX512_F64_2 _mm_mask_movedup_pd
#define GREX_VEC_INSERT_AVX512_F64_4 _mm256_mask_broadcastsd_pd
#define GREX_VEC_INSERT_AVX512_F64_8 _mm512_mask_broadcastsd_pd
#define GREX_VEC_INSERT_AVX512_FP(KIND, BITS, SIZE, BITPREFIX) \
  GREX_VEC_INSERT_AVX512_F##BITS##_##SIZE(v.r, GREX_INSERT_MASK(KIND, BITS, SIZE).r, \
                                          expand_any(value, index_tag<GREX_DIVIDE(128, BITS)>).r)
#define GREX_VEC_INSERT_AVX512_INT(KIND, BITS, SIZE, BITPREFIX) \
  BITPREFIX##_mask_set1_epi##BITS(v.r, GREX_INSERT_MASK(KIND, BITS, SIZE).r, \
                                  GREX_KINDCAST_SINGLE(KIND, i, BITS, value))
#define GREX_VEC_INSERT_AVX512_f GREX_VEC_INSERT_AVX512_FP
#define GREX_VEC_INSERT_AVX512_i GREX_VEC_INSERT_AVX512_INT
#define GREX_VEC_INSERT_AVX512_u GREX_VEC_INSERT_AVX512_INT
#define GREX_VEC_INSERT_AVX512(KIND, BITS, SIZE, BITPREFIX) \
  return {.r = GREX_VEC_INSERT_AVX512_##KIND(KIND, BITS, SIZE, BITPREFIX)};

// Binary16 needs no special handling here either, since `broadcast` keeps it in a vector register.
#define GREX_VEC_INSERT_FALLBACK(KIND, BITS, SIZE, BITPREFIX) \
  return blend(GREX_INSERT_MASK(KIND, BITS, SIZE), v, \
               broadcast(value, type_tag<NativeVector<KIND##BITS, SIZE>>));

//==================================================================================================
// Mask insert: Bit fiddling with AVX-512, a delegation to the matching integer vector otherwise
//==================================================================================================

// Set bit `index` to `value` following `(m & ~(1 << index)) | (value << index)`.
// TODO Use the btr instruction explicitly
#define GREX_MASK_INSERT_AVX512(KIND, BITS, SIZE, BITPREFIX) \
  using Idx = GREX_CAT(u, GREX_MAX(SIZE, 8)); \
  return {.r = GREX_MMASK_CAST(SIZE, (m.r & ~(Idx{1} << index)) | (Idx{value} << index))};
#define GREX_MASK_INSERT_FALLBACK(KIND, BITS, SIZE, BITPREFIX) \
  const i##BITS entry = GREX_OPCAST(i, BITS, -i##BITS(value)); \
  return {.r = insert(NativeVector<i##BITS, SIZE>{.r = m.r}, index, entry).r};

//==================================================================================================
// Definitions
//==================================================================================================

#if GREX_X86_64_LEVEL >= 4
#define GREX_VEC_INSERT_IMPL GREX_VEC_INSERT_AVX512
#define GREX_MASK_INSERT_IMPL GREX_MASK_INSERT_AVX512
#else
#define GREX_VEC_INSERT_IMPL GREX_VEC_INSERT_FALLBACK
#define GREX_MASK_INSERT_IMPL GREX_MASK_INSERT_FALLBACK
#endif

#define GREX_VEC_INSERT(KIND, BITS, SIZE, BITPREFIX) \
  inline NativeVector<KIND##BITS, SIZE> insert(NativeVector<KIND##BITS, SIZE> v, \
                                               std::size_t index, KIND##BITS value) { \
    GREX_VEC_INSERT_IMPL(KIND, BITS, SIZE, BITPREFIX) \
  }
#define GREX_MASK_INSERT(KIND, BITS, SIZE, BITPREFIX) \
  inline NativeMask<KIND##BITS, SIZE> insert(NativeMask<KIND##BITS, SIZE> m, std::size_t index, \
                                             bool value) { \
    GREX_MASK_INSERT_IMPL(KIND, BITS, SIZE, BITPREFIX) \
  }

// The two loops cannot be merged into one: The mask fallback delegates to the vector `insert` of
// the same-width signed integer, which every vector definition must therefore precede.
#define GREX_INSERT_ALL(REGISTERBITS, BITPREFIX) \
  GREX_FOREACH_TYPE_EXT(GREX_VEC_INSERT, REGISTERBITS, BITPREFIX) \
  GREX_FOREACH_TYPE_EXT(GREX_MASK_INSERT, REGISTERBITS, BITPREFIX)
GREX_FOREACH_X86_64_LEVEL(GREX_INSERT_ALL)
} // namespace grex::backend

#include "grex/backend/shared/operations/insert.hpp" // IWYU pragma: export

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_INSERT_HPP
