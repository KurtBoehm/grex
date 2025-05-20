// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_INSERT_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_INSERT_HPP

#include <cstddef>

#include "thesauros/types/type-tag.hpp" // IWYU pragma: keep

#include "grex/backend/x86/helpers.hpp"
#include "grex/backend/x86/instruction-sets.hpp"
#include "grex/backend/x86/operations/blend.hpp" // IWYU pragma: keep
#include "grex/backend/x86/operations/compare.hpp" // IWYU pragma: keep
#include "grex/backend/x86/operations/mask-index.hpp" // IWYU pragma: keep
#include "grex/backend/x86/operations/set.hpp" // IWYU pragma: keep
#include "grex/backend/x86/operations/store.hpp" // IWYU pragma: keep
#include "grex/backend/x86/types.hpp"
#include "grex/base/defs.hpp" // IWYU pragma: keep

namespace grex::backend {
#define GREX_INSERT_AVX512_F32_4 _mm_mask_broadcastss_ps
#define GREX_INSERT_AVX512_F32_8 _mm256_mask_broadcastss_ps
#define GREX_INSERT_AVX512_F32_16 _mm512_mask_broadcastss_ps
#define GREX_INSERT_AVX512_F64_2 _mm_mask_movedup_pd
#define GREX_INSERT_AVX512_F64_4 _mm256_mask_broadcastsd_pd
#define GREX_INSERT_AVX512_F64_8 _mm512_mask_broadcastsd_pd
#define GREX_INSERT_AVX512_FP(KIND, BITS, SIZE, BITPREFIX) \
  GREX_INSERT_AVX512_F##BITS##_##SIZE(vec.r, GREX_SIZEMMASK(SIZE)(1U << index), \
                                      GREX_CAT(_mm_set_s, GREX_FP_LETTER(KIND##BITS))(value))
#define GREX_INSERT_AVX512_INT(KIND, BITS, SIZE, BITPREFIX) \
  BITPREFIX##_mask_set1_epi##BITS(vec.r, GREX_SIZEMMASK(SIZE)(1U << index), value)
#define GREX_INSERT_AVX512_f GREX_INSERT_AVX512_FP
#define GREX_INSERT_AVX512_i GREX_INSERT_AVX512_INT
#define GREX_INSERT_AVX512_u GREX_INSERT_AVX512_INT
#define GREX_INSERT_AVX512(KIND, BITS, SIZE, BITPREFIX) \
  inline Vector<KIND##BITS, SIZE> insert(Vector<KIND##BITS, SIZE> vec, std::size_t index, \
                                         KIND##BITS value) { \
    return {.r = GREX_INSERT_AVX512_##KIND(KIND, BITS, SIZE, BITPREFIX)}; \
  }

#define GREX_INSERT_FALLBACK(KIND, BITS, SIZE, BITPREFIX) \
  inline Vector<KIND##BITS, SIZE> insert(Vector<KIND##BITS, SIZE> vec, std::size_t index, \
                                         KIND##BITS value) { \
    return blend(single_mask(index, thes::type_tag<Mask<KIND##BITS, SIZE>>), \
                 broadcast(value, thes::type_tag<Vector<KIND##BITS, SIZE>>), vec); \
  }

#if GREX_X86_64_LEVEL >= 4
#define GREX_INSERT GREX_INSERT_AVX512
#else
#define GREX_INSERT GREX_INSERT_FALLBACK
#endif

#define GREX_INSERT_ALL(REGISTERBITS, BITPREFIX) \
  GREX_FOREACH_TYPE(GREX_INSERT, REGISTERBITS, BITPREFIX)
GREX_FOREACH_X86_64_LEVEL(GREX_INSERT_ALL)
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_INSERT_HPP
