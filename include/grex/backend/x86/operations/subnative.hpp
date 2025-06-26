// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_SUBNATIVE_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_SUBNATIVE_HPP

#include <immintrin.h>

#include "grex/backend/defs.hpp"
#include "grex/backend/x86/helpers.hpp"
#include "grex/backend/x86/instruction-sets.hpp"
#include "grex/backend/x86/macros/base.hpp"
#include "grex/backend/x86/operations/mask-index.hpp"
#include "grex/backend/x86/types.hpp"

namespace grex::backend {
#define GREX_CUTOFF_SUB_64(KIND, BITS, PART) \
  {.r = GREX_KINDCAST(i, KIND, BITS, 128, \
                      _mm_move_epi64(GREX_KINDCAST(KIND, i, BITS, 128, v.registr())))}
#if GREX_X86_64_LEVEL >= 2
#define GREX_CUTOFF_SUB_32(KIND, BITS, PART) \
  {.r = _mm_castps_si128(_mm_blend_ps(_mm_setzero_ps(), _mm_castsi128_ps(v.registr()), 1))}
#define GREX_CUTOFF_SUB_16(KIND, BITS, PART) \
  {.r = _mm_blend_epi16(_mm_setzero_si128(), v.registr(), 1)}
#else
#define GREX_CUTOFF_SUB_32(KIND, BITS, PART) cutoff(PART, v.full)
#define GREX_CUTOFF_SUB_16(KIND, BITS, PART) cutoff(PART, v.full)
#endif
#define GREX_CUTOFF_SUB(KIND, BITS, PART, SIZE) \
  inline Vector<KIND##BITS, SIZE> full_cutoff(SubVector<KIND##BITS, PART, SIZE> v) { \
    return GREX_CAT(GREX_CUTOFF_SUB_, GREX_PARTBITS(BITS, PART))(KIND, BITS, PART); \
  }

GREX_FOREACH_SUB(GREX_CUTOFF_SUB)
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_SUBNATIVE_HPP
