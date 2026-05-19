// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_CONVERT_128_MASK_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_CONVERT_128_MASK_HPP

#include <immintrin.h>

#include "grex/backend/macros/base.hpp"
#include "grex/backend/macros/math.hpp"
#include "grex/backend/x86/instruction-sets.hpp"
#include "grex/backend/x86/macros/types.hpp"
#include "grex/backend/x86/types.hpp" // IWYU pragma: keep

#if GREX_X86_64_LEVEL == 1
#include "grex/backend/choosers.hpp"
#include "grex/backend/x86/operations/convert/128-vector.hpp"
#include "grex/backend/x86/operations/mask-convert.hpp"
#endif
#if GREX_X86_64_LEVEL < 4
#include "grex/backend/base.hpp"
#include "grex/backend/x86/operations/convert/base.hpp"
#include "grex/base.hpp"
#endif

// Conversion implementations for 128-bit masks below level 4.

namespace grex::backend {
// Baseline: route mask conversions through signed integer vectors.
#define GREX_CVTMSK_IMPL_INT(DSTKIND, DSTBITS, SRCKIND, SRCBITS, SIZE) \
  const __m128i r = convert(mask2vector(m), type_tag<i##DSTBITS>).registr(); \
  return GREX_MASK_TYPE(DSTKIND, DSTBITS, SIZE){r};

// Widening by ×2: duplicate elements with `unpacklo`.
#define GREX_CVTMSK_IMPL_DOUBLE(DSTKIND, DSTBITS, SRCKIND, SRCBITS, ...) \
  return {.r = _mm_unpacklo_epi##SRCBITS(m.registr(), m.registr())};

// Widening by ×2 for super-native masks: unpack low/high halves separately.
#define GREX_CVTMSK_IMPL_DOUBLE_SUPER(DSTKIND, DSTBITS, SRCKIND, SRCBITS, ...) \
  return { \
    .lower = {.r = _mm_unpacklo_epi##SRCBITS(m.registr(), m.registr())}, \
    .upper = {.r = _mm_unpackhi_epi##SRCBITS(m.registr(), m.registr())}, \
  };

// Widening by >2: first widen to an intermediate size, then widen again.
#define GREX_CVTMSK_IMPL_HALFINCR(DSTKIND, DSTBITS, SRCKIND, SRCBITS, SIZE) \
  using TempMask = GREX_MASK_TYPE(SRCKIND, SRCBITS, GREX_MULTIPLY(SIZE, 2)); \
  using Half = GREX_CAT(DSTKIND, GREX_DIVIDE(DSTBITS, 2)); \
  const __m128i half = convert(TempMask{m.registr()}, type_tag<Half>).r; \
  return convert(GREX_MASK_TYPE(DSTKIND, GREX_DIVIDE(DSTBITS, 2), SIZE){half}, \
                 type_tag<DSTKIND##DSTBITS>);

// Widening by >2 for super-native masks: same as above, but start from a super-mask.
#define GREX_CVTMSK_IMPL_HALFINCR_SUPER(DSTKIND, DSTBITS, SRCKIND, SRCBITS, SIZE) \
  const __m128i half = convert(m, type_tag<GREX_CAT(DSTKIND, GREX_DIVIDE(DSTBITS, 2))>).r; \
  return convert(GREX_MASK_TYPE(DSTKIND, GREX_DIVIDE(DSTBITS, 2), SIZE){half}, \
                 type_tag<DSTKIND##DSTBITS>);

// Narrowing by ÷2 using `packus` instructions.
#define GREX_CVTMSK_IMPL_HALVE_PACKUS(DSTKIND, DSTBITS, SRCKIND, SRCBITS, SIZE) \
  return GREX_MASK_TYPE(DSTKIND, DSTBITS, SIZE){_mm_packs_epi16(m.r, _mm_setzero_si128())};

#define GREX_CVTMSK_IMPL_HALVE_PACKUS_SUPER(DSTKIND, DSTBITS, SRCKIND, SRCBITS, SIZE) \
  return GREX_MASK_TYPE(DSTKIND, DSTBITS, SIZE){_mm_packs_epi16(m.lower.r, m.upper.r)};

// Narrowing by >2: narrow step-wise via an intermediate size.
#define GREX_CVTMSK_IMPL_HALFDECR(DSTKIND, DSTBITS, SRCKIND, SRCBITS, SIZE) \
  using Half = GREX_CAT(SRCKIND, GREX_DIVIDE(SRCBITS, 2)); \
  const auto half = convert(m, type_tag<Half>); \
  return GREX_MASK_TYPE(DSTKIND, DSTBITS, \
                        SIZE){convert(half, type_tag<DSTKIND##DSTBITS>).registr()};

// Narrowing for sub-native masks: convert the full underlying vector and reuse its register.
#define GREX_CVTMSK_IMPL_DECRSUB(DSTKIND, DSTBITS, SRCKIND, SRCBITS, SIZE) \
  const auto full = convert(m.full, type_tag<DSTKIND##DSTBITS>); \
  return GREX_MASK_TYPE(DSTKIND, DSTBITS, SIZE){full.registr()};

// ×2
#define GREX_CVTMSK_IMPL_16_8_8 GREX_CVTMSK_IMPL_DOUBLE
#define GREX_CVTMSK_IMPL_32_16_4 GREX_CVTMSK_IMPL_DOUBLE
#define GREX_CVTMSK_IMPL_64_32_2 GREX_CVTMSK_IMPL_DOUBLE
#define GREX_CVTMSK_IMPL_16_8_16 GREX_CVTMSK_IMPL_DOUBLE_SUPER
#define GREX_CVTMSK_IMPL_32_16_8 GREX_CVTMSK_IMPL_DOUBLE_SUPER
#define GREX_CVTMSK_IMPL_64_32_4 GREX_CVTMSK_IMPL_DOUBLE_SUPER
// ×4
#define GREX_CVTMSK_IMPL_32_8_4 GREX_CVTMSK_IMPL_HALFINCR
#define GREX_CVTMSK_IMPL_64_16_2 GREX_CVTMSK_IMPL_HALFINCR
#define GREX_CVTMSK_IMPL_32_8_8 GREX_CVTMSK_IMPL_HALFINCR_SUPER
#define GREX_CVTMSK_IMPL_64_16_4 GREX_CVTMSK_IMPL_HALFINCR_SUPER
// ×8
#define GREX_CVTMSK_IMPL_64_8_2 GREX_CVTMSK_IMPL_HALFINCR
#define GREX_CVTMSK_IMPL_64_8_4 GREX_CVTMSK_IMPL_HALFINCR_SUPER

// ÷2
// 8←16
#define GREX_CVTMSK_IMPL_8_16_16 GREX_CVTMSK_IMPL_HALVE_PACKUS_SUPER // super-native
#define GREX_CVTMSK_IMPL_8_16_8 GREX_CVTMSK_IMPL_HALVE_PACKUS
#define GREX_CVTMSK_IMPL_8_16_4 GREX_CVTMSK_IMPL_DECRSUB // sub-native
#define GREX_CVTMSK_IMPL_8_16_2 GREX_CVTMSK_IMPL_DECRSUB // sub-native
// 16←32
#define GREX_CVTMSK_IMPL_16_32_8 GREX_CVTMSK_IMPL_HALVE_PACKUS_SUPER // super-native
#define GREX_CVTMSK_IMPL_16_32_4 GREX_CVTMSK_IMPL_HALVE_PACKUS
#define GREX_CVTMSK_IMPL_16_32_2 GREX_CVTMSK_IMPL_DECRSUB // sub-native
// 32←64
#define GREX_CVTMSK_IMPL_32_64_2 GREX_CVTMSK_IMPL_INT
#define GREX_CVTMSK_IMPL_32_64_4 GREX_CVTMSK_IMPL_INT // super-native

// ÷4
// 8←32
#define GREX_CVTMSK_IMPL_8_32_16 GREX_CVTMSK_IMPL_HALFDECR // super-native
#define GREX_CVTMSK_IMPL_8_32_8 GREX_CVTMSK_IMPL_HALFDECR // super-native
#define GREX_CVTMSK_IMPL_8_32_4 GREX_CVTMSK_IMPL_HALFDECR
#define GREX_CVTMSK_IMPL_8_32_2 GREX_CVTMSK_IMPL_DECRSUB // sub-native
// 16←64
#define GREX_CVTMSK_IMPL_16_64_8 GREX_CVTMSK_IMPL_HALFDECR // super-native
#define GREX_CVTMSK_IMPL_16_64_4 GREX_CVTMSK_IMPL_HALFDECR // super-native
#define GREX_CVTMSK_IMPL_16_64_2 GREX_CVTMSK_IMPL_HALFDECR
// ÷8
#define GREX_CVTMSK_IMPL_8_64_16 GREX_CVTMSK_IMPL_HALFDECR
#define GREX_CVTMSK_IMPL_8_64_8 GREX_CVTMSK_IMPL_HALFDECR
#define GREX_CVTMSK_IMPL_8_64_4 GREX_CVTMSK_IMPL_HALFDECR
#define GREX_CVTMSK_IMPL_8_64_2 GREX_CVTMSK_IMPL_HALFDECR

// Zen 4 prefers unpacking to integer conversions, while Tiger Lake/Arrow Lake are indifferent.

#if GREX_X86_64_LEVEL < 4
// ×2
GREX_CVTMSK(i, 16, i, 8, 8)
GREX_CVTMSK(i, 32, i, 16, 4)
GREX_CVTMSK(i, 64, i, 32, 2)
// ×4
GREX_CVTMSK(i, 32, i, 8, 4)
GREX_CVTMSK(i, 64, i, 16, 2)
// ×8
GREX_CVTMSK(i, 64, i, 8, 2)
#endif

// 256-bit super-native cases.
#if GREX_X86_64_LEVEL < 3
// ×2
GREX_CVTMSK(i, 16, i, 8, 16)
GREX_CVTMSK(i, 32, i, 16, 8)
GREX_CVTMSK(i, 64, i, 32, 4)
// ×4
GREX_CVTMSK(i, 32, i, 8, 8)
GREX_CVTMSK(i, 64, i, 16, 4)
// ×8
GREX_CVTMSK(i, 64, i, 8, 4)
#endif

// Above x86-64-v1 the integer vector conversions are faster than unpack-based sequences.
#if GREX_X86_64_LEVEL == 1
// ÷2
GREX_CVTMSK(i, 8, i, 16, 8)
GREX_CVTMSK(i, 8, i, 16, 4) // sub-native
GREX_CVTMSK(i, 8, i, 16, 2) // sub-native
GREX_CVTMSK(i, 16, i, 32, 4)
GREX_CVTMSK(i, 16, i, 32, 2) // sub-native
GREX_CVTMSK(i, 32, i, 64, 2)
GREX_CVTMSK(i, 32, i, 64, 4) // super-native
// ÷4
GREX_CVTMSK(i, 8, i, 32, 4)
GREX_CVTMSK(i, 8, i, 32, 2) // sub-native
GREX_CVTMSK(i, 16, i, 64, 2)
// ÷8
GREX_CVTMSK(i, 8, i, 64, 2)

// 256-bit super-native
// ÷2
GREX_CVTMSK(i, 8, i, 16, 16) // super-native
GREX_CVTMSK(i, 16, i, 32, 8) // super-native
// ÷4
GREX_CVTMSK(i, 8, i, 32, 8) // super-native
GREX_CVTMSK(i, 16, i, 64, 4) // super-native
// ÷8
GREX_CVTMSK(i, 8, i, 64, 4) // super-native

// 512-bit super-native
// ÷4
GREX_CVTMSK(i, 8, i, 32, 16) // super-native
GREX_CVTMSK(i, 16, i, 64, 8) // super-native
// ÷8
GREX_CVTMSK(i, 8, i, 64, 8) // super-native

// 1024-bit super-native
GREX_CVTMSK(i, 8, i, 64, 16) // super-native
#endif
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_CONVERT_128_MASK_HPP
