// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_EXTRACT_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_EXTRACT_HPP

#include <cstddef>

#include "grex/backend/x86/helpers.hpp"
#include "grex/backend/x86/instruction-sets.hpp"
#include "grex/backend/x86/types.hpp"
#include "grex/base/defs.hpp"

#if GREX_X86_64_LEVEL < 4
#include <array>

#include "grex/backend/x86/operations/store.hpp"
#endif

#define GREX_EXTRACT_BASIC_FALLBACK(ELEMENT, SIZE, REGISTER, CALL, CONVERT) \
  inline ELEMENT extract(ELEMENT##x##SIZE v, std::size_t i) { \
    std::array<ELEMENT, SIZE> x{}; \
    store(x.data(), v); \
    return x[i % SIZE]; \
  }
#define GREX_EXTRACT_BASIC_AVX512(ELEMENT, SIZE, REGISTER, CALL, CONVERT) \
  inline ELEMENT extract(ELEMENT##x##SIZE v, std::size_t i) { \
    const REGISTER x = CALL; \
    return CONVERT; \
  }

// Define for floating-point types
#if GREX_X86_64_LEVEL >= 4
#define GREX_EXTRACT_FP_BASIC(TYPE, SIZE, REGISTER, CALL, CONVERT) \
  GREX_EXTRACT_BASIC_AVX512(TYPE, SIZE, REGISTER, CALL, CONVERT)
#else
#define GREX_EXTRACT_FP_BASIC(TYPE, SIZE, REGISTER, CALL, CONVERT) \
  GREX_EXTRACT_BASIC_FALLBACK(TYPE, SIZE, REGISTER, CALL, CONVERT)
#endif
#define GREX_EXTRACT_FP_COMPRESS(TYPE, SIZE, REGISTER, INTRINSIC, MMASK, CONVERT) \
  GREX_EXTRACT_FP_BASIC(TYPE, SIZE, REGISTER, INTRINSIC(MMASK(u16{1} << i), v.r), CONVERT(x))

// Define for integer types
#define GREX_EXTRACT_CVTSI128_8 _mm_cvtsi128_si32
#define GREX_EXTRACT_CVTSI128_16 _mm_cvtsi128_si32
#define GREX_EXTRACT_CVTSI128_32 _mm_cvtsi128_si32
#define GREX_EXTRACT_CVTSI128_64 _mm_cvtsi128_si64
#define GREX_EXTRACT_VALUE_128(BITS) GREX_EXTRACT_CVTSI128_##BITS(x)
#define GREX_EXTRACT_VALUE_256(BITS) GREX_EXTRACT_CVTSI128_##BITS(_mm256_castsi256_si128(x))
#define GREX_EXTRACT_VALUE_512(BITS) GREX_EXTRACT_CVTSI128_##BITS(_mm512_castsi512_si128(x))

#if GREX_X86_64_LEVEL >= 4 && GREX_HAS_AVX512VBMI2
#define GREX_EXTRACT_INT_BASIC(TYPE, SIZE, REGISTER, CALL, CONVERT) \
  GREX_EXTRACT_BASIC_AVX512(TYPE, SIZE, REGISTER, CALL, CONVERT)
#else
#define GREX_EXTRACT_INT_BASIC(TYPE, SIZE, REGISTER, CALL, CONVERT) \
  GREX_EXTRACT_BASIC_FALLBACK(TYPE, SIZE, REGISTER, CALL, CONVERT)
#endif
#define GREX_EXTRACT_INT_COMPRESS(KIND, BITS, SIZE, REGISTERBITS, BITPREFIX) \
  GREX_EXTRACT_INT_BASIC( \
    KIND##BITS, GREX_ELEMENTS(REGISTERBITS, BITS), __m##REGISTERBITS##i, \
    BITPREFIX##_maskz_compress_epi##BITS(GREX_SIZEMMASK(SIZE)(u64{1} << i), v.r), \
    KIND##BITS(GREX_EXTRACT_VALUE_##REGISTERBITS(BITS)))
#define GREX_EXTRACT_INT_ALL(REGISTERBITS, BITPREFIX) \
  GREX_FOREACH_INT_TYPE(GREX_EXTRACT_INT_COMPRESS, REGISTERBITS, REGISTERBITS, BITPREFIX)

// Define mask extraction
#if GREX_X86_64_LEVEL >= 4
// With AVX-512, the mask can just be converted to an integer to get the requested bit
#define GREX_EXTRACT_MASK_IMPL(KIND, BITS, SIZE, UMMASK) \
  inline bool extract(Mask<KIND##BITS, SIZE> v, std::size_t i) { \
    return (UMMASK(UMMASK(v.r) >> i) & 1U) != 0; \
  }
#define GREX_EXTRACT_MASK(KIND, BITS, SIZE) \
  GREX_EXTRACT_MASK_IMPL(KIND, BITS, SIZE, BOOST_PP_CAT(u, GREX_MMASKSIZE(SIZE)))
#else
// Pre-AVX-512, we extract the corresponding part of the vector and compare it to 0
#define GREX_EXTRACT_MASK(KIND, BITS, SIZE) \
  inline bool extract(Mask<KIND##BITS, SIZE> v, std::size_t i) { \
    return extract(Vector<u##BITS, SIZE>{v.r}, i) != 0; \
  }
#endif
#define GREX_EXTRACT_MASK_ALL(REGISTERBITS, BITPREFIX) \
  GREX_FOREACH_TYPE(GREX_EXTRACT_MASK, REGISTERBITS)

namespace grex::backend {
GREX_EXTRACT_FP_COMPRESS(f32, 4, __m128, _mm_maskz_compress_ps, __mmask8, _mm_cvtss_f32)
#if GREX_X86_64_LEVEL >= 3
GREX_EXTRACT_FP_COMPRESS(f32, 8, __m256, _mm256_maskz_compress_ps, __mmask8, _mm256_cvtss_f32)
#endif
#if GREX_X86_64_LEVEL >= 4
GREX_EXTRACT_FP_COMPRESS(f32, 16, __m512, _mm512_maskz_compress_ps, __mmask16, _mm512_cvtss_f32)
#endif

GREX_EXTRACT_FP_BASIC(f64, 2, __m128d, _mm_mask_unpackhi_pd(v.r, __mmask8(i), v.r, v.r),
                      _mm_cvtsd_f64(x))
#if GREX_X86_64_LEVEL >= 3
GREX_EXTRACT_FP_COMPRESS(f64, 4, __m256d, _mm256_maskz_compress_pd, __mmask8, _mm256_cvtsd_f64)
#endif
#if GREX_X86_64_LEVEL >= 4
GREX_EXTRACT_FP_COMPRESS(f64, 8, __m512d, _mm512_maskz_compress_pd, __mmask8, _mm512_cvtsd_f64)
#endif

GREX_FOREACH_X86_64_LEVEL(GREX_EXTRACT_INT_ALL)

GREX_FOREACH_X86_64_LEVEL(GREX_EXTRACT_MASK_ALL)
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_EXTRACT_HPP
