// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_EXTRACT_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_EXTRACT_HPP

#include <array>
#include <cstddef>

#include <immintrin.h>

#include "grex/backend/defs.hpp"
#include "grex/backend/x86/instruction-sets.hpp"
#include "grex/backend/x86/macros/base.hpp"
#include "grex/backend/x86/macros/for-each.hpp"
#include "grex/backend/x86/macros/intrinsics.hpp"
#include "grex/backend/x86/operations/store.hpp"
#include "grex/backend/x86/types.hpp"
#include "grex/base/defs.hpp"

namespace grex::backend {
#define GREX_EXTRACT_BASIC_FALLBACK(ELEMENT, SIZE, CALL, CONVERT) \
  std::array<ELEMENT, SIZE> x{}; \
  store(x.data(), v); \
  return x[i % SIZE];
#define GREX_EXTRACT_BASIC_AVX512(ELEMENT, SIZE, CALL, CONVERT) \
  const auto x = CALL; \
  return CONVERT;
#define GREX_MASKZ_CMPR(KIND, BITS, SIZE, BITPREFIX) \
  GREX_CAT(BITPREFIX##_maskz_compress_, GREX_EPI_SUFFIX(KIND, BITS)) \
  (GREX_SIZEMMASK(SIZE)(u64{1} << i), v.r)

// Define for floating-point types
#if GREX_X86_64_LEVEL >= 4
#define GREX_EXTRACT_FP_BASIC GREX_EXTRACT_BASIC_AVX512
#else
#define GREX_EXTRACT_FP_BASIC GREX_EXTRACT_BASIC_FALLBACK
#endif
#define GREX_EXTRACT_FP_COMPRESS(REGISTERBITS, BITPREFIX, BITS, SIZE, ELETTER) \
  GREX_EXTRACT_FP_BASIC(f##BITS, SIZE, GREX_MASKZ_CMPR(f, BITS, SIZE, BITPREFIX), \
                        BITPREFIX##_cvts##ELETTER##_f##BITS(x))
#define GREX_EXTRACT_F32(...) GREX_EXTRACT_FP_COMPRESS(__VA_ARGS__)
#define GREX_EXTRACT_F64_128(...) \
  GREX_EXTRACT_FP_BASIC(f64, 2, _mm_mask_unpackhi_pd(v.r, __mmask8(i), v.r, v.r), _mm_cvtsd_f64(x))
#define GREX_EXTRACT_F64_256(...) GREX_EXTRACT_FP_COMPRESS(__VA_ARGS__)
#define GREX_EXTRACT_F64_512(...) GREX_EXTRACT_FP_COMPRESS(__VA_ARGS__)
#define GREX_EXTRACT_F64(REGISTERBITS, ...) \
  GREX_EXTRACT_F64_##REGISTERBITS(REGISTERBITS, __VA_ARGS__)

// Define for integer types
#define GREX_EXTRACT_CVTSI128_8 _mm_cvtsi128_si32
#define GREX_EXTRACT_CVTSI128_16 _mm_cvtsi128_si32
#define GREX_EXTRACT_CVTSI128_32 _mm_cvtsi128_si32
#define GREX_EXTRACT_CVTSI128_64 _mm_cvtsi128_si64
#define GREX_EXTRACT_VALUE_128(BITS) GREX_EXTRACT_CVTSI128_##BITS(x)
#define GREX_EXTRACT_VALUE_256(BITS) GREX_EXTRACT_CVTSI128_##BITS(_mm256_castsi256_si128(x))
#define GREX_EXTRACT_VALUE_512(BITS) GREX_EXTRACT_CVTSI128_##BITS(_mm512_castsi512_si128(x))

#if GREX_X86_64_LEVEL >= 4 && GREX_HAS_AVX512VBMI2
#define GREX_EXTRACT_INT_BASIC GREX_EXTRACT_BASIC_AVX512
#else
#define GREX_EXTRACT_INT_BASIC GREX_EXTRACT_BASIC_FALLBACK
#endif
#define GERX_EXTRACT_INT_IMPL(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  GREX_EXTRACT_INT_BASIC(KIND##BITS, SIZE, GREX_MASKZ_CMPR(KIND, BITS, SIZE, BITPREFIX), \
                         KIND##BITS(GREX_EXTRACT_VALUE_##REGISTERBITS(BITS)))

// Merge vector implementations
#define GREX_EXTRACT_VEC_f(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  GREX_EXTRACT_F##BITS(REGISTERBITS, BITPREFIX, BITS, SIZE, GREX_FP_LETTER(BITS))
#define GREX_EXTRACT_VEC_i GERX_EXTRACT_INT_IMPL
#define GREX_EXTRACT_VEC_u GERX_EXTRACT_INT_IMPL
#define GREX_EXTRACT_VEC(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  inline KIND##BITS extract(Vector<KIND##BITS, SIZE> v, std::size_t i) { \
    GREX_EXTRACT_VEC_##KIND(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  }
#define GREX_EXTRACT_VEC_ALL(REGISTERBITS, BITPREFIX) \
  GREX_FOREACH_TYPE(GREX_EXTRACT_VEC, REGISTERBITS, BITPREFIX, REGISTERBITS)

// Define mask extraction
#if GREX_X86_64_LEVEL >= 4
// With AVX-512, the mask can just be converted to an integer to get the requested bit
#define GREX_EXTRACT_MASK_IMPL(KIND, BITS, SIZE, UMMASK) \
  inline bool extract(Mask<KIND##BITS, SIZE> v, std::size_t i) { \
    return (UMMASK(UMMASK(v.r) >> i) & 1U) != 0; \
  }
#define GREX_EXTRACT_MASK(KIND, BITS, SIZE) \
  GREX_EXTRACT_MASK_IMPL(KIND, BITS, SIZE, GREX_CAT(u, GREX_MAX(SIZE, 8)))
#else
// Pre-AVX-512, we extract the corresponding part of the vector and compare it to 0
#define GREX_EXTRACT_MASK(KIND, BITS, SIZE) \
  inline bool extract(Mask<KIND##BITS, SIZE> v, std::size_t i) { \
    return extract(Vector<u##BITS, SIZE>{v.r}, i) != 0; \
  }
#endif
#define GREX_EXTRACT_MASK_ALL(REGISTERBITS, BITPREFIX) \
  GREX_FOREACH_TYPE(GREX_EXTRACT_MASK, REGISTERBITS)

// Instantiate for each vector/mask type
GREX_FOREACH_X86_64_LEVEL(GREX_EXTRACT_VEC_ALL)
GREX_FOREACH_X86_64_LEVEL(GREX_EXTRACT_MASK_ALL)

// SubVector/SubMask
template<Vectorizable T, std::size_t tPart, std::size_t tSize>
inline T extract(SubVector<T, tPart, tSize> v, std::size_t index) {
  return extract(v.full, index);
}
template<Vectorizable T, std::size_t tPart, std::size_t tSize>
inline bool extract(SubMask<T, tPart, tSize> v, std::size_t index) {
  return extract(v.full, index);
}

// SuperVector/SuperMask
template<typename THalf>
inline THalf::Value extract(SuperVector<THalf> v, std::size_t i) {
  if (i < THalf::size) {
    return extract(v.lower, i);
  }
  return extract(v.upper, i - THalf::size);
}
template<typename THalf>
inline bool extract(SuperMask<THalf> m, std::size_t i) {
  if (i < THalf::size) {
    return extract(m.lower, i);
  }
  return extract(m.upper, i - THalf::size);
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_EXTRACT_HPP
