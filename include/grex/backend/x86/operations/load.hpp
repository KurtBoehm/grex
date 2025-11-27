// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_LOAD_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_LOAD_HPP

#include <cstddef>

#include <immintrin.h>

#include "grex/backend/defs.hpp"
#include "grex/backend/macros/base.hpp"
#include "grex/backend/macros/for-each.hpp"
#include "grex/backend/macros/math.hpp"
#include "grex/backend/x86/instruction-sets.hpp"
#include "grex/backend/x86/macros/for-each.hpp"
#include "grex/backend/x86/macros/intrinsics.hpp"
#include "grex/backend/x86/operations/insert.hpp"
#include "grex/backend/x86/operations/set.hpp"
#include "grex/backend/x86/types.hpp"
#include "grex/base.hpp"

#if GREX_X86_64_LEVEL > 2
#include "grex/backend/x86/operations/mask-index.hpp"
#endif
#if GREX_X86_64_LEVEL < 4
#include "grex/backend/x86/operations/intrinsics.hpp"
#endif
#if GREX_X86_64_LEVEL == 3
#include "grex/backend/x86/operations/merge.hpp"
#endif

namespace grex::backend {
// sadly, a cast to “const __m128i*” (or a larger register type) is required for integer vectors
#define GREX_LOAD_CAST_f(REGISTERBITS, X) X
#define GREX_LOAD_CAST_i(REGISTERBITS, X) reinterpret_cast<const __m##REGISTERBITS##i*>(X)
#define GREX_LOAD_CAST_u(REGISTERBITS, X) reinterpret_cast<const __m##REGISTERBITS##i*>(X)
#define GREX_LOAD_CAST(KIND, REGISTERBITS, X) GREX_LOAD_CAST_##KIND(REGISTERBITS, X)

#define GREX_LOAD(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS, NAME, OP) \
  inline Vector<KIND##BITS, SIZE> NAME(const KIND##BITS* ptr, TypeTag<Vector<KIND##BITS, SIZE>>) { \
    return {.r = GREX_CAT(BITPREFIX##_##OP##_, GREX_SI_SUFFIX(KIND, BITS, REGISTERBITS))( \
              GREX_LOAD_CAST(KIND, REGISTERBITS, ptr))}; \
  }

// Partial loading
// AVX-512: Use intrinsics
// 128 bit:
// - Level 3, 32/64 bit: Use maskload
// - Otherwise: Case distinctions to use appropriate instructions
// 256 bit:
// - 32/64 bit: Use maskload
// - Otherwise: Partially load one half only

// 256 bit, 32/64 bit: maskload with casts
#define GREX_MASKLOAD_CAST_32 reinterpret_cast<const int*>(ptr)
#define GREX_MASKLOAD_CAST_64 reinterpret_cast<const long long*>(ptr)
#define GREX_PARTLOAD_MASKLOAD(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  return {.r = GREX_KINDCAST( \
            i, KIND, BITS, REGISTERBITS, \
            BITPREFIX##_maskload_epi##BITS( \
              GREX_MASKLOAD_CAST_##BITS, cutoff_mask(size, type_tag<Mask<KIND##BITS, SIZE>>).r))};
#if GREX_X86_64_LEVEL >= 3
#define GREX_PARTLOAD_128_64(KIND) GREX_PARTLOAD_MASKLOAD(KIND, 64, 2, _mm, 128)
#define GREX_PARTLOAD_128_32(KIND) GREX_PARTLOAD_MASKLOAD(KIND, 32, 4, _mm, 128)
#else
#define GREX_PARTLOAD_128_64(KIND) \
  if (size >= 1) [[likely]] { \
    if (size >= 2) [[unlikely]] { \
      return {.r = GREX_KINDCAST(i, KIND, 64, 128, \
                                 _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr)))}; \
    } \
    return {.r = GREX_KINDCAST(i, KIND, 64, 128, _mm_loadu_si64(ptr))}; \
  } \
  return {.r = GREX_KINDCAST(i, KIND, 64, 128, _mm_setzero_si128())};
#define GREX_PARTLOAD_128_32(KIND) \
  if (size >= 2) { \
    if (size >= 4) [[unlikely]] { \
      return {.r = GREX_KINDCAST(i, KIND, 32, 128, \
                                 _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr)))}; \
    } \
    __m128i out = _mm_loadu_si64(ptr); \
    if (size == 3) { \
      out = _mm_castps_si128( \
        _mm_movelh_ps(_mm_castsi128_ps(out), _mm_castsi128_ps(_mm_loadu_si32(ptr + 2)))); \
    } \
    return {.r = GREX_KINDCAST(i, KIND, 32, 128, out)}; \
  } \
  if (size == 1) [[likely]] { \
    return {.r = GREX_KINDCAST(i, KIND, 32, 128, _mm_loadu_si32(ptr))}; \
  } \
  return {.r = GREX_KINDCAST(i, KIND, 32, 128, _mm_setzero_si128())};
#endif
#define GREX_PARTLOAD_128_16(KIND) \
  const std::size_t size2 = size / 2; \
  __m128i out = \
    load_part(reinterpret_cast<const KIND##32 *>(ptr), size2, type_tag<Vector<KIND##32, 4>>).r; \
  if ((size & 1U) != 0) { \
    switch (size2) { \
      case 0: out = _mm_loadu_si16(ptr); break; \
      case 1: out = mm::insert_epi16(out, ptr[2], int_tag<2>); break; \
      case 2: out = mm::insert_epi16(out, ptr[4], int_tag<4>); break; \
      case 3: out = mm::insert_epi16(out, ptr[6], int_tag<6>); break; \
      default: break; \
    } \
  } \
  return {.r = out};
#define GREX_PARTLOAD_128_8(KIND) \
  const std::size_t size2 = size / 2; \
  __m128i out = \
    load_part(reinterpret_cast<const KIND##16 *>(ptr), size2, type_tag<Vector<KIND##16, 8>>).r; \
  if ((size & 1U) != 0) { \
    const std::size_t i = size - 1; \
    out = insert(Vector<KIND##8, 16>{out}, i, ptr[i]).r; \
  } \
  return {.r = out};
// 256 bit: Split
#define GREX_PARTLOAD_SPLIT(KIND, BITS, SIZE, REGISTERBITS, BITPREFIX2, REGISTERBITS2) \
  if (size == 0) [[unlikely]] { \
    return zeros(type_tag<Vector<KIND##BITS, SIZE>>); \
  } \
  if (size >= SIZE) [[unlikely]] { \
    return load(ptr, type_tag<Vector<KIND##BITS, SIZE>>); \
  } \
  if (size >= GREX_DIVIDE(SIZE, 2)) { \
    return merge(load(ptr, type_tag<Vector<KIND##BITS, GREX_DIVIDE(SIZE, 2)>>), \
                 load_part(ptr + GREX_DIVIDE(SIZE, 2), size - GREX_DIVIDE(SIZE, 2), \
                           type_tag<Vector<KIND##BITS, GREX_DIVIDE(SIZE, 2)>>)); \
  } \
  return merge(load_part(ptr, size, type_tag<Vector<KIND##BITS, GREX_DIVIDE(SIZE, 2)>>), \
               zeros(type_tag<Vector<KIND##BITS, GREX_DIVIDE(SIZE, 2)>>));
// AVX-512: Intrinsics
#define GREX_PARTLOAD_AVX512(KIND, BITS, SIZE, BITPREFIX) \
  return {.r = GREX_CAT(BITPREFIX##_maskz_loadu_, GREX_EPI_SUFFIX(KIND, BITS))( \
            cutoff_mask(size, type_tag<Mask<KIND##BITS, SIZE>>).r, ptr)};
// Case distinction based on register size
#if GREX_X86_64_LEVEL >= 4
#define GREX_PARTLOAD_128(KIND, BITS, SIZE) GREX_PARTLOAD_AVX512(KIND, BITS, SIZE, _mm)
#define GREX_PARTLOAD_256(KIND, BITS, SIZE) GREX_PARTLOAD_AVX512(KIND, BITS, SIZE, _mm256)
#define GREX_PARTLOAD_512(KIND, BITS, SIZE) GREX_PARTLOAD_AVX512(KIND, BITS, SIZE, _mm512)
#else
#define GREX_PARTLOAD_128(KIND, BITS, SIZE) GREX_PARTLOAD_128_##BITS(KIND)
#define GREX_PARTLOAD_256(...) GREX_PARTLOAD_SPLIT(__VA_ARGS__, 256, _mm, 128)
#define GREX_PARTLOAD_512(...) GREX_PARTLOAD_SPLIT(__VA_ARGS__, 512, _mm256, 256)
#endif

#define GREX_PARTLOAD(KIND, BITS, SIZE, REGISTERBITS) \
  inline Vector<KIND##BITS, SIZE> load_part(const KIND##BITS* ptr, std::size_t size, \
                                            TypeTag<Vector<KIND##BITS, SIZE>>) { \
    GREX_PARTLOAD_##REGISTERBITS(KIND, BITS, SIZE) \
  }

#define GREX_LOAD_ALL(REGISTERBITS, BITPREFIX) \
  GREX_FOREACH_TYPE(GREX_LOAD, REGISTERBITS, BITPREFIX, REGISTERBITS, load, loadu) \
  GREX_FOREACH_TYPE(GREX_LOAD, REGISTERBITS, BITPREFIX, REGISTERBITS, load_aligned, load)
GREX_FOREACH_X86_64_LEVEL(GREX_LOAD_ALL)

#define GREX_PARTLOAD_ALL(REGISTERBITS, BITPREFIX) \
  GREX_FOREACH_TYPE(GREX_PARTLOAD, REGISTERBITS, REGISTERBITS)
GREX_FOREACH_X86_64_LEVEL(GREX_PARTLOAD_ALL)

// Sub-native vectors: Separate implementations which ensure to the compiler
// that only the given amount of memory is ever touched
#define GREX_LOAD_SUB_IMPL(NAME, KIND, BITS, PART, SIZE) \
  inline SubVector<KIND##BITS, PART, SIZE> NAME(const KIND##BITS* ptr, \
                                                TypeTag<SubVector<KIND##BITS, PART, SIZE>>) { \
    const __m128i r = GREX_CAT(_mm_loadu_si, GREX_MULTIPLY(BITS, PART))(ptr); \
    return SubVector<KIND##BITS, PART, SIZE>{GREX_KINDCAST(i, KIND, BITS, 128, r)}; \
  }
#define GREX_LOAD_SUB(...) \
  GREX_LOAD_SUB_IMPL(load, __VA_ARGS__) \
  GREX_LOAD_SUB_IMPL(load_aligned, __VA_ARGS__)
GREX_FOREACH_SUB(GREX_LOAD_SUB)

#define GREX_PARTLOAD_SUB_32_2(KIND) \
  using Out = SubVector<KIND##32, 2, 4>; \
  if (size >= 1) [[likely]] { \
    if (size >= 2) [[unlikely]] { \
      return Out{GREX_KINDCAST(i, KIND, 32, 128, _mm_loadu_si64(ptr))}; \
    } \
    return Out{GREX_KINDCAST(i, KIND, 32, 128, _mm_loadu_si32(ptr))}; \
  } \
  return Out{GREX_KINDCAST(i, KIND, 32, 128, _mm_setzero_si128())};
#define GREX_PARTLOAD_SUB_16_4(KIND) \
  using Out = SubVector<KIND##16, 4, 8>; \
  if (size >= 2) { \
    if (size >= 4) [[unlikely]] { \
      return Out{GREX_KINDCAST(i, KIND, 16, 128, _mm_loadu_si64(ptr))}; \
    } \
    __m128i out = _mm_loadu_si32(ptr); \
    if (size == 3) { \
      out = _mm_unpacklo_epi32(out, _mm_loadu_si16(ptr + 2)); \
    } \
    return Out{GREX_KINDCAST(i, KIND, 16, 128, out)}; \
  } \
  if (size == 1) [[likely]] { \
    return Out{GREX_KINDCAST(i, KIND, 16, 128, _mm_loadu_si16(ptr))}; \
  } \
  return Out{GREX_KINDCAST(i, KIND, 16, 128, _mm_setzero_si128())};
#define GREX_PARTLOAD_SUB_16_2(KIND) \
  using Out = SubVector<KIND##16, 2, 8>; \
  if (size >= 1) [[likely]] { \
    if (size >= 2) [[unlikely]] { \
      return Out{GREX_KINDCAST(i, KIND, 16, 128, _mm_loadu_si32(ptr))}; \
    } \
    return Out{GREX_KINDCAST(i, KIND, 16, 128, _mm_loadu_si16(ptr))}; \
  } \
  return Out{GREX_KINDCAST(i, KIND, 16, 128, _mm_setzero_si128())};
#define GREX_PARTLOAD_SUB_8_8(KIND) \
  const std::size_t size2 = size / 2; \
  __m128i out = \
    load_part(reinterpret_cast<const KIND##16 *>(ptr), size2, type_tag<SubVector<KIND##16, 4, 8>>) \
      .full.r; \
  if ((size & 1U) != 0) { \
    const std::size_t i = size - 1; \
    out = insert(Vector<KIND##8, 16>{out}, i, ptr[i]).r; \
  } \
  return SubVector<KIND##8, 8, 16>{out};
#define GREX_PARTLOAD_SUB_8_4(KIND) \
  using Out = SubVector<KIND##8, 4, 16>; \
  if (size >= 2) { \
    if (size >= 4) [[unlikely]] { \
      return Out{GREX_KINDCAST(i, KIND, 8, 128, _mm_loadu_si32(ptr))}; \
    } \
    __m128i out = _mm_loadu_si16(ptr); \
    if (size == 3) { \
      out = insert(Vector<KIND##8, 16>{out}, 2, ptr[2]).r; \
    } \
    return Out{GREX_KINDCAST(i, KIND, 8, 128, out)}; \
  } \
  if (size == 1) [[likely]] { \
    const __m128i bc = _mm_set1_epi8(GREX_KINDCAST_SINGLE(KIND, i, 8, ptr[0])); \
    return Out{GREX_KINDCAST(i, KIND, 8, 128, _mm_and_si128(_mm_set_epi64x(0, 255), bc))}; \
  } \
  return Out{GREX_KINDCAST(i, KIND, 8, 128, _mm_setzero_si128())};
#define GREX_PARTLOAD_SUB_8_2(KIND) \
  using Out = SubVector<KIND##8, 2, 16>; \
  if (size >= 1) [[likely]] { \
    if (size >= 2) [[unlikely]] { \
      return Out{GREX_KINDCAST(i, KIND, 8, 128, _mm_loadu_si16(ptr))}; \
    } \
    const __m128i bc = _mm_set1_epi8(GREX_KINDCAST_SINGLE(KIND, i, 8, ptr[0])); \
    return Out{GREX_KINDCAST(i, KIND, 8, 128, _mm_and_si128(_mm_set_epi64x(0, 255), bc))}; \
  } \
  return Out{GREX_KINDCAST(i, KIND, 8, 128, _mm_setzero_si128())};
#define GREX_PARTLOAD_SUB(KIND, BITS, PART, SIZE) \
  inline SubVector<KIND##BITS, PART, SIZE> load_part(const KIND##BITS* ptr, std::size_t size, \
                                                     TypeTag<SubVector<KIND##BITS, PART, SIZE>>) { \
    GREX_PARTLOAD_SUB_##BITS##_##PART(KIND) \
  }
GREX_FOREACH_SUB(GREX_PARTLOAD_SUB)
} // namespace grex::backend

#include "grex/backend/shared/operations/load.hpp" // IWYU pragma: export

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_LOAD_HPP
