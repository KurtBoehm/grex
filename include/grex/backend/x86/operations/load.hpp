// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_LOAD_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_LOAD_HPP

#include <cstddef>

#include "grex/backend/defs.hpp"
#include "grex/backend/x86/helpers.hpp"
#include "grex/backend/x86/instruction-sets.hpp"
#include "grex/backend/x86/operations/insert.hpp" // IWYU pragma: keep
#include "grex/backend/x86/operations/mask-index.hpp"
#include "grex/backend/x86/operations/merge.hpp" // IWYU pragma: keep
#include "grex/backend/x86/operations/set.hpp" // IWYU pragma: keep
#include "grex/backend/x86/types.hpp"
#include "grex/base/defs.hpp" // IWYU pragma: keep

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
    return {.r = GREX_KINDCAST(i, KIND, 64, 128, \
                               _mm_loadu_si64(reinterpret_cast<const __m128i*>(ptr)))}; \
  } \
  return {.r = GREX_KINDCAST(i, KIND, 64, 128, _mm_setzero_si128())};
#define GREX_PARTLOAD_128_32(KIND) \
  if (size >= 2) { \
    if (size >= 4) [[unlikely]] { \
      return {.r = GREX_KINDCAST(i, KIND, 32, 128, \
                                 _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr)))}; \
    } \
    __m128i out = _mm_loadu_si64(reinterpret_cast<const __m128i*>(ptr)); \
    if (size == 3) { \
      out = _mm_castps_si128(_mm_movelh_ps(_mm_castsi128_ps(out), \
                                           _mm_loadu_ps(reinterpret_cast<const f32*>(ptr + 2)))); \
    } \
    return {.r = GREX_KINDCAST(i, KIND, 32, 128, out)}; \
  } \
  if (size == 1) [[likely]] { \
    return {.r = GREX_KINDCAST(i, KIND, 32, 128, \
                               _mm_loadu_si32(reinterpret_cast<const __m128i*>(ptr)))}; \
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
    case 1: out = _mm_insert_epi16(out, ptr[2], 2); break; \
    case 2: out = _mm_insert_epi16(out, ptr[4], 4); break; \
    case 3: out = _mm_insert_epi16(out, ptr[6], 6); break; \
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
  if (size >= GREX_HALF(SIZE)) { \
    return merge(load(ptr, type_tag<Vector<KIND##BITS, GREX_HALF(SIZE)>>), \
                 load_part(ptr + GREX_HALF(SIZE), size - GREX_HALF(SIZE), \
                           type_tag<Vector<KIND##BITS, GREX_HALF(SIZE)>>)); \
  } \
  return merge(load_part(ptr, size, type_tag<Vector<KIND##BITS, GREX_HALF(SIZE)>>), \
               zeros(type_tag<Vector<KIND##BITS, GREX_HALF(SIZE)>>));
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

template<typename THalf>
inline SuperVector<THalf> load(const typename THalf::Value* ptr,
                               TypeTag<SuperVector<THalf>> /*tag*/) {
  return {
    .lower = load(ptr, type_tag<THalf>),
    .upper = load(ptr + THalf::size, type_tag<THalf>),
  };
}
template<typename THalf>
inline SuperVector<THalf> load_part(const typename THalf::Value* ptr, std::size_t size,
                                    TypeTag<SuperVector<THalf>> /*tag*/) {
  if (size <= THalf::size) {
    return {
      .lower = load_part(ptr, size, type_tag<THalf>),
      .upper = zeros(type_tag<THalf>),
    };
  }
  return {
    .lower = load(ptr, type_tag<THalf>),
    .upper = load_part(ptr, size - THalf::size, type_tag<THalf>),
  };
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_LOAD_HPP
