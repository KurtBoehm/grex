// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_LOAD_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_LOAD_HPP

#include <cstddef>

#include "thesauros/types/type-tag.hpp"
#include "thesauros/types/value-tag.hpp"

#include "grex/backend/x86/helpers.hpp"
#include "grex/backend/x86/operations/insert.hpp"
#include "grex/backend/x86/operations/merge.hpp"
#include "grex/backend/x86/operations/set.hpp"
#include "grex/backend/x86/types.hpp"
#include "grex/base/defs.hpp" // IWYU pragma: keep

namespace grex::backend {
// sadly, a cast to “const __m128i*” (or a larger register type) is required for integer vectors
#define GREX_LOAD_CAST_f(REGISTERBITS, X) X
#define GREX_LOAD_CAST_i(REGISTERBITS, X) reinterpret_cast<const __m##REGISTERBITS##i*>(X)
#define GREX_LOAD_CAST_u(REGISTERBITS, X) reinterpret_cast<const __m##REGISTERBITS##i*>(X)
#define GREX_LOAD_CAST(KIND, REGISTERBITS, X) GREX_LOAD_CAST_##KIND(REGISTERBITS, X)

#define GREX_LOAD(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS, NAME, OP) \
  inline Vector<KIND##BITS, SIZE> NAME(const KIND##BITS* ptr, thes::IndexTag<SIZE>) { \
    return {.r = GREX_CAT(BITPREFIX##_##OP##_, GREX_SI_SUFFIX(KIND, BITS, REGISTERBITS))( \
              GREX_LOAD_CAST(KIND, REGISTERBITS, ptr))}; \
  }

// Partial loading
// 128 bit: Case distinctions to use appropriate instructions
// 256 bit: Partially load one half only
// AVX-512: Use intrinsics
#define GREX_PARTLOAD_128_64(KIND) \
  if (size >= 1) { \
    if (size >= 2) [[likely]] { \
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
#define GREX_PARTLOAD_128_16(KIND) \
  const std::size_t size2 = size / 2; \
  __m128i out = load_part(reinterpret_cast<const KIND##32 *>(ptr), size2, thes::index_tag<4>).r; \
  if (size & 1U) { \
    switch (size2) { \
    case 0: out = _mm_loadu_si32(reinterpret_cast<const __m128i*>(ptr)); break; \
    case 1: out = _mm_insert_epi16(out, ptr[2], 2); break; \
    case 2: out = _mm_insert_epi16(out, ptr[4], 4); break; \
    case 3: out = _mm_insert_epi16(out, ptr[6], 6); break; \
    default: break; \
    } \
  } \
  return {.r = out};
#define GREX_PARTLOAD_128_8(KIND) \
  const std::size_t size2 = size / 2; \
  __m128i out = load_part(reinterpret_cast<const KIND##16 *>(ptr), size2, thes::index_tag<8>).r; \
  if (size & 1U) { \
    const std::size_t i = size - 1; \
    out = insert(Vector<KIND##8, 16>{out}, i, ptr[i]).r; \
  } \
  return {.r = out};
#define GREX_PARTLOAD_128(KIND, BITS, SIZE) GREX_PARTLOAD_128_##BITS(KIND)
// 256/512 bit: split
#define GREX_PARTLOAD_SPLIT(KIND, BITS, SIZE, REGISTERBITS, BITPREFIX2, REGISTERBITS2) \
  if (size == 0) [[unlikely]] { \
    return zero(thes::type_tag<Vector<KIND##BITS, SIZE>>); \
  } \
  if (size >= SIZE) [[unlikely]] { \
    return load(ptr, thes::index_tag<SIZE>); \
  } \
  if (size >= GREX_HALF(SIZE)) { \
    return merge( \
      load(ptr, thes::index_tag<GREX_HALF(SIZE)>), \
      load_part(ptr + GREX_HALF(SIZE), size - GREX_HALF(SIZE), thes::index_tag<GREX_HALF(SIZE)>)); \
  } \
  return merge( \
    load_part(ptr + GREX_HALF(SIZE), size - GREX_HALF(SIZE), thes::index_tag<GREX_HALF(SIZE)>), \
    zero(thes::type_tag<Vector<KIND##BITS, GREX_HALF(SIZE)>>));
#define GREX_PARTLOAD_256(...) GREX_PARTLOAD_SPLIT(__VA_ARGS__, 256, _mm, 128)
#define GREX_PARTLOAD_512(...) GREX_PARTLOAD_SPLIT(__VA_ARGS__, 512, _mm256, 256)

#define GREX_PARTLOAD(KIND, BITS, SIZE, REGISTERBITS) \
  inline Vector<KIND##BITS, SIZE> load_part(const KIND##BITS* ptr, std::size_t size, \
                                            thes::IndexTag<SIZE>) { \
    GREX_PARTLOAD_##REGISTERBITS(KIND, BITS, SIZE) \
  }

#define GREX_LOAD_ALL(REGISTERBITS, BITPREFIX) \
  GREX_FOREACH_TYPE(GREX_LOAD, REGISTERBITS, BITPREFIX, REGISTERBITS, load, loadu) \
  GREX_FOREACH_TYPE(GREX_LOAD, REGISTERBITS, BITPREFIX, REGISTERBITS, load_aligned, load)
GREX_FOREACH_X86_64_LEVEL(GREX_LOAD_ALL)

#define GREX_PARTLOAD_ALL(REGISTERBITS, BITPREFIX) \
  GREX_FOREACH_TYPE(GREX_PARTLOAD, REGISTERBITS, REGISTERBITS)
GREX_FOREACH_X86_64_LEVEL(GREX_PARTLOAD_ALL)
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_LOAD_HPP
