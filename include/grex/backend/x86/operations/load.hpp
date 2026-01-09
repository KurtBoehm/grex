// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_LOAD_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_LOAD_HPP

#include <cstddef>
#include <cstring>

#include <immintrin.h>

#include "grex/backend/base.hpp"
#include "grex/backend/macros/base.hpp"
#include "grex/backend/macros/for-each.hpp"
#include "grex/backend/macros/math.hpp"
#include "grex/backend/x86/instruction-sets.hpp"
#include "grex/backend/x86/macros/for-each.hpp"
#include "grex/backend/x86/macros/intrinsics.hpp"
#include "grex/backend/x86/types.hpp"
#include "grex/base.hpp"

#if GREX_X86_64_LEVEL == 3
#include "grex/backend/x86/operations/merge.hpp"
#endif
#if GREX_X86_64_LEVEL < 4
#include "grex/backend/choosers.hpp"
#include "grex/backend/x86/operations/set.hpp"
#endif
#if GREX_X86_64_LEVEL >= 3
#include "grex/backend/x86/operations/mask-index.hpp"
#endif

namespace grex::backend {
// sadly, a cast to “const __m128i*” (or a larger register type) is required for integer vectors
#define GREX_LOAD_CAST_f(REGISTERBITS) ptr
#define GREX_LOAD_CAST_i(REGISTERBITS) reinterpret_cast<const __m##REGISTERBITS##i*>(ptr)
#define GREX_LOAD_CAST_u(REGISTERBITS) reinterpret_cast<const __m##REGISTERBITS##i*>(ptr)

#define GREX_LOAD_BASE(NAME, INFIX, KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  inline Vector<KIND##BITS, SIZE> NAME(const KIND##BITS* ptr, TypeTag<Vector<KIND##BITS, SIZE>>) { \
    return {.r = GREX_CAT(BITPREFIX##_##INFIX##_, GREX_SI_SUFFIX(KIND, BITS, REGISTERBITS))( \
              GREX_LOAD_CAST_##KIND(REGISTERBITS))}; \
  }
#define GREX_LOAD(...) \
  GREX_LOAD_BASE(load, loadu, __VA_ARGS__) \
  GREX_LOAD_BASE(load_aligned, load, __VA_ARGS__)

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
#define GREX_PARTLOAD_MASKLOAD(KIND, BITS, SIZE, REGISTERBITS, BITPREFIX) \
  return {.r = GREX_KINDCAST( \
            i, KIND, BITS, REGISTERBITS, \
            BITPREFIX##_maskload_epi##BITS( \
              GREX_MASKLOAD_CAST_##BITS, cutoff_mask(size, type_tag<Mask<KIND##BITS, SIZE>>).r))};

#define GREX_PARTLOAD_FALLBACK_INIT(KIND, BITS, SIZE) \
  if (size >= SIZE) [[unlikely]] { \
    return load(ptr, type_tag<VectorFor<KIND##BITS, SIZE>>); \
  } \
  if (size == 0) [[unlikely]] { \
    return zeros(type_tag<VectorFor<KIND##BITS, SIZE>>); \
  }

#define GREX_PARTLOAD_SWITCH(KIND, BITS, SIZE, CASE1_EXPR, DEFAULT_EXPR) \
  switch (size) { \
    [[unlikely]] case 0: \
      return zeros(type_tag<VectorFor<KIND##BITS, SIZE>>); \
    [[likely]] case 1: \
      return CASE1_EXPR; \
    [[unlikely]] default: \
      return DEFAULT_EXPR; \
  }

#if GREX_X86_64_LEVEL >= 3
#define GREX_PARTLOAD_128_64 GREX_PARTLOAD_MASKLOAD
#define GREX_PARTLOAD_128_32 GREX_PARTLOAD_MASKLOAD
#define GREX_PARTLOAD_256_64 GREX_PARTLOAD_MASKLOAD
#define GREX_PARTLOAD_256_32 GREX_PARTLOAD_MASKLOAD
#else
#define GREX_PARTLOAD_128_64(KIND, ...) \
  GREX_PARTLOAD_SWITCH( \
    KIND, 64, 2, {.r = GREX_KINDCAST(i, KIND, 64, 128, _mm_loadu_si64(ptr))}, \
    {.r = \
       GREX_KINDCAST(i, KIND, 64, 128, _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr)))})
#define GREX_PARTLOAD_128_32(KIND, ...) \
  GREX_PARTLOAD_FALLBACK_INIT(KIND, 32, 4) \
  u64 out = 0; \
  if ((size & 1U) != 0) { \
    std::memcpy(&out, ptr + (size / 2 * 2), 4); \
  } \
  if ((size & 2U) != 0) { \
    u64 lo; \
    std::memcpy(&lo, ptr, 8); \
    return {.r = _mm_set_epi64x(std::bit_cast<i64>(out), std::bit_cast<i64>(lo))}; \
  } \
  return {.r = _mm_set_epi64x(0, std::bit_cast<i64>(out))};
#endif
#define GREX_PARTLOAD_128_16(KIND, ...) \
  GREX_PARTLOAD_FALLBACK_INIT(KIND, 16, 8) \
  u64 out = 0; \
  if ((size & 1U) != 0) { \
    std::memcpy(&out, ptr + (size / 2 * 2), 2); \
  } \
  if ((size & 2U) != 0) { \
    out <<= 32; \
    std::memcpy(&out, ptr + (size / 4 * 4), 4); \
  } \
  if ((size & 4U) != 0) { \
    u64 lo; \
    std::memcpy(&lo, ptr, 8); \
    return {.r = _mm_set_epi64x(std::bit_cast<i64>(out), std::bit_cast<i64>(lo))}; \
  } \
  return {.r = _mm_set_epi64x(0, std::bit_cast<i64>(out))};
#define GREX_PARTLOAD_128_8(KIND, ...) \
  GREX_PARTLOAD_FALLBACK_INIT(KIND, 8, 16) \
  u64 out = 0; \
  if ((size & 1U) != 0) { \
    std::memcpy(&out, ptr + (size / 2 * 2), 1); \
  } \
  if ((size & 2U) != 0) { \
    out <<= 16; \
    std::memcpy(&out, ptr + (size / 4 * 4), 2); \
  } \
  if ((size & 4U) != 0) { \
    out <<= 32; \
    std::memcpy(&out, ptr + (size / 8 * 8), 4); \
  } \
  if ((size & 8U) != 0) { \
    u64 lo; \
    std::memcpy(&lo, ptr, 8); \
    return {.r = _mm_set_epi64x(std::bit_cast<i64>(out), std::bit_cast<i64>(lo))}; \
  } \
  return {.r = _mm_set_epi64x(0, std::bit_cast<i64>(out))};

// 256/512 bits: Split
#define GREX_PARTLOAD_SPLIT(KIND, BITS, SIZE, ...) \
  if (size >= SIZE) [[unlikely]] { \
    return load(ptr, type_tag<Vector<KIND##BITS, SIZE>>); \
  } \
  if (size == 0) [[unlikely]] { \
    return zeros(type_tag<Vector<KIND##BITS, SIZE>>); \
  } \
  if (size >= GREX_DIVIDE(SIZE, 2)) { \
    return merge(load(ptr, type_tag<Vector<KIND##BITS, GREX_DIVIDE(SIZE, 2)>>), \
                 load_part(ptr + GREX_DIVIDE(SIZE, 2), size - GREX_DIVIDE(SIZE, 2), \
                           type_tag<Vector<KIND##BITS, GREX_DIVIDE(SIZE, 2)>>)); \
  } \
  return merge(load_part(ptr, size, type_tag<Vector<KIND##BITS, GREX_DIVIDE(SIZE, 2)>>), \
               zeros(type_tag<Vector<KIND##BITS, GREX_DIVIDE(SIZE, 2)>>));
#define GREX_PARTLOAD_256_16 GREX_PARTLOAD_SPLIT
#define GREX_PARTLOAD_256_8 GREX_PARTLOAD_SPLIT

#define GREX_PARTLOAD_AVX512(KIND, BITS, SIZE, REGISTERBITS, BITPREFIX) \
  return {.r = GREX_CAT(BITPREFIX##_maskz_loadu_, GREX_EPI_SUFFIX(KIND, BITS))( \
            cutoff_mask(size, type_tag<Mask<KIND##BITS, SIZE>>).r, ptr)};

#if GREX_X86_64_LEVEL >= 4
#define GREX_PARTLOAD_128 GREX_PARTLOAD_AVX512
#define GREX_PARTLOAD_256 GREX_PARTLOAD_AVX512
#define GREX_PARTLOAD_512 GREX_PARTLOAD_AVX512
#elif GREX_X86_64_LEVEL == 3
#define GREX_PARTLOAD_128(KIND, BITS, ...) GREX_PARTLOAD_128_##BITS(KIND, BITS, __VA_ARGS__)
#define GREX_PARTLOAD_256(KIND, BITS, ...) GREX_PARTLOAD_256_##BITS(KIND, BITS, __VA_ARGS__)
#define GREX_PARTLOAD_512 GREX_PARTLOAD_SPLIT
#else
#define GREX_PARTLOAD_128(KIND, BITS, ...) GREX_PARTLOAD_128_##BITS(KIND)
#define GREX_PARTLOAD_256 GREX_PARTLOAD_SPLIT
#define GREX_PARTLOAD_512 GREX_PARTLOAD_SPLIT
#endif

#define GREX_PARTLOAD(KIND, BITS, SIZE, REGISTERBITS, BITPREFIX) \
  inline Vector<KIND##BITS, SIZE> load_part(const KIND##BITS* ptr, std::size_t size, \
                                            TypeTag<Vector<KIND##BITS, SIZE>>) { \
    GREX_PARTLOAD_##REGISTERBITS(KIND, BITS, SIZE, REGISTERBITS, BITPREFIX) \
  }

#define GREX_LOAD_ALL(REGISTERBITS, BITPREFIX) \
  GREX_FOREACH_TYPE(GREX_LOAD, REGISTERBITS, BITPREFIX, REGISTERBITS)
GREX_FOREACH_X86_64_LEVEL(GREX_LOAD_ALL)

#define GREX_PARTLOAD_ALL(REGISTERBITS, BITPREFIX) \
  GREX_FOREACH_TYPE(GREX_PARTLOAD, REGISTERBITS, REGISTERBITS, BITPREFIX)
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

#if GREX_X86_64_LEVEL <= 3
#define GREX_PARTLOAD_SUB_32_2(KIND) \
  using Out = SubVector<KIND##32, 2, 4>; \
  GREX_PARTLOAD_SWITCH(KIND, 32, 2, Out{GREX_KINDCAST(i, KIND, 32, 128, _mm_loadu_si32(ptr))}, \
                       Out{GREX_KINDCAST(i, KIND, 32, 128, _mm_loadu_si64(ptr))})
#define GREX_PARTLOAD_SUB_16_4(KIND) \
  GREX_PARTLOAD_FALLBACK_INIT(KIND, 16, 4) \
  u64 out = 0; \
  if ((size & 1U) != 0) { \
    std::memcpy(&out, ptr + (size / 2 * 2), 2); \
  } \
  if ((size & 2U) != 0) { \
    out <<= 32; \
    std::memcpy(&out, ptr, 4); \
  } \
  return SubVector<KIND##16, 4, 8>{_mm_set_epi64x(0, std::bit_cast<i64>(out))};
#define GREX_PARTLOAD_SUB_16_2(KIND) \
  using Out = SubVector<KIND##16, 2, 8>; \
  GREX_PARTLOAD_SWITCH(KIND, 16, 2, Out{GREX_KINDCAST(i, KIND, 16, 128, _mm_loadu_si16(ptr))}, \
                       Out{GREX_KINDCAST(i, KIND, 16, 128, _mm_loadu_si32(ptr))})

#define GREX_PARTLOAD_SUB_8_8(KIND) \
  GREX_PARTLOAD_FALLBACK_INIT(KIND, 8, 8) \
  u64 out = 0; \
  if ((size & 1U) != 0) { \
    std::memcpy(&out, ptr + (size / 2 * 2), 1); \
  } \
  if ((size & 2U) != 0) { \
    out <<= 16; \
    std::memcpy(&out, ptr + (size / 4 * 4), 2); \
  } \
  if ((size & 4U) != 0) { \
    out <<= 32; \
    std::memcpy(&out, ptr + (size / 8 * 8), 4); \
  } \
  return SubVector<KIND##8, 8, 16>{_mm_set_epi64x(0, std::bit_cast<i64>(out))};
#define GREX_PARTLOAD_SUB_8_4(KIND) \
  GREX_PARTLOAD_FALLBACK_INIT(KIND, 8, 4) \
  u64 out = 0; \
  if ((size & 1U) != 0) { \
    std::memcpy(&out, ptr + (size / 2 * 2), 1); \
  } \
  if ((size & 2U) != 0) { \
    out <<= 16; \
    std::memcpy(&out, ptr + (size / 4 * 4), 2); \
  } \
  return SubVector<KIND##8, 4, 16>{_mm_set_epi64x(0, std::bit_cast<i64>(out))};
#define GREX_PARTLOAD_SUB_8_2(KIND) \
  using Out = SubVector<KIND##8, 2, 16>; \
  GREX_PARTLOAD_SWITCH( \
    KIND, 8, 2, \
    Out{GREX_KINDCAST(i, KIND, 8, 128, \
                      _mm_and_si128(_mm_set_epi64x(0, 255), \
                                    _mm_set1_epi8(GREX_KINDCAST_SINGLE(KIND, i, 8, ptr[0]))))}, \
    Out{GREX_KINDCAST(i, KIND, 8, 128, _mm_loadu_si16(ptr))})
#define GREX_PARTLOAD_SUB_IMPL(KIND, BITS, PART, SIZE) GREX_PARTLOAD_SUB_##BITS##_##PART(KIND)
#else
#define GREX_PARTLOAD_SUB_IMPL(KIND, BITS, PART, SIZE) \
  return SubVector<KIND##BITS, PART, SIZE>{ \
    load_part(ptr, size, type_tag<Vector<KIND##BITS, SIZE>>)};
#endif
#define GREX_PARTLOAD_SUB(KIND, BITS, PART, SIZE) \
  inline SubVector<KIND##BITS, PART, SIZE> load_part(const KIND##BITS* ptr, std::size_t size, \
                                                     TypeTag<SubVector<KIND##BITS, PART, SIZE>>) { \
    GREX_PARTLOAD_SUB_IMPL(KIND, BITS, PART, SIZE) \
  }
GREX_FOREACH_SUB(GREX_PARTLOAD_SUB)
} // namespace grex::backend

#include "grex/backend/shared/operations/load.hpp" // IWYU pragma: export

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_LOAD_HPP
