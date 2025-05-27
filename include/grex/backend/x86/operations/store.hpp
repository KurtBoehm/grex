// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_STORE_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_STORE_HPP

#include <array> // IWYU pragma: keep
#include <cstddef>

#include <immintrin.h>

#include "thesauros/types/type-tag.hpp"
#include "thesauros/types/value-tag.hpp" // IWYU pragma: keep

#include "grex/backend/defs.hpp"
#include "grex/backend/x86/helpers.hpp"
#include "grex/backend/x86/instruction-sets.hpp"
#include "grex/backend/x86/operations/mask-index.hpp"
#include "grex/backend/x86/operations/split.hpp" // IWYU pragma: keep
#include "grex/backend/x86/types.hpp"
#include "grex/base/defs.hpp" // IWYU pragma: keep

namespace grex::backend {
// Define the casts
#define GREX_STORE_CAST_f(REGISTERBITS) dst
#define GREX_STORE_CAST_u(REGISTERBITS) reinterpret_cast<__m##REGISTERBITS##i*>(dst)
#define GREX_STORE_CAST_i(REGISTERBITS) reinterpret_cast<__m##REGISTERBITS##i*>(dst)

#define GREX_STORE_BASE(NAME, INFIX, KIND, BITS, SIZE, KINDPREFIX, REGISTERBITS) \
  inline void NAME(KIND##BITS* dst, Vector<KIND##BITS, SIZE> src) { \
    GREX_CAT(KINDPREFIX##_##INFIX##_, GREX_SI_SUFFIX(KIND, BITS, REGISTERBITS)) \
    (GREX_STORE_CAST_##KIND(REGISTERBITS), src.r); \
  }
#define GREX_STORE(...) \
  GREX_STORE_BASE(store, storeu, __VA_ARGS__) \
  GREX_STORE_BASE(store_aligned, store, __VA_ARGS__)

// Partial storing
// AVX-512: Use intrinsics
// 128 bit:
// - Level 3, 32/64 bit: Use maskstore
// - Otherwise: switch-case to use appropriate instructions (maskmove is too slow)
// 256 bit:
// - 32/64 bit: use maskstore
// - Otherwise: Partially store one half only

// 256 bit, 32/64 bit: maskstore with casts
#define GREX_MASKSTORE_CAST_32 reinterpret_cast<int*>(dst)
#define GREX_MASKSTORE_CAST_64 reinterpret_cast<long long*>(dst)
#define GREX_PARTSTORE_MASKSTORE(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  BITPREFIX##_maskstore_epi##BITS(GREX_MASKSTORE_CAST_##BITS, \
                                  cutoff_mask(size, thes::type_tag<Mask<KIND##BITS, SIZE>>).r, \
                                  GREX_KINDCAST(KIND, i, BITS, REGISTERBITS, src.r));
#if GREX_X86_64_LEVEL >= 3
#define GREX_PARTSTORE_128_64(KIND) GREX_PARTSTORE_MASKSTORE(KIND, 64, 2, _mm, 128)
#define GREX_PARTSTORE_128_32(KIND) GREX_PARTSTORE_MASKSTORE(KIND, 32, 4, _mm, 128)
#else
#define GREX_PARTSTORE_128_64(KIND) \
  switch (size) { \
  [[unlikely]] case 0: \
    return; \
  [[likely]] case 1: \
    return _mm_storeu_si64(dst, GREX_KINDCAST(KIND, i, 64, 128, src.r)); \
  [[unlikely]] default: \
    return _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), \
                            GREX_KINDCAST(KIND, i, 64, 128, src.r)); \
  }
#define GREX_PARTSTORE_128_32(KIND) \
  switch (size) { \
  [[unlikely]] case 0: \
    return; \
  case 1: return _mm_storeu_si32(dst, GREX_KINDCAST(KIND, i, 32, 128, src.r)); \
  case 2: return _mm_storeu_si64(dst, GREX_KINDCAST(KIND, i, 32, 128, src.r)); \
  case 3: \
    _mm_storeu_si64(dst, GREX_KINDCAST(KIND, i, 32, 128, src.r)); \
    return _mm_storeu_si32( \
      dst + 2, _mm_castps_si128(_mm_movehl_ps(GREX_KINDCAST(KIND, f, 32, 128, src.r), \
                                              GREX_KINDCAST(KIND, f, 32, 128, src.r)))); \
  [[unlikely]] default: \
    return _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), \
                            GREX_KINDCAST(KIND, i, 32, 128, src.r)); \
  }
#endif
#if GREX_X86_64_LEVEL >= 2
#define GREX_PARTSTORE_128_16(KIND) \
  const std::size_t size2 = size / 2; \
  store_part(reinterpret_cast<KIND##32 *>(dst), Vector<KIND##32, 4>{.r = src.r}, size2); \
  if ((size & 1U) != 0) { \
    const std::size_t idx = size - 1; \
    const __m128i shuf = _mm_set1_epi16(i16(((2 * idx + 1) << 8U) + 2 * idx)); \
    _mm_storeu_si16(dst + idx, _mm_shuffle_epi8(src.r, shuf)); \
  }
#define GREX_PARTSTORE_128_8(KIND) \
  const std::size_t size4 = size / 4; \
  store_part(reinterpret_cast<KIND##32 *>(dst), Vector<KIND##32, 4>{.r = src.r}, size4); \
  if ((size & 3U) != 0) { \
    const std::size_t idx = 4 * size4; \
    const __m128i shufo = _mm_set1_epi32(i32(idx * 0x01010101 + 0x03020100)); \
    const __m128i shuf = _mm_shuffle_epi8(src.r, shufo); \
    if ((size & 2U) != 0) { \
      _mm_storeu_si16(dst + idx, shuf); \
    } \
    if ((size & 1U) != 0) { \
      dst[idx + 2] = _mm_extract_epi8(shuf, 2); \
    } \
  }
#else
#define GREX_PARTSTORE_128_16(KIND) \
  const std::size_t size2 = size / 2; \
  store_part(reinterpret_cast<KIND##32 *>(dst), Vector<KIND##32, 4>{.r = src.r}, size2); \
  if ((size & 1U) != 0) { \
    switch (size2) { \
    case 0: return _mm_storeu_si16(dst, src.r); \
    case 1: dst[2] = _mm_extract_epi16(src.r, 2); return; \
    case 2: dst[4] = _mm_extract_epi16(src.r, 4); return; \
    case 3: dst[6] = _mm_extract_epi16(src.r, 6); return; \
    default: break; \
    } \
  }
#define GREX_PARTSTORE_128_8(KIND) \
  if (size >= 16) [[unlikely]] { \
    store(dst, src); \
    return; \
  } \
  if (size == 0) [[unlikely]] { \
    return; \
  } \
  std::array<KIND##8, 16> buf{}; \
  store(buf.data(), src); \
  std::size_t j = 0; \
  if ((size & 8U) != 0) { \
    reinterpret_cast<u64*>(dst)[0] = reinterpret_cast<u64*>(buf.data())[0]; \
    j += 8; \
  } \
  if ((size & 4U) != 0) { \
    reinterpret_cast<u32*>(dst)[j / 4] = reinterpret_cast<u32*>(buf.data())[j / 4]; \
    j += 4; \
  } \
  if ((size & 2U) != 0) { \
    reinterpret_cast<u16*>(dst)[j / 2] = reinterpret_cast<u16*>(buf.data())[j / 2]; \
    j += 2; \
  } \
  if ((size & 1U) != 0) { \
    dst[j] = buf[j]; \
  }
#endif
// 256 bit: Split
#define GREX_PARTSTORE_SPLIT(KIND, BITS, SIZE, REGISTERBITS, BITPREFIX2, REGISTERBITS2) \
  if (size == 0) [[unlikely]] { \
    return; \
  } \
  if (size >= SIZE) [[unlikely]] { \
    return store(dst, src); \
  } \
  if (size >= GREX_HALF(SIZE)) { \
    store(dst, split(src, thes::index_tag<0>)); \
    store_part(dst + GREX_HALF(SIZE), split(src, thes::index_tag<1>), size - GREX_HALF(SIZE)); \
  } else { \
    store_part(dst, split(src, thes::index_tag<0>), size); \
  } \
// AVX-512: Intrinsics
#define GREX_PARTSTORE_AVX512(KIND, BITS, SIZE, BITPREFIX) \
  GREX_CAT(BITPREFIX##_mask_storeu_, GREX_EPI_SUFFIX(KIND, BITS)) \
  (dst, cutoff_mask(size, thes::type_tag<Mask<KIND##BITS, SIZE>>).r, src.r);
#if GREX_X86_64_LEVEL >= 4
#define GREX_PARTSTORE_128(KIND, BITS, SIZE) GREX_PARTSTORE_AVX512(KIND, BITS, SIZE, _mm)
#define GREX_PARTSTORE_256(KIND, BITS, SIZE) GREX_PARTSTORE_AVX512(KIND, BITS, SIZE, _mm256)
#define GREX_PARTSTORE_512(KIND, BITS, SIZE) GREX_PARTSTORE_AVX512(KIND, BITS, SIZE, _mm512)
#else
#define GREX_PARTSTORE_128(KIND, BITS, SIZE) GREX_PARTSTORE_128_##BITS(KIND)
#define GREX_PARTSTORE_256(...) GREX_PARTSTORE_SPLIT(__VA_ARGS__, 256, _mm, 128)
#define GREX_PARTSTORE_512(...) GREX_PARTSTORE_SPLIT(__VA_ARGS__, 512, _mm256, 256)
#endif

#define GREX_PARTSTORE(KIND, BITS, SIZE, REGISTERBITS) \
  inline void store_part(KIND##BITS* dst, Vector<KIND##BITS, SIZE> src, std::size_t size) { \
    GREX_PARTSTORE_##REGISTERBITS(KIND, BITS, SIZE) \
  }

#define GREX_STORE_ALL(REGISTERBITS, KINDPREFIX) \
  GREX_FOREACH_TYPE(GREX_STORE, REGISTERBITS, KINDPREFIX, REGISTERBITS)
GREX_FOREACH_X86_64_LEVEL(GREX_STORE_ALL)

#define GREX_PARTSTORE_ALL(REGISTERBITS, BITPREFIX) \
  GREX_FOREACH_TYPE(GREX_PARTSTORE, REGISTERBITS, REGISTERBITS)
GREX_FOREACH_X86_64_LEVEL(GREX_PARTSTORE_ALL)

template<typename THalf>
inline void store(typename THalf::Value* dst, SuperVector<THalf> src) {
  store(dst, src.lower);
  store(dst + THalf::size, src.upper);
}
template<typename THalf>
inline void store_part(typename THalf::Value* dst, SuperVector<THalf> src, std::size_t size) {
  if (size <= THalf::size) {
    store_part(dst, src.lower, size);
  }
  store(dst, src.lower);
  store_part(dst + THalf::size, src.upper, size - THalf::size);
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_STORE_HPP
