// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_STORE_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_STORE_HPP

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

#if GREX_X86_64_LEVEL > 2
#include "grex/backend/x86/operations/mask-index.hpp"
#else
#include <bit>
#endif
#if GREX_X86_64_LEVEL <= 3
#include "grex/backend/x86/operations/reinterpret.hpp"
#endif
#if GREX_X86_64_LEVEL == 3
#include "grex/backend/x86/operations/split.hpp"
#endif

namespace grex::backend {
// Define the casts
#define GREX_STORE_CAST_f(REGISTERBITS) dst
#define GREX_STORE_CAST_i(REGISTERBITS) reinterpret_cast<__m##REGISTERBITS##i*>(dst)
#define GREX_STORE_CAST_u(REGISTERBITS) reinterpret_cast<__m##REGISTERBITS##i*>(dst)

#define GREX_STORE_BASE(NAME, INFIX, KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  inline void NAME(KIND##BITS* dst, Vector<KIND##BITS, SIZE> src) { \
    GREX_CAT(BITPREFIX##_##INFIX##_, GREX_SI_SUFFIX(KIND, BITS, REGISTERBITS)) \
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
#define GREX_PARTSTORE_MASKSTORE(KIND, BITS, SIZE, REGISTERBITS, BITPREFIX) \
  BITPREFIX##_maskstore_epi##BITS(GREX_MASKSTORE_CAST_##BITS, \
                                  cutoff_mask(size, type_tag<Mask<KIND##BITS, SIZE>>).r, \
                                  GREX_KINDCAST(KIND, i, BITS, REGISTERBITS, src.r));

#define GREX_PARTSTORE_FALLBACK_128_INIT(KIND, BITS, SIZE) \
  if (size >= SIZE) [[unlikely]] { \
    store(dst, src); \
    return; \
  } \
  if (size == 0) [[unlikely]] { \
    return; \
  } \
  const auto ru64 = reinterpret<u64>(src).r; \
  if ((size & GREX_DIVIDE(SIZE, 2)) != 0) { \
    _mm_storeu_si64(dst, ru64); \
  } \
  const u64 lo64 = std::bit_cast<u64>(_mm_cvtsi128_si64(ru64)); \
  const u64 hi64 = std::bit_cast<u64>(_mm_cvtsi128_si64(_mm_shuffle_epi32(ru64, 0b11101110))); \
  u64 r64 = (size >= GREX_DIVIDE(SIZE, 2)) ? hi64 : lo64;
#define GREX_PARTSTORE_FALLBACK_SUB_INIT(BITS, PART) \
  const __m128i reg = reinterpret<u##BITS>(src).full.r; \
  if (size >= PART) [[unlikely]] { \
    _mm_storeu_si##BITS(dst, reg); \
    return; \
  } \
  if (size == 0) [[unlikely]] { \
    return; \
  } \
  u##BITS r##BITS = std::bit_cast<u##BITS>(_mm_cvtsi128_si##BITS(reg));
#define GREX_PARTSTORE_FALLBACK_64_INIT(KIND, BITS, PART) GREX_PARTSTORE_FALLBACK_SUB_INIT(64, PART)
#define GREX_PARTSTORE_FALLBACK_32_INIT(KIND, BITS, PART) GREX_PARTSTORE_FALLBACK_SUB_INIT(32, PART)

#define GREX_PARTSTORE_SWITCH(CASE1_STMT, DEFAULT_STMT) \
  switch (size) { \
    [[unlikely]] case 0: \
      return; \
    [[likely]] case 1: \
      CASE1_STMT; \
      return; \
    [[unlikely]] default: \
      DEFAULT_STMT; \
      return; \
  }

#if GREX_X86_64_LEVEL >= 3
#define GREX_PARTSTORE_128_64 GREX_PARTSTORE_MASKSTORE
#define GREX_PARTSTORE_128_32 GREX_PARTSTORE_MASKSTORE
#define GREX_PARTSTORE_256_64 GREX_PARTSTORE_MASKSTORE
#define GREX_PARTSTORE_256_32 GREX_PARTSTORE_MASKSTORE
#else
#define GREX_PARTSTORE_128_64(KIND, ...) \
  GREX_PARTSTORE_SWITCH( \
    _mm_storeu_si64(dst, GREX_KINDCAST(KIND, i, 64, 128, src.r)), \
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), GREX_KINDCAST(KIND, i, 64, 128, src.r)))
#define GREX_PARTSTORE_128_32(KIND, ...) \
  GREX_PARTSTORE_FALLBACK_128_INIT(KIND, 32, 4) \
  if ((size & 1U) != 0) { \
    std::memcpy(dst + (size / 2 * 2), &r64, 4); \
  }
#endif
#define GREX_PARTSTORE_128_16(KIND, ...) \
  GREX_PARTSTORE_FALLBACK_128_INIT(KIND, 16, 8) \
  if ((size & 2U) != 0) { \
    std::memcpy(dst + (size / 4 * 4), &r64, 4); \
    r64 >>= 32; \
  } \
  if ((size & 1U) != 0) { \
    std::memcpy(dst + (size / 2 * 2), &r64, 2); \
  }
#define GREX_PARTSTORE_128_8(KIND, ...) \
  GREX_PARTSTORE_FALLBACK_128_INIT(KIND, 8, 16) \
  if ((size & 4U) != 0) { \
    std::memcpy(dst + (size / 8 * 8), &r64, 4); \
    r64 >>= 32; \
  } \
  if ((size & 2U) != 0) { \
    std::memcpy(dst + (size / 4 * 4), &r64, 2); \
    r64 >>= 16; \
  } \
  if ((size & 1U) != 0) { \
    std::memcpy(dst + (size / 2 * 2), &r64, 1); \
  }

// 256/512 bits: Split
#define GREX_PARTSTORE_SPLIT(KIND, BITS, SIZE, ...) \
  if (size >= SIZE) [[unlikely]] { \
    store(dst, src); \
    return; \
  } \
  if (size == 0) [[unlikely]] { \
    return; \
  } \
  if (size >= GREX_DIVIDE(SIZE, 2)) { \
    store(dst, get_low(src)); \
    store_part(dst + GREX_DIVIDE(SIZE, 2), get_high(src), size - GREX_DIVIDE(SIZE, 2)); \
  } else { \
    store_part(dst, get_low(src), size); \
  }
#define GREX_PARTSTORE_256_16 GREX_PARTSTORE_SPLIT
#define GREX_PARTSTORE_256_8 GREX_PARTSTORE_SPLIT

#define GREX_PARTSTORE_AVX512(KIND, BITS, SIZE, REGISTERBITS, BITPREFIX) \
  GREX_CAT(BITPREFIX##_mask_storeu_, GREX_EPI_SUFFIX(KIND, BITS)) \
  (dst, cutoff_mask(size, type_tag<Mask<KIND##BITS, SIZE>>).r, src.r);

#if GREX_X86_64_LEVEL >= 4
#define GREX_PARTSTORE_128 GREX_PARTSTORE_AVX512
#define GREX_PARTSTORE_256 GREX_PARTSTORE_AVX512
#define GREX_PARTSTORE_512 GREX_PARTSTORE_AVX512
#elif GREX_X86_64_LEVEL == 3
#define GREX_PARTSTORE_128(KIND, BITS, ...) GREX_PARTSTORE_128_##BITS(KIND, BITS, __VA_ARGS__)
#define GREX_PARTSTORE_256(KIND, BITS, ...) GREX_PARTSTORE_256_##BITS(KIND, BITS, __VA_ARGS__)
#define GREX_PARTSTORE_512 GREX_PARTSTORE_SPLIT
#else
#define GREX_PARTSTORE_128(KIND, BITS, ...) GREX_PARTSTORE_128_##BITS(KIND)
#define GREX_PARTSTORE_256 GREX_PARTSTORE_SPLIT
#define GREX_PARTSTORE_512 GREX_PARTSTORE_SPLIT
#endif

#define GREX_PARTSTORE(KIND, BITS, SIZE, REGISTERBITS, BITPREFIX) \
  inline void store_part(KIND##BITS* dst, Vector<KIND##BITS, SIZE> src, std::size_t size) { \
    GREX_PARTSTORE_##REGISTERBITS(KIND, BITS, SIZE, REGISTERBITS, BITPREFIX) \
  }

#define GREX_STORE_ALL(REGISTERBITS, BITPREFIX) \
  GREX_FOREACH_TYPE(GREX_STORE, REGISTERBITS, BITPREFIX, REGISTERBITS)
GREX_FOREACH_X86_64_LEVEL(GREX_STORE_ALL)

#define GREX_PARTSTORE_ALL(REGISTERBITS, BITPREFIX) \
  GREX_FOREACH_TYPE(GREX_PARTSTORE, REGISTERBITS, REGISTERBITS, BITPREFIX)
GREX_FOREACH_X86_64_LEVEL(GREX_PARTSTORE_ALL)

// Sub-native vectors: Separate implementations which ensure to the compiler
// that only the given amount of memory is ever touched
#define GREX_STORE_SUB_IMPL(NAME, KIND, BITS, PART, SIZE) \
  inline void NAME(KIND##BITS* dst, SubVector<KIND##BITS, PART, SIZE> src) { \
    const __m128i r = GREX_KINDCAST(KIND, i, BITS, 128, src.full.r); \
    GREX_CAT(_mm_storeu_si, GREX_MULTIPLY(BITS, PART))(dst, r); \
  }
#define GREX_STORE_SUB(...) \
  GREX_STORE_SUB_IMPL(store, __VA_ARGS__) \
  GREX_STORE_SUB_IMPL(store_aligned, __VA_ARGS__)
GREX_FOREACH_SUB(GREX_STORE_SUB)

#if GREX_X86_64_LEVEL <= 3
#define GREX_PARTSTORE_SUB_32_2(KIND) \
  GREX_PARTSTORE_SWITCH(_mm_storeu_si32(dst, GREX_KINDCAST(KIND, i, 32, 128, src.full.r)), \
                        _mm_storeu_si64(dst, GREX_KINDCAST(KIND, i, 32, 128, src.full.r)))
#define GREX_PARTSTORE_SUB_16_4(KIND) \
  GREX_PARTSTORE_FALLBACK_64_INIT(KIND, 16, 4) \
  if ((size & 2U) != 0) { \
    std::memcpy(dst, &r64, 4); \
    r64 >>= 32; \
  } \
  if ((size & 1U) != 0) { \
    std::memcpy(dst + (size / 2 * 2), &r64, 2); \
  }
#define GREX_PARTSTORE_SUB_16_2(KIND) \
  GREX_PARTSTORE_SWITCH(_mm_storeu_si16(dst, src.full.r), _mm_storeu_si32(dst, src.full.r))

#define GREX_PARTSTORE_SUB_8_8(KIND) \
  GREX_PARTSTORE_FALLBACK_64_INIT(KIND, 8, 8) \
  if ((size & 4U) != 0) { \
    std::memcpy(dst, &r64, 4); \
    r64 >>= 32; \
  } \
  if ((size & 2U) != 0) { \
    std::memcpy(dst + (size / 4 * 4), &r64, 2); \
    r64 >>= 16; \
  } \
  if ((size & 1U) != 0) { \
    std::memcpy(dst + (size / 2 * 2), &r64, 1); \
  }
#define GREX_PARTSTORE_SUB_8_4(KIND) \
  GREX_PARTSTORE_FALLBACK_32_INIT(KIND, 8, 4) \
  if ((size & 2U) != 0) { \
    std::memcpy(dst, &r32, 2); \
    r32 >>= 16; \
  } \
  if ((size & 1U) != 0) { \
    std::memcpy(dst + (size / 2 * 2), &r32, 1); \
  }
#define GREX_PARTSTORE_SUB_8_2(KIND) \
  GREX_PARTSTORE_SWITCH(dst[0] = KIND##8(_mm_cvtsi128_si32(src.full.r)), \
                        _mm_storeu_si16(dst, src.full.r))
#define GREX_PARTSTORE_SUB_IMPL(KIND, BITS, PART, SIZE) GREX_PARTSTORE_SUB_##BITS##_##PART(KIND)
#else
#define GREX_PARTSTORE_SUB_IMPL(...) return store_part(dst, src.full, size);
#endif
#define GREX_PARTSTORE_SUB(KIND, BITS, PART, SIZE) \
  inline void store_part(KIND##BITS* dst, SubVector<KIND##BITS, PART, SIZE> src, \
                         std::size_t size) { \
    GREX_PARTSTORE_SUB_IMPL(KIND, BITS, PART, SIZE) \
  }
GREX_FOREACH_SUB(GREX_PARTSTORE_SUB)
} // namespace grex::backend

#include "grex/backend/shared/operations/store.hpp" // IWYU pragma: export

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_STORE_HPP
