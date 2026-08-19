// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_STORE_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_STORE_HPP

#include <cstddef>

#include <immintrin.h>

#include "grex/backend/base.hpp"
#include "grex/backend/macros/base.hpp"
#include "grex/backend/macros/for-each.hpp"
#include "grex/backend/macros/math.hpp"
#include "grex/backend/macros/types.hpp"
#include "grex/backend/x86/instruction-sets.hpp"
#include "grex/backend/x86/macros/for-each.hpp"
#include "grex/backend/x86/macros/intrinsics.hpp"
#include "grex/backend/x86/types.hpp"
#include "grex/base.hpp"

#if GREX_X86_64_LEVEL >= 3
#include "grex/backend/x86/operations/mask-index.hpp"
#endif
#if GREX_X86_64_LEVEL == 3
#include "grex/backend/x86/operations/split.hpp"
#endif

namespace grex::backend {
//==================================================================================================
// Full storing
//==================================================================================================

// Floating-point store intrinsics already take the correct pointer type, integer ones do not.
#define GREX_STORE_CAST_f(REGISTERBITS) dst
#define GREX_STORE_CAST_i(REGISTERBITS) reinterpret_cast<__m##REGISTERBITS##i*>(dst)
#define GREX_STORE_CAST_u(REGISTERBITS) reinterpret_cast<__m##REGISTERBITS##i*>(dst)

// A single native store function, unaligned for `INFIX=storeu` and aligned for `INFIX=store`.
#define GREX_STORE_BASE(NAME, INFIX, KIND, BITS, SIZE, BITPREFIX, REGISTERBITS, RKIND) \
  inline void NAME(KIND##BITS* dst, NativeVector<KIND##BITS, SIZE> src) { \
    GREX_CAT(BITPREFIX##_##INFIX##_, GREX_SI_SUFFIX(RKIND, BITS, REGISTERBITS)) \
    (GREX_STORE_CAST_##RKIND(REGISTERBITS), src.r); \
  }
#define GREX_STORE_I(...) \
  GREX_STORE_BASE(store, storeu, __VA_ARGS__) \
  GREX_STORE_BASE(store_aligned, store, __VA_ARGS__)
#define GREX_STORE(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  GREX_STORE_I(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS, GREX_REGKIND(KIND, BITS))

#define GREX_STORE_ALL(REGISTERBITS, BITPREFIX) \
  GREX_FOREACH_TYPE_EXT(GREX_STORE, REGISTERBITS, BITPREFIX, REGISTERBITS)
GREX_FOREACH_X86_64_LEVEL(GREX_STORE_ALL)

// Sub-native vectors store exactly the bytes they hold, using the matching narrow store, which
// makes it evident to the compiler that no memory beyond them is ever touched.
#define GREX_STORE_SUB_BASE(NAME, KIND, BITS, PART, RKIND) \
  inline void NAME(KIND##BITS* dst, SubVector<KIND##BITS, PART> src) { \
    GREX_CAT(_mm_storeu_si, GREX_MULTIPLY(BITS, PART)) \
    (dst, GREX_KINDCAST(RKIND, i, BITS, 128, src.full.r)); \
  }
#define GREX_STORE_SUB_I(...) \
  GREX_STORE_SUB_BASE(store, __VA_ARGS__) \
  GREX_STORE_SUB_BASE(store_aligned, __VA_ARGS__)
#define GREX_STORE_SUB(KIND, BITS, PART, SIZE) \
  GREX_STORE_SUB_I(KIND, BITS, PART, GREX_REGKIND(KIND, BITS))
GREX_FOREACH_SUB_EXT(GREX_STORE_SUB)

//==================================================================================================
// Partial storing
//==================================================================================================
//
// Storing the first `size` elements without ever touching memory beyond them:
//   * AVX-512: `mask_storeu` intrinsics at every register width.
//   * AVX2, 32/64-bit elements: `maskstore` intrinsics.
//   * 256-bit registers otherwise: a full store of the lower half and a partial store of the upper.
//   * 128-bit registers otherwise: `partstore::store_prefix`, which works purely in bytes.

#if GREX_X86_64_LEVEL >= 4
#define GREX_PARTSTORE_IMPL(KIND, BITS, SIZE, REGISTERBITS, BITPREFIX, RKIND) \
  GREX_CAT(BITPREFIX##_mask_storeu_, GREX_EPI_SUFFIX(RKIND, BITS)) \
  (dst, cutoff_mask(size, type_tag<NativeMask<KIND##BITS, SIZE>>).r, src.r);
#else
//--------------------------------------------------------------------------------------------------
// Byte-wise partial stores out of a 128-bit register
//--------------------------------------------------------------------------------------------------

namespace partstore {
/** Stores the low `tBytes` bytes of `v` at `ptr`. */
template<std::size_t tBytes>
requires(tBytes == 1 || tBytes == 2 || tBytes == 4 || tBytes == 8 || tBytes == 16)
GREX_ALWAYS_INLINE inline void store_bytes(u8* ptr, __m128i v) {
  if constexpr (tBytes == 16) {
    _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), v);
  } else if constexpr (tBytes == 8) {
    _mm_storeu_si64(ptr, v);
  } else if constexpr (tBytes == 4) {
    _mm_storeu_si32(ptr, v);
  } else if constexpr (tBytes == 2) {
    _mm_storeu_si16(ptr, v);
  } else {
    // There is no `_mm_storeu_si8`, so this is the one place a value passes through a register.
    ptr[0] = u8(_mm_cvtsi128_si32(v));
  }
}

/**
 * Scatters the `bytes ∈ [tBlock, 2·tBlock)` low bytes of `v`, made up of `tElementBytes`-byte
 * elements, to `ptr`: the two overlapping stores `ptr[0, tBlock)` and `ptr[bytes - tBlock, bytes)`
 * cover exactly `[0, bytes)`, so the latter merely needs `v` shifted down by the `bytes - tBlock`
 * bytes the former already provides. Since `bytes` is a multiple of `tElementBytes`, the only such
 * `bytes` for `tElementBytes == tBlock` is `tBlock` itself, which the first store covers alone.
 */
template<std::size_t tBlock, std::size_t tElementBytes>
requires((tBlock == 2 || tBlock == 4) && tElementBytes <= tBlock)
GREX_ALWAYS_INLINE inline void scatter_blocks(u8* ptr, __m128i v, std::size_t bytes) {
  store_bytes<tBlock>(ptr, v);
  if constexpr (tElementBytes < tBlock) {
    store_bytes<tBlock>(ptr + bytes - tBlock,
                        _mm_srl_epi64(v, _mm_cvtsi32_si128(int(8 * (bytes - tBlock)))));
  }
}

/**
 * Stores the low `bytes < tTotal` bytes of `v`, made up of `tElementBytes`-byte elements, at `ptr`,
 * writing no memory beyond them. Everything below eight bytes lives in the low half of `v`, where
 * `psrlq` provides the variable shift, so the bytes above are dealt with by storing the first eight
 * of them and moving the upper half down.
 */
template<std::size_t tElementBytes, std::size_t tTotal>
requires(tElementBytes <= tTotal && tTotal <= 16)
GREX_ALWAYS_INLINE inline void store_prefix_bytes(u8* ptr, __m128i v, std::size_t bytes) {
  if constexpr (tTotal > 8) {
    if (bytes >= 8) {
      store_bytes<8>(ptr, v);
      store_prefix_bytes<tElementBytes, tTotal - 8>(ptr + 8, _mm_unpackhi_epi64(v, v), bytes - 8);
      return;
    }
  }
  // Two overlapping stores of the largest block that fits, halved down to a single byte.
  if constexpr (tElementBytes <= 4 && tTotal > 4) {
    if (bytes >= 4) {
      scatter_blocks<4, tElementBytes>(ptr, v, bytes);
      return;
    }
  }
  if constexpr (tElementBytes <= 2 && tTotal > 2) {
    if (bytes >= 2) {
      scatter_blocks<2, tElementBytes>(ptr, v, bytes);
      return;
    }
  }
  if constexpr (tElementBytes == 1 && tTotal > 1) {
    if (bytes == 1) {
      store_bytes<1>(ptr, v);
    }
  }
}

/**
 * Stores the first `min(count, tCount)` elements of `tElementBytes` bytes each held in the low
 * bytes of `v` at `base`, writing no memory beyond them. The element size and count are
 * compile-time constants so that the unreachable cases, which are the majority for all but 8-bit
 * elements, are pruned.
 */
template<std::size_t tElementBytes, std::size_t tCount>
requires((tElementBytes * tCount) <= 16)
GREX_ALWAYS_INLINE inline void store_prefix(void* base, __m128i v, std::size_t count) {
  static constexpr std::size_t total = tElementBytes * tCount;

  auto* ptr = static_cast<u8*>(base);
  if (count >= tCount) [[unlikely]] {
    store_bytes<total>(ptr, v);
    return;
  }
  // Cannot overflow, since `count < tCount` and `total ≤ 16`.
  store_prefix_bytes<tElementBytes, total>(ptr, v, tElementBytes * count);
}
} // namespace partstore

//--------------------------------------------------------------------------------------------------
// Partial stores of native and sub-native vectors
//--------------------------------------------------------------------------------------------------

// Everything in a 128-bit register that no masked store covers: a byte-wise prefix store.
#define GREX_PARTSTORE_PREFIX(KIND, BITS, SIZE, REGISTERBITS, BITPREFIX, RKIND) \
  partstore::store_prefix<sizeof(*dst), SIZE>(dst, GREX_KINDCAST(RKIND, i, BITS, 128, src.r), size);

#if GREX_X86_64_LEVEL == 3
// AVX2 maskstore intrinsics expect `int*`/`long long*`.
#define GREX_MASKSTORE_CAST_32 reinterpret_cast<int*>(dst)
#define GREX_MASKSTORE_CAST_64 reinterpret_cast<long long*>(dst)
#define GREX_PARTSTORE_MASKSTORE(KIND, BITS, SIZE, REGISTERBITS, BITPREFIX, RKIND) \
  BITPREFIX##_maskstore_epi##BITS(GREX_MASKSTORE_CAST_##BITS, \
                                  cutoff_mask(size, type_tag<NativeMask<KIND##BITS, SIZE>>).r, \
                                  GREX_KINDCAST(RKIND, i, BITS, REGISTERBITS, src.r));

// 256-bit registers with 8/16-bit elements: split into two 128-bit halves.
#define GREX_PARTSTORE_SPLIT(KIND, BITS, SIZE, REGISTERBITS, BITPREFIX, RKIND) \
  static constexpr std::size_t half = GREX_DIVIDE(SIZE, 2); \
  if (size >= SIZE) [[unlikely]] { \
    store(dst, src); \
    return; \
  } \
  if (size <= half) { \
    store_part(dst, get_low(src), size); \
    return; \
  } \
  store(dst, get_low(src)); \
  store_part(dst + half, get_high(src), size - half);

#define GREX_PARTSTORE_128_64 GREX_PARTSTORE_MASKSTORE
#define GREX_PARTSTORE_128_32 GREX_PARTSTORE_MASKSTORE
#define GREX_PARTSTORE_128_16 GREX_PARTSTORE_PREFIX
#define GREX_PARTSTORE_128_8 GREX_PARTSTORE_PREFIX
#define GREX_PARTSTORE_256_64 GREX_PARTSTORE_MASKSTORE
#define GREX_PARTSTORE_256_32 GREX_PARTSTORE_MASKSTORE
#define GREX_PARTSTORE_256_16 GREX_PARTSTORE_SPLIT
#define GREX_PARTSTORE_256_8 GREX_PARTSTORE_SPLIT
#define GREX_PARTSTORE_IMPL(KIND, BITS, SIZE, REGISTERBITS, ...) \
  GREX_CAT(GREX_PARTSTORE_, REGISTERBITS, _, BITS)(KIND, BITS, SIZE, REGISTERBITS, __VA_ARGS__)
#else
// Below AVX2, 128-bit registers are the only native ones.
#define GREX_PARTSTORE_IMPL GREX_PARTSTORE_PREFIX
#endif
#endif

#define GREX_PARTSTORE_I(KIND, BITS, SIZE, REGISTERBITS, BITPREFIX, RKIND) \
  inline void store_part(KIND##BITS* dst, NativeVector<KIND##BITS, SIZE> src, std::size_t size) { \
    GREX_PARTSTORE_IMPL(KIND, BITS, SIZE, REGISTERBITS, BITPREFIX, RKIND) \
  }
#define GREX_PARTSTORE(KIND, BITS, SIZE, REGISTERBITS, BITPREFIX) \
  GREX_PARTSTORE_I(KIND, BITS, SIZE, REGISTERBITS, BITPREFIX, GREX_REGKIND(KIND, BITS))

#define GREX_PARTSTORE_ALL(REGISTERBITS, BITPREFIX) \
  GREX_FOREACH_TYPE_EXT(GREX_PARTSTORE, REGISTERBITS, REGISTERBITS, BITPREFIX)
GREX_FOREACH_X86_64_LEVEL(GREX_PARTSTORE_ALL)

#if GREX_X86_64_LEVEL >= 4
// On AVX-512, the native partial store already touches no more memory than requested.
#define GREX_PARTSTORE_SUB_IMPL(KIND, BITS, PART, SIZE, RKIND) store_part(dst, src.full, size);
#else
#define GREX_PARTSTORE_SUB_IMPL(KIND, BITS, PART, SIZE, RKIND) \
  partstore::store_prefix<sizeof(*dst), PART>(dst, GREX_KINDCAST(RKIND, i, BITS, 128, src.full.r), \
                                              size);
#endif

#define GREX_PARTSTORE_SUB_I(KIND, BITS, PART, SIZE, RKIND) \
  inline void store_part(KIND##BITS* dst, SubVector<KIND##BITS, PART> src, std::size_t size) { \
    GREX_PARTSTORE_SUB_IMPL(KIND, BITS, PART, SIZE, RKIND) \
  }
#define GREX_PARTSTORE_SUB(KIND, BITS, PART, SIZE) \
  GREX_PARTSTORE_SUB_I(KIND, BITS, PART, SIZE, GREX_REGKIND(KIND, BITS))
GREX_FOREACH_SUB_EXT(GREX_PARTSTORE_SUB)
} // namespace grex::backend

#include "grex/backend/shared/operations/store.hpp" // IWYU pragma: export

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_STORE_HPP
