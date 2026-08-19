// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_LOAD_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_LOAD_HPP

#include <cstddef>

#include <immintrin.h>

#include "grex/backend/base.hpp"
#include "grex/backend/macros/base.hpp"
#include "grex/backend/macros/for-each.hpp"
#include "grex/backend/macros/math.hpp"
#include "grex/backend/macros/types.hpp"
#include "grex/backend/x86/instruction-sets.hpp" // IWYU pragma: keep
#include "grex/backend/x86/macros/for-each.hpp"
#include "grex/backend/x86/macros/intrinsics.hpp"
#include "grex/backend/x86/types.hpp"
#include "grex/base.hpp"

#if GREX_X86_64_LEVEL < 4
#include "grex/backend/x86/operations/set.hpp"
#endif
#if GREX_X86_64_LEVEL == 3
#include <array>

#include "grex/backend/x86/operations/merge.hpp"
#endif
#if GREX_X86_64_LEVEL >= 3
#include "grex/backend/x86/operations/mask-index.hpp"
#endif

namespace grex::backend {
//==================================================================================================
// Full loading
//==================================================================================================

// Floating-point load intrinsics already take the correct pointer type, integer ones do not.
#define GREX_LOAD_CAST_f(REGISTERBITS) ptr
#define GREX_LOAD_CAST_i(REGISTERBITS) reinterpret_cast<const __m##REGISTERBITS##i*>(ptr)
#define GREX_LOAD_CAST_u(REGISTERBITS) reinterpret_cast<const __m##REGISTERBITS##i*>(ptr)

// A single native load function, unaligned for `INFIX=loadu` and aligned for `INFIX=load`.
#define GREX_LOAD_BASE(NAME, INFIX, KIND, BITS, SIZE, BITPREFIX, REGISTERBITS, RKIND) \
  inline NativeVector<KIND##BITS, SIZE> NAME(const KIND##BITS* ptr, \
                                             TypeTag<NativeVector<KIND##BITS, SIZE>>) { \
    return {.r = GREX_CAT(BITPREFIX##_##INFIX##_, GREX_SI_SUFFIX(RKIND, BITS, REGISTERBITS))( \
              GREX_LOAD_CAST_##RKIND(REGISTERBITS))}; \
  }
#define GREX_LOAD_I(...) \
  GREX_LOAD_BASE(load, loadu, __VA_ARGS__) \
  GREX_LOAD_BASE(load_aligned, load, __VA_ARGS__)
#define GREX_LOAD(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  GREX_LOAD_I(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS, GREX_REGKIND(KIND, BITS))

#define GREX_LOAD_ALL(REGISTERBITS, BITPREFIX) \
  GREX_FOREACH_TYPE_EXT(GREX_LOAD, REGISTERBITS, BITPREFIX, REGISTERBITS)
GREX_FOREACH_X86_64_LEVEL(GREX_LOAD_ALL)

// Sub-native vectors load exactly the bytes they hold, using the matching narrow load.
#define GREX_LOAD_SUB_BASE(NAME, KIND, BITS, PART, RKIND) \
  inline SubVector<KIND##BITS, PART> NAME(const KIND##BITS* ptr, \
                                          TypeTag<SubVector<KIND##BITS, PART>>) { \
    const __m128i r = GREX_CAT(_mm_loadu_si, GREX_MULTIPLY(BITS, PART))(ptr); \
    return SubVector<KIND##BITS, PART>{GREX_KINDCAST(i, RKIND, BITS, 128, r)}; \
  }
#define GREX_LOAD_SUB_I(...) \
  GREX_LOAD_SUB_BASE(load, __VA_ARGS__) \
  GREX_LOAD_SUB_BASE(load_aligned, __VA_ARGS__)
#define GREX_LOAD_SUB(KIND, BITS, PART, SIZE) \
  GREX_LOAD_SUB_I(KIND, BITS, PART, GREX_REGKIND(KIND, BITS))
GREX_FOREACH_SUB_EXT(GREX_LOAD_SUB)

//==================================================================================================
// Partial loading
//==================================================================================================
//
// Loading the first `size` elements without ever touching memory beyond them, leaving the
// remaining lanes unspecified:
//   * AVX-512: `maskz_loadu` intrinsics at every register width.
//   * AVX2, 32/64-bit elements: `maskload` intrinsics.
//   * 256-bit registers otherwise: full and partial loads of the two halves, merged.
//   * 128-bit registers otherwise: `partload::load_prefix`, which works purely in bytes.

#if GREX_X86_64_LEVEL >= 4
#define GREX_PARTLOAD_IMPL(KIND, BITS, SIZE, REGISTERBITS, BITPREFIX, RKIND) \
  return {.r = GREX_CAT(BITPREFIX##_maskz_loadu_, GREX_EPI_SUFFIX(RKIND, BITS))( \
            cutoff_mask(size, type_tag<NativeMask<KIND##BITS, SIZE>>).r, ptr)};
#else

//--------------------------------------------------------------------------------------------------
// Byte-wise partial loads into a 128-bit register
//--------------------------------------------------------------------------------------------------
namespace partload {
#if GREX_X86_64_LEVEL == 3
/**
 * Byte indices whose 16-byte window starting at `16 - n` is the `pshufb` control row that moves the
 * top `n` bytes of a register down to the bottom, zeroing the bytes above them.
 */
inline constexpr std::array<u8, 32> shift_down_idxs = [] {
  std::array<u8, 32> idxs{};
  for (std::size_t i = 0; i < 16; ++i) {
    idxs[i] = u8(i);
    idxs[i + 16] = 0x80;
  }
  return idxs;
}();

/** Moves the top `tail ≤ 16` bytes of `v` down to the bottom, zeroing the bytes above them. */
GREX_ALWAYS_INLINE inline __m128i shift_down(__m128i v, std::size_t tail) {
  const auto* row = shift_down_idxs.data() + (16 - tail);
  return _mm_shuffle_epi8(v, _mm_loadu_si128(reinterpret_cast<const __m128i*>(row)));
}
#endif

/** Loads exactly `tBytes` bytes at `ptr` into the low bytes of a 128-bit register. */
template<std::size_t tBytes>
requires(tBytes == 2 || tBytes == 4 || tBytes == 8 || tBytes == 16)
GREX_ALWAYS_INLINE inline __m128i load_bytes(const u8* ptr) {
  if constexpr (tBytes == 16) {
    return _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr));
  } else if constexpr (tBytes == 8) {
    return _mm_loadu_si64(ptr);
  } else if constexpr (tBytes == 4) {
    return _mm_loadu_si32(ptr);
  } else {
    return _mm_loadu_si16(ptr);
  }
}

/**
 * Gathers the `bytes ∈ [tBlock, 2·tBlock)` bytes at `ptr`, made up of `tElementBytes`-byte
 * elements, into the low bytes of a 128-bit register, zeroing the bytes above them: the two
 * overlapping loads `lo = src[0, tBlock)` and `src[bytes - tBlock, bytes)` cover everything, so the
 * top `bytes - tBlock` bytes of the latter, the only ones `lo` does not provide, are shifted down
 * to the bottom and interleaved above `lo`. Since a shift by 64 bits or more yields zero,
 * `bytes == tBlock` needs no special treatment.
 */
template<std::size_t tBlock, std::size_t tElementBytes>
requires((tBlock == 2 || tBlock == 4 || tBlock == 8) && tElementBytes <= tBlock)
GREX_ALWAYS_INLINE inline __m128i gather_blocks(const u8* ptr, std::size_t bytes) {
  const __m128i lo = load_bytes<tBlock>(ptr);
  if constexpr (tElementBytes == tBlock) {
    // The only multiple of `tBlock` in `[tBlock, 2·tBlock)` is `tBlock`, so `lo` is everything.
    return lo;
  } else {
    const __m128i hi = _mm_srl_epi64(load_bytes<tBlock>(ptr + bytes - tBlock),
                                     _mm_cvtsi32_si128(int(8 * (2 * tBlock - bytes))));
    if constexpr (tBlock == 8) {
      return _mm_unpacklo_epi64(lo, hi);
    } else if constexpr (tBlock == 4) {
      return _mm_unpacklo_epi32(lo, hi);
    } else {
      return _mm_unpacklo_epi16(lo, hi);
    }
  }
}

/**
 * Loads the first `min(count, tCount)` elements of `tElementBytes` bytes each at `base` into the
 * low bytes of a 128-bit register, zeroing the bytes above them and reading no memory beyond them.
 * The element size and count are compile-time constants so that the unreachable cases, which are
 * the majority for all but 8-bit elements, are pruned.
 */
template<std::size_t tElementBytes, std::size_t tCount>
requires((tElementBytes * tCount) <= 16)
GREX_ALWAYS_INLINE inline __m128i load_prefix(const void* base, std::size_t count) {
  static constexpr std::size_t total = tElementBytes * tCount;

  const auto* ptr = static_cast<const u8*>(base);
  if (count >= tCount) [[unlikely]] {
    return load_bytes<total>(ptr);
  }
  // Cannot overflow, since `count < tCount` and `total ≤ 16`.
  const std::size_t bytes = tElementBytes * count;

  // Two overlapping loads of the largest block that fits, halved down to a single byte.
  if constexpr (total > 8) {
    if (bytes >= 8) {
      return gather_blocks<8, tElementBytes>(ptr, bytes);
    }
  }
  if constexpr (tElementBytes <= 4 && total > 4) {
    if (bytes >= 4) {
      return gather_blocks<4, tElementBytes>(ptr, bytes);
    }
  }
  if constexpr (tElementBytes <= 2 && total > 2) {
    if (bytes >= 2) {
      return gather_blocks<2, tElementBytes>(ptr, bytes);
    }
  }
  if constexpr (tElementBytes == 1) {
    if (bytes == 1) {
      return _mm_cvtsi32_si128(ptr[0]);
    }
  }
  return _mm_setzero_si128();
}
} // namespace partload

//--------------------------------------------------------------------------------------------------
// Partial loads of native and sub-native vectors
//--------------------------------------------------------------------------------------------------

// Narrow loads of a single element: there is no `_mm_loadu_si8`.
#define GREX_PARTLOAD_ONE_8 _mm_cvtsi32_si128(u8(*ptr))
#define GREX_PARTLOAD_ONE_16 _mm_loadu_si16(ptr)
#define GREX_PARTLOAD_ONE_32 _mm_loadu_si32(ptr)
#define GREX_PARTLOAD_ONE_64 _mm_loadu_si64(ptr)

// Two-element vectors: every case is a single narrow load, which beats the byte-wise path.
#define GREX_PARTLOAD_TWO(VECTOR, KIND, BITS, PART, RKIND) \
  switch (size) { \
    [[unlikely]] case 0: \
      return zeros(type_tag<VECTOR<KIND##BITS, PART>>); \
    [[likely]] case 1: \
      return VECTOR<KIND##BITS, PART>{ \
        GREX_KINDCAST(i, RKIND, BITS, 128, GREX_PARTLOAD_ONE_##BITS)}; \
    [[unlikely]] default: \
      return VECTOR<KIND##BITS, PART>{ \
        GREX_KINDCAST(i, RKIND, BITS, 128, \
                      partload::load_bytes<2 * sizeof(*ptr)>(reinterpret_cast<const u8*>(ptr)))}; \
  }

// Everything else in a 128-bit register: a byte-wise prefix load.
// The extra parentheses hide the comma in the template argument list from the preprocessor.
#define GREX_PARTLOAD_PREFIX(VECTOR, KIND, BITS, PART, RKIND) \
  return VECTOR<KIND##BITS, PART>{ \
    GREX_KINDCAST(i, RKIND, BITS, 128, (partload::load_prefix<sizeof(*ptr), PART>(ptr, size)))};

#if GREX_X86_64_LEVEL >= 3
// AVX/AVX2 maskload intrinsics expect `int*`/`long long*`.
#define GREX_MASKLOAD_CAST_32 reinterpret_cast<const int*>(ptr)
#define GREX_MASKLOAD_CAST_64 reinterpret_cast<const long long*>(ptr)
#define GREX_PARTLOAD_MASKLOAD(KIND, BITS, SIZE, REGISTERBITS, BITPREFIX, RKIND) \
  return {.r = GREX_KINDCAST(i, RKIND, BITS, REGISTERBITS, \
                             BITPREFIX##_maskload_epi##BITS( \
                               GREX_MASKLOAD_CAST_##BITS, \
                               cutoff_mask(size, type_tag<NativeMask<KIND##BITS, SIZE>>).r))};

// 256-bit registers with 8/16-bit elements: combine two 128-bit halves.
#define GREX_PARTLOAD_SPLIT(KIND, BITS, SIZE, REGISTERBITS, BITPREFIX, RKIND) \
  static constexpr std::size_t half = GREX_DIVIDE(SIZE, 2); \
  using Half = NativeVector<KIND##BITS, half>; \
  if (size <= half) { \
    return merge(load_part(ptr, size, type_tag<Half>), undefined(type_tag<Half>)); \
  } \
  if (size >= SIZE) [[unlikely]] { \
    return load(ptr, type_tag<NativeVector<KIND##BITS, SIZE>>); \
  } \
  /* The requested elements end at `ptr[size]`, so the whole 16-byte load below stays in bounds, \
   * and the upper half only has to be moved down by the bytes the lower half already covers. */ \
  const auto lo = load(ptr, type_tag<Half>); \
  const __m128i hi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr + (size - half))); \
  return merge(lo, Half{GREX_KINDCAST(i, RKIND, BITS, 128, \
                                      partload::shift_down(hi, sizeof(*ptr) * size - 16))});

#define GREX_PARTLOAD_128_64 GREX_PARTLOAD_MASKLOAD
#define GREX_PARTLOAD_128_32 GREX_PARTLOAD_MASKLOAD
#define GREX_PARTLOAD_256_64 GREX_PARTLOAD_MASKLOAD
#define GREX_PARTLOAD_256_32 GREX_PARTLOAD_MASKLOAD
#define GREX_PARTLOAD_256_16 GREX_PARTLOAD_SPLIT
#define GREX_PARTLOAD_256_8 GREX_PARTLOAD_SPLIT
#define GREX_PARTLOAD_IMPL(KIND, BITS, SIZE, REGISTERBITS, ...) \
  GREX_CAT(GREX_PARTLOAD_, REGISTERBITS, _, BITS)(KIND, BITS, SIZE, REGISTERBITS, __VA_ARGS__)
#else
#define GREX_PARTLOAD_128_64(KIND, BITS, SIZE, REGISTERBITS, BITPREFIX, RKIND) \
  GREX_PARTLOAD_TWO(NativeVector, KIND, BITS, SIZE, RKIND)
#define GREX_PARTLOAD_128_32 GREX_PARTLOAD_128_PREFIX
#define GREX_PARTLOAD_IMPL(KIND, BITS, SIZE, REGISTERBITS, ...) \
  GREX_CAT(GREX_PARTLOAD_128_, BITS)(KIND, BITS, SIZE, REGISTERBITS, __VA_ARGS__)
#endif

#define GREX_PARTLOAD_128_PREFIX(KIND, BITS, SIZE, REGISTERBITS, BITPREFIX, RKIND) \
  GREX_PARTLOAD_PREFIX(NativeVector, KIND, BITS, SIZE, RKIND)
#define GREX_PARTLOAD_128_16 GREX_PARTLOAD_128_PREFIX
#define GREX_PARTLOAD_128_8 GREX_PARTLOAD_128_PREFIX
#endif

#define GREX_PARTLOAD_I(KIND, BITS, SIZE, REGISTERBITS, BITPREFIX, RKIND) \
  inline NativeVector<KIND##BITS, SIZE> load_part(const KIND##BITS* ptr, std::size_t size, \
                                                  TypeTag<NativeVector<KIND##BITS, SIZE>>) { \
    GREX_PARTLOAD_IMPL(KIND, BITS, SIZE, REGISTERBITS, BITPREFIX, RKIND) \
  }
#define GREX_PARTLOAD(KIND, BITS, SIZE, REGISTERBITS, BITPREFIX) \
  GREX_PARTLOAD_I(KIND, BITS, SIZE, REGISTERBITS, BITPREFIX, GREX_REGKIND(KIND, BITS))

#define GREX_PARTLOAD_ALL(REGISTERBITS, BITPREFIX) \
  GREX_FOREACH_TYPE_EXT(GREX_PARTLOAD, REGISTERBITS, REGISTERBITS, BITPREFIX)
GREX_FOREACH_X86_64_LEVEL(GREX_PARTLOAD_ALL)

#if GREX_X86_64_LEVEL >= 4
// On AVX-512, the native partial load already touches no more memory than requested.
#define GREX_PARTLOAD_SUB_IMPL(KIND, BITS, PART, SIZE, RKIND) \
  return SubVector<KIND##BITS, PART>{ \
    load_part(ptr, size, type_tag<NativeVector<KIND##BITS, SIZE>>)};
#else
#define GREX_PARTLOAD_SUB_2(KIND, BITS, PART, RKIND) \
  GREX_PARTLOAD_TWO(SubVector, KIND, BITS, PART, RKIND)
#define GREX_PARTLOAD_SUB_PREFIX(KIND, BITS, PART, RKIND) \
  GREX_PARTLOAD_PREFIX(SubVector, KIND, BITS, PART, RKIND)
#define GREX_PARTLOAD_SUB_4 GREX_PARTLOAD_SUB_PREFIX
#define GREX_PARTLOAD_SUB_8 GREX_PARTLOAD_SUB_PREFIX
#define GREX_PARTLOAD_SUB_IMPL(KIND, BITS, PART, SIZE, RKIND) \
  GREX_CAT(GREX_PARTLOAD_SUB_, PART)(KIND, BITS, PART, RKIND)
#endif

#define GREX_PARTLOAD_SUB_I(KIND, BITS, PART, SIZE, RKIND) \
  inline SubVector<KIND##BITS, PART> load_part(const KIND##BITS* ptr, std::size_t size, \
                                               TypeTag<SubVector<KIND##BITS, PART>>) { \
    GREX_PARTLOAD_SUB_IMPL(KIND, BITS, PART, SIZE, RKIND) \
  }
#define GREX_PARTLOAD_SUB(KIND, BITS, PART, SIZE) \
  GREX_PARTLOAD_SUB_I(KIND, BITS, PART, SIZE, GREX_REGKIND(KIND, BITS))
GREX_FOREACH_SUB_EXT(GREX_PARTLOAD_SUB)
} // namespace grex::backend

#include "grex/backend/shared/operations/load.hpp" // IWYU pragma: export

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_LOAD_HPP
