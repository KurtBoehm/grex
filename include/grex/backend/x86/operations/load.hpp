// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_LOAD_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_LOAD_HPP

#include <array>
#include <cstddef>
#include <cstring>

#include <immintrin.h>

#include "grex/backend/base.hpp"
#include "grex/backend/macros/base.hpp"
#include "grex/backend/macros/for-each.hpp"
#include "grex/backend/macros/math.hpp"
#include "grex/backend/x86/instruction-sets.hpp" // IWYU pragma: keep
#include "grex/backend/x86/macros/for-each.hpp"
#include "grex/backend/x86/macros/intrinsics.hpp"
#include "grex/backend/x86/operations/reinterpret.hpp"
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
#if GREX_X86_64_LEVEL == 1
#include <bit>
#endif

namespace grex::backend {
// Helper casts for intrinsic loads.
// Floating-point load intrinsics already take the correct pointer type,
// while integer loads require a cast to __m{128,256,512}i*.
#define GREX_LOAD_CAST_f(REGISTERBITS) ptr
#define GREX_LOAD_CAST_i(REGISTERBITS) reinterpret_cast<const __m##REGISTERBITS##i*>(ptr)
#define GREX_LOAD_CAST_u(REGISTERBITS) reinterpret_cast<const __m##REGISTERBITS##i*>(ptr)

// Generate a single load function (unaligned or aligned) for a given
//   KIND ∈ {f, i, u},
//   BITS = element width in bits,
//   SIZE = number of elements in the vector,
//   REGISTERBITS = SIMD register width in bits.
#define GREX_LOAD_BASE(NAME, INFIX, KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  inline NativeVector<KIND##BITS, SIZE> NAME(const KIND##BITS* ptr, \
                                             TypeTag<NativeVector<KIND##BITS, SIZE>>) { \
    return {.r = GREX_CAT(BITPREFIX##_##INFIX##_, GREX_SI_SUFFIX(KIND, BITS, REGISTERBITS))( \
              GREX_LOAD_CAST_##KIND(REGISTERBITS))}; \
  }

// Define both unaligned (load) and aligned (load_aligned) variants.
#define GREX_LOAD(...) \
  GREX_LOAD_BASE(load, loadu, __VA_ARGS__) \
  GREX_LOAD_BASE(load_aligned, load, __VA_ARGS__)

// Helpers for building shuffle tables used by SSE/SSSE3-based partial loads.
// The tables contain pshufb control masks that assemble prefix slices of
// a scalar array into the low bytes of a 128-bit register.
namespace shld {
// One pshufb control row: 16 indices/0x80 (zero) selectors.
using ShuffleRow = std::array<u8, 16>;

// For a fixed block size B, we support all result lengths n where B ≤ n ≤ min(2*B, 16).
// Table index is (n - B).
template<std::size_t tBlockBytes>
using ShuffleTable = std::array<ShuffleRow, 17 - tBlockBytes>;

// Generic compile-time shuffle-table generator for a given block size B.
//
// Conceptually we form AB = [src[0..B-1], src[n-B..n-1]] (2*B bytes),
// stored in the low bytes of a __m128i. A subsequent
//
//   pshufb(AB, mask[n])
//
// yields:
//   result[0 .. n-1] = src[0 .. n-1]
//   result[n .. 15]  = 0
//
// Valid n for a given B:
//
//   B ≤ n ≤ min(2*B, 16).
template<std::size_t tBlockBytes>
consteval ShuffleTable<tBlockBytes> make_shuffle_table_block() {
  static_assert(tBlockBytes == 8 || tBlockBytes == 4 || tBlockBytes == 2);

  ShuffleTable<tBlockBytes> table{};

  for (std::size_t n = tBlockBytes; n <= 16; ++n) {
    auto& row = table[n - tBlockBytes];
    row.fill(0x80); // default: zero all bytes

    if (n > 2 * tBlockBytes) {
      continue;
    }

    for (std::size_t i = 0; i < 16; ++i) {
      u8 idx = 0x80; // default: zero this output byte

      if (i < tBlockBytes) {
        // First block: src[0 .. B-1] -> AB[0 .. B-1]
        idx = u8(i);
      } else if (i < n) {
        // Second block: src[n-B .. n-1] -> AB[B .. 2B-1]
        // For i in [B, n-1], map to AB index:
        //   idx = i + 2B - n
        idx = u8(i + 2 * tBlockBytes - n);
      }

      row[i] = idx;
    }
  }

  return table;
}

consteval std::array<ShuffleRow, 17> make_shuffle_table_block_256hi() {
  std::array<ShuffleRow, 17> table{};

  for (std::size_t n = 0; n <= 16; ++n) {
    auto& row = table[n];
    row.fill(0x80); // default: zero all bytes
    for (std::size_t i = 0; i < n; ++i) {
      row[i] = u8(i + 16 - n); // shuffle low bytes
    }
  }

  return table;
}

// Precomputed shuffle tables, block sizes 8/4/2 bytes.
alignas(16) inline constexpr std::array<ShuffleRow, 17> idxs16 = make_shuffle_table_block_256hi();
alignas(16) inline constexpr ShuffleTable<8> idxs8 = make_shuffle_table_block<8>();
alignas(16) inline constexpr ShuffleTable<4> idxs4 = make_shuffle_table_block<4>();
alignas(16) inline constexpr ShuffleTable<2> idxs2 = make_shuffle_table_block<2>();
} // namespace shld

/////////////////////
// Partial loading //
/////////////////////
//
// Goal: load only the first `size` elements (0 ≤ size ≤ vector length) from memory into a SIMD
// register while never touching out-of-bounds memory.
//
// Strategy by ISA level:
//   * AVX-512: use native maskz_loadu intrinsics.
//   * 128-bit registers:
//       - Level 3 (AVX2/AVX512F), 32/64-bit elements: use maskload.
//       - Lower levels: use overlapping loads + pshufb (SSSE3) or piecewise scalar memcpy into
//         an __m128i.
//   * 256-bit registers:
//       - 32/64-bit elements: use maskload (AVX2).
//       - Smaller element sizes: implement via two 128-bit partial loads.
//   * 512-bit registers (no AVX-512 load): recursively split into halves.

// AVX/AVX2 maskload intrinsics expect int*/long long*.
#define GREX_MASKLOAD_CAST_32 reinterpret_cast<const int*>(ptr)
#define GREX_MASKLOAD_CAST_64 reinterpret_cast<const long long*>(ptr)

// Common AVX/AVX2 maskload-based partial load implementation.
#define GREX_PARTLOAD_MASKLOAD(KIND, BITS, SIZE, REGISTERBITS, BITPREFIX) \
  return {.r = GREX_KINDCAST(i, KIND, BITS, REGISTERBITS, \
                             BITPREFIX##_maskload_epi##BITS( \
                               GREX_MASKLOAD_CAST_##BITS, \
                               cutoff_mask(size, type_tag<NativeMask<KIND##BITS, SIZE>>).r))};

// Fast-path and degenerate cases shared by several scalar/SSSE3 fallbacks.
// - If size >= full width, fall back to a normal load.
// - If size == 0, return an all-zero vector.
// Dst is the concrete vector type being loaded.
#define GREX_PARTLOAD_FALLBACK_INIT(KIND, BITS, SIZE) \
  using Dst = VectorFor<KIND##BITS, SIZE>; \
  if (size >= SIZE) [[unlikely]] { \
    return load(ptr, type_tag<Dst>); \
  } \
  if (size == 0) [[unlikely]] { \
    return zeros(type_tag<Dst>); \
  }

// Small helper for 2- or 4-element sub-vectors: handle size==0/1/else
// without repeating boilerplate. `size` is in elements, not bytes.
#define GREX_PARTLOAD_SWITCH(KIND, BITS, SIZE, CASE1_EXPR, DEFAULT_EXPR) \
  switch (size) { \
    [[unlikely]] case 0: \
      return zeros(type_tag<VectorFor<KIND##BITS, SIZE>>); \
    [[likely]] case 1: \
      return CASE1_EXPR; \
    [[unlikely]] default: \
      return DEFAULT_EXPR; \
  }

// 64-bit entries: maskload on level 3+, otherwise a simple 2-case switch
// using scalar-sized loads (_mm_loadu_si64 or full __m128i load).
#if GREX_X86_64_LEVEL >= 3
#define GREX_PARTLOAD_128_64 GREX_PARTLOAD_MASKLOAD
#define GREX_PARTLOAD_256_64 GREX_PARTLOAD_MASKLOAD
#else
#define GREX_PARTLOAD_128_64(KIND, ...) \
  GREX_PARTLOAD_SWITCH( \
    KIND, 64, 2, {.r = GREX_KINDCAST(i, KIND, 64, 128, _mm_loadu_si64(ptr))}, \
    {.r = \
       GREX_KINDCAST(i, KIND, 64, 128, _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr)))})
#endif

// 32-bit entries:
// * Level 3+: maskload.
// * Level 2: overlapping loads + pshufb using precomputed shuffle tables.
// * Lower levels: piecewise memcpy into an __m128i.
#if GREX_X86_64_LEVEL >= 3
#define GREX_PARTLOAD_128_32 GREX_PARTLOAD_MASKLOAD
#define GREX_PARTLOAD_256_32 GREX_PARTLOAD_MASKLOAD
#elif GREX_X86_64_LEVEL >= 2
#define GREX_PARTLOAD_128_32(KIND, ...) \
  GREX_PARTLOAD_FALLBACK_INIT(KIND, 32, 4) \
  /* 8-byte block path: size ∈ [2,3] */ \
  if (size >= 2) { \
    __m128i lo = _mm_loadu_si64(ptr); \
    __m128i hi = _mm_loadu_si64(ptr + (size - 2)); \
    /* AB = [ptr[0..1], ptr[size-2..size-1]] in 2×64-bit lanes */ \
    __m128i ab = _mm_unpacklo_epi64(lo, hi); \
\
    __m128i mask = load(shld::idxs8[4 * size - 8].data(), type_tag<u8x16>).r; \
    return {.r = GREX_KINDCAST(i, KIND, 32, 128, _mm_shuffle_epi8(ab, mask))}; \
  } \
\
  /* size == 1 */ \
  return {.r = GREX_KINDCAST(i, KIND, 32, 128, _mm_loadu_si32(ptr))};
#else
#define GREX_PARTLOAD_128_32(KIND, ...) \
  GREX_PARTLOAD_FALLBACK_INIT(KIND, 32, 4) \
  /* Accumulate up to 3 dwords into a 64-bit temporary using memcpy. */ \
  u64 out = 0; \
  if ((size & 1U) != 0) { \
    std::memcpy(&out, ptr + (size / 2 * 2), 4); \
  } \
  if ((size & 2U) != 0) { \
    u64 lo; \
    std::memcpy(&lo, ptr, 8); \
    const auto merged = _mm_set_epi64x(std::bit_cast<i64>(out), std::bit_cast<i64>(lo)); \
    return {.r = GREX_KINDCAST(i, KIND, 32, 128, merged)}; \
  } \
  const auto merged = _mm_set_epi64x(0, std::bit_cast<i64>(out)); \
  return {.r = GREX_KINDCAST(i, KIND, 32, 128, merged)};
#endif

// 16-bit entries:
// * Level 2+: overlapping 64-bit and 32-bit loads + pshufb.
// * Lower levels: scalar memcpy into a pair of 64-bit lanes.
#if GREX_X86_64_LEVEL >= 2
#define GREX_PARTLOAD_128_16(KIND, ...) \
  GREX_PARTLOAD_FALLBACK_INIT(KIND, 16, 8) \
\
  /* 8-byte block path: size ∈ [4,8] */ \
  if (size >= 4) { \
    __m128i lo = _mm_loadu_si64(ptr); \
    __m128i hi = _mm_loadu_si64(ptr + (size - 4)); \
    /* AB = [ptr[0..3], ptr[size-4..size-1]] in 2×64-bit lanes */ \
    __m128i ab = _mm_unpacklo_epi64(lo, hi); \
\
    __m128i mask = load(shld::idxs8[2 * size - 8].data(), type_tag<u8x16>).r; \
    return {.r = _mm_shuffle_epi8(ab, mask)}; \
  } \
\
  /* 4-byte block path: size ∈ [2,3] */ \
  if (size >= 2) { \
    __m128i lo = _mm_loadu_si32(ptr); \
    __m128i hi = _mm_loadu_si32(ptr + (size - 2)); \
    /* AB = [ptr[0..1], ptr[size-2..size-1]] in bytes [0..3] */ \
    __m128i ab = _mm_unpacklo_epi32(lo, hi); \
\
    __m128i mask = load(shld::idxs4[2 * size - 4].data(), type_tag<u8x16>).r; \
\
    return {.r = _mm_shuffle_epi8(ab, mask)}; \
  } \
\
  /* size == 1 */ \
  return {.r = _mm_loadu_si16(ptr)};
#else
#define GREX_PARTLOAD_128_16(KIND, ...) \
  GREX_PARTLOAD_FALLBACK_INIT(KIND, 16, 8) \
  /* Accumulate up to 7 words into a 64-bit temporary. */ \
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
#endif

// 8-bit entries:
// * Level 2+: overlapping 64/32/16-bit loads + pshufb for all sizes.
// * Lower levels: scalar memcpy accumulation into 64-bit lanes.
#if GREX_X86_64_LEVEL >= 2
#define GREX_PARTLOAD_128_8(KIND, ...) \
  GREX_PARTLOAD_FALLBACK_INIT(KIND, 8, 16) \
\
  /* 8-byte block path: size ∈ [8,16] */ \
  if (size >= 8) { \
    const __m128i lo = _mm_loadu_si64(ptr); \
    const __m128i hi = _mm_loadu_si64(ptr + (size - 8)); \
    /* AB = [ptr[0..7], ptr[size-8..size-1]] */ \
    const __m128i ab = _mm_unpacklo_epi64(lo, hi); \
\
    const __m128i mask = load(shld::idxs8[size - 8].data(), type_tag<u8x16>).r; \
    return {.r = _mm_shuffle_epi8(ab, mask)}; \
  } \
\
  /* 4-byte block path: size ∈ [4,7] */ \
  if (size >= 4) { \
    const __m128i lo = _mm_loadu_si32(ptr); \
    const __m128i hi = _mm_loadu_si32(ptr + (size - 4)); \
    /* AB = [ptr[0..3], ptr[size-4..size-1]] in bytes [0..7] */ \
    const __m128i ab = _mm_unpacklo_epi32(lo, hi); \
\
    const __m128i mask = load(shld::idxs4[size - 4].data(), type_tag<u8x16>).r; \
    return {.r = _mm_shuffle_epi8(ab, mask)}; \
  } \
\
  /* 2-byte block path: size ∈ [2,3] */ \
  if (size >= 2) { \
    const __m128i lo = _mm_loadu_si16(ptr); \
    const __m128i hi = _mm_loadu_si16(ptr + (size - 2)); \
    /* AB = [ptr[0..1], ptr[size-2..size-1]] in bytes [0..3] */ \
    const __m128i ab = _mm_unpacklo_epi16(lo, hi); \
\
    const __m128i mask = load(shld::idxs2[size - 2].data(), type_tag<u8x16>).r; \
    return {.r = _mm_shuffle_epi8(ab, mask)}; \
  } \
\
  /* size == 1 */ \
  return {.r = _mm_cvtsi32_si128(u8(*ptr))};
#else
#define GREX_PARTLOAD_128_8(KIND, ...) \
  GREX_PARTLOAD_FALLBACK_INIT(KIND, 8, 16) \
  /* Accumulate up to 15 bytes into a 64-bit temporary. */ \
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
#endif

// 256/512-bit vectors without native partial-load support:
// recursively split into two half-width vectors, partial-load the tail,
// and merge with a full/zeroed head.
#define GREX_PARTLOAD_SPLIT(KIND, BITS, SIZE, ...) \
  using Value = KIND##BITS; \
  using Half = VectorFor<Value, GREX_DIVIDE(SIZE, 2)>; \
\
  if (size <= GREX_DIVIDE(SIZE, 2)) { \
    return merge(load_part(ptr, size, type_tag<Half>), zeros(type_tag<Half>)); \
  } \
\
  if (size >= SIZE) [[unlikely]] { \
    return load(ptr, type_tag<VectorFor<Value, SIZE>>); \
  } \
\
  /* 16-byte block path: len ∈ [SIZE/2, SIZE] */ \
  const auto lo = load(ptr, type_tag<Half>); \
  const auto hi = as<u8>(load(ptr + (size - GREX_DIVIDE(SIZE, 2)), type_tag<Half>)).r; \
\
  __m128i mask = load(shld::idxs16[sizeof(Value) * size - 16].data(), type_tag<u8x16>).r; \
  return merge(lo, as<Value>(u8x16{_mm_shuffle_epi8(hi, mask)}));

#define GREX_PARTLOAD_256_16 GREX_PARTLOAD_SPLIT
#define GREX_PARTLOAD_256_8 GREX_PARTLOAD_SPLIT

// AVX-512 maskz_loadu-based partial loads for any register width.
#define GREX_PARTLOAD_AVX512(KIND, BITS, SIZE, REGISTERBITS, BITPREFIX) \
  return {.r = GREX_CAT(BITPREFIX##_maskz_loadu_, GREX_EPI_SUFFIX(KIND, BITS))( \
            cutoff_mask(size, type_tag<NativeMask<KIND##BITS, SIZE>>).r, ptr)};

#if GREX_X86_64_LEVEL >= 4
#define GREX_PARTLOAD_128 GREX_PARTLOAD_AVX512
#define GREX_PARTLOAD_256 GREX_PARTLOAD_AVX512
#define GREX_PARTLOAD_512 GREX_PARTLOAD_AVX512
#elif GREX_X86_64_LEVEL == 3
#define GREX_PARTLOAD_128(KIND, BITS, ...) GREX_PARTLOAD_128_##BITS(KIND, BITS, __VA_ARGS__)
#define GREX_PARTLOAD_256(KIND, BITS, ...) GREX_PARTLOAD_256_##BITS(KIND, BITS, __VA_ARGS__)
#else
#define GREX_PARTLOAD_128(KIND, BITS, ...) GREX_PARTLOAD_128_##BITS(KIND)
#endif

// Public partial-load entry point for a full native vector.
#define GREX_PARTLOAD(KIND, BITS, SIZE, REGISTERBITS, BITPREFIX) \
  inline NativeVector<KIND##BITS, SIZE> load_part(const KIND##BITS* ptr, std::size_t size, \
                                                  TypeTag<NativeVector<KIND##BITS, SIZE>>) { \
    GREX_PARTLOAD_##REGISTERBITS(KIND, BITS, SIZE, REGISTERBITS, BITPREFIX) \
  }

// Instantiate all load()/load_aligned() overloads for every vector type
// and register width supported at the current x86-64 level.
#define GREX_LOAD_ALL(REGISTERBITS, BITPREFIX) \
  GREX_FOREACH_TYPE(GREX_LOAD, REGISTERBITS, BITPREFIX, REGISTERBITS)
GREX_FOREACH_X86_64_LEVEL(GREX_LOAD_ALL)

// Instantiate all load_part() overloads for every vector type and register
// width supported at the current x86-64 level.
#define GREX_PARTLOAD_ALL(REGISTERBITS, BITPREFIX) \
  GREX_FOREACH_TYPE(GREX_PARTLOAD, REGISTERBITS, REGISTERBITS, BITPREFIX)
GREX_FOREACH_X86_64_LEVEL(GREX_PARTLOAD_ALL)

////////////////////////
// Sub-native vectors //
////////////////////////

// Basic sub-vector loads: always load exactly PART elements worth of bytes
// via the corresponding scalar-sized _mm_loadu_siN intrinsic.
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
// Sub-vector partial loads on pre-AVX-512: specialized small-code paths.

// 2×32-bit sub-vector: size ∈ {0,1,2}
#define GREX_PARTLOAD_SUB_32_2(KIND) \
  using Dst = SubVector<KIND##32, 2, 4>; \
  GREX_PARTLOAD_SWITCH(KIND, 32, 2, Dst{GREX_KINDCAST(i, KIND, 32, 128, _mm_loadu_si32(ptr))}, \
                       Dst{GREX_KINDCAST(i, KIND, 32, 128, _mm_loadu_si64(ptr))})

#if GREX_X86_64_LEVEL >= 2
#define GREX_PARTLOAD_SUB_16_4(KIND) \
  GREX_PARTLOAD_FALLBACK_INIT(KIND, 16, 4) \
\
  /* 4-byte block path: size ∈ [2,3] */ \
  if (size >= 2) { \
    __m128i lo = _mm_loadu_si32(ptr); \
    __m128i hi = _mm_loadu_si32(ptr + (size - 2)); \
    /* AB = [ptr[0..1], ptr[size-2..size-1]] in bytes [0..3] */ \
    __m128i ab = _mm_unpacklo_epi32(lo, hi); \
\
    __m128i mask = load(shld::idxs4[2 * size - 4].data(), type_tag<u8x16>).r; \
\
    return Dst{_mm_shuffle_epi8(ab, mask)}; \
  } \
\
  /* size == 1 */ \
  return Dst{_mm_loadu_si16(ptr)};
#else
// 4×16-bit sub-vector: scalar memcpy into the low 64-bit lane.
#define GREX_PARTLOAD_SUB_16_4(KIND) \
  GREX_PARTLOAD_FALLBACK_INIT(KIND, 16, 4) \
  /* Accumulate 1-3 words into a 64-bit temporary in the low lane. */ \
  u64 out = 0; \
  if ((size & 1U) != 0) { \
    std::memcpy(&out, ptr + (size / 2 * 2), 2); \
  } \
  if ((size & 2U) != 0) { \
    out <<= 32; \
    std::memcpy(&out, ptr, 4); \
  } \
  return Dst{_mm_set_epi64x(0, std::bit_cast<i64>(out))};
#endif

// 2×16-bit sub-vector: size ∈ {0,1,2}
#define GREX_PARTLOAD_SUB_16_2(KIND) \
  using Dst = SubVector<KIND##16, 2, 8>; \
  GREX_PARTLOAD_SWITCH(KIND, 16, 2, Dst{GREX_KINDCAST(i, KIND, 16, 128, _mm_loadu_si16(ptr))}, \
                       Dst{GREX_KINDCAST(i, KIND, 16, 128, _mm_loadu_si32(ptr))})

#if GREX_X86_64_LEVEL >= 2
#define GREX_PARTLOAD_SUB_8_8(KIND) \
  GREX_PARTLOAD_FALLBACK_INIT(KIND, 8, 8) \
\
  /* 4-byte block path: size ∈ [4,7] */ \
  if (size >= 4) { \
    const __m128i lo = _mm_loadu_si32(ptr); \
    const __m128i hi = _mm_loadu_si32(ptr + (size - 4)); \
    /* AB = [ptr[0..3], ptr[size-4..size-1]] in bytes [0..7] */ \
    const __m128i ab = _mm_unpacklo_epi32(lo, hi); \
\
    const __m128i mask = load(shld::idxs4[size - 4].data(), type_tag<u8x16>).r; \
    return Dst{_mm_shuffle_epi8(ab, mask)}; \
  } \
\
  /* 2-byte block path: size ∈ [2,3] */ \
  if (size >= 2) { \
    const __m128i lo = _mm_loadu_si16(ptr); \
    const __m128i hi = _mm_loadu_si16(ptr + (size - 2)); \
    /* AB = [ptr[0..1], ptr[size-2..size-1]] in bytes [0..3] */ \
    const __m128i ab = _mm_unpacklo_epi16(lo, hi); \
\
    const __m128i mask = load(shld::idxs2[size - 2].data(), type_tag<u8x16>).r; \
    return Dst{_mm_shuffle_epi8(ab, mask)}; \
  } \
\
  /* size == 1 */ \
  return Dst{_mm_cvtsi32_si128(u8(*ptr))};
#else
// 8×8-bit sub-vector: scalar memcpy accumulation into low 64 bits.
#define GREX_PARTLOAD_SUB_8_8(KIND) \
  GREX_PARTLOAD_FALLBACK_INIT(KIND, 8, 8) \
  /* Accumulate 1-7 bytes into a 64-bit temporary in the low lane. */ \
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
  return Dst{_mm_set_epi64x(0, std::bit_cast<i64>(out))};
#endif

#if GREX_X86_64_LEVEL >= 2
#define GREX_PARTLOAD_SUB_8_4(KIND) \
  GREX_PARTLOAD_FALLBACK_INIT(KIND, 8, 4) \
\
  /* 2-byte block path: size ∈ [2,3] */ \
  if (size >= 2) { \
    const __m128i lo = _mm_loadu_si16(ptr); \
    const __m128i hi = _mm_loadu_si16(ptr + (size - 2)); \
    /* AB = [ptr[0..1], ptr[size-2..size-1]] in bytes [0..3] */ \
    const __m128i ab = _mm_unpacklo_epi16(lo, hi); \
\
    const __m128i mask = load(shld::idxs2[size - 2].data(), type_tag<u8x16>).r; \
    return Dst{_mm_shuffle_epi8(ab, mask)}; \
  } \
\
  /* size == 1 */ \
  return Dst{_mm_cvtsi32_si128(u8(*ptr))};
#else
// 4×8-bit sub-vector: scalar memcpy accumulation into low 64 bits.
#define GREX_PARTLOAD_SUB_8_4(KIND) \
  GREX_PARTLOAD_FALLBACK_INIT(KIND, 8, 4) \
  /* Accumulate 1-3 bytes into a 64-bit temporary in the low lane. */ \
  u64 out = 0; \
  if ((size & 1U) != 0) { \
    std::memcpy(&out, ptr + (size / 2 * 2), 1); \
  } \
  if ((size & 2U) != 0) { \
    out <<= 16; \
    std::memcpy(&out, ptr + (size / 4 * 4), 2); \
  } \
  return Dst{_mm_set_epi64x(0, std::bit_cast<i64>(out))};
#endif

// 2×8-bit sub-vector: size ∈ {0,1,2} using scalar byte/16-bit loads.
#define GREX_PARTLOAD_SUB_8_2(KIND) \
  using Dst = SubVector<KIND##8, 2, 16>; \
  GREX_PARTLOAD_SWITCH( \
    KIND, 8, 2, \
    Dst{GREX_KINDCAST(i, KIND, 8, 128, \
                      _mm_and_si128(_mm_set_epi64x(0, 255), \
                                    _mm_set1_epi8(GREX_KINDCAST_SINGLE(KIND, i, 8, ptr[0]))))}, \
    Dst{GREX_KINDCAST(i, KIND, 8, 128, _mm_loadu_si16(ptr))})

#define GREX_PARTLOAD_SUB_IMPL(KIND, BITS, PART, SIZE) GREX_PARTLOAD_SUB_##BITS##_##PART(KIND)
#else
// On AVX-512, sub-vectors delegate to the native vector partial-load logic.
#define GREX_PARTLOAD_SUB_IMPL(KIND, BITS, PART, SIZE) \
  return SubVector<KIND##BITS, PART, SIZE>{ \
    load_part(ptr, size, type_tag<NativeVector<KIND##BITS, SIZE>>)};
#endif

#if GREX_X86_64_LEVEL == 2
template<AnyNativeVector THalf>
GREX_ALWAYS_INLINE inline SuperVector<THalf> load_part(const ValueOf<THalf>* ptr, std::size_t size,
                                                       grex::TypeTag<SuperVector<THalf>> tag) {
  using Value = ValueOf<THalf>;
  constexpr std::size_t vsize = 2 * size_of<THalf>;

  if (size <= vsize / 2) {
    return merge(load_part(ptr, size, grex::type_tag<THalf>), zeros(type_tag<THalf>));
  }

  if (size >= vsize) [[unlikely]] {
    return load(ptr, tag);
  }

  // 16-byte block path: len ∈ [vsize/2, vsize]
  const auto lo = load(ptr, type_tag<THalf>);
  const auto hi = as<u8>(load(ptr + (size - vsize / 2), type_tag<THalf>)).r;

  __m128i mask = load(shld::idxs16[sizeof(Value) * size - 16].data(), type_tag<u8x16>).r;
  return merge(lo, as<Value>(u8x16{_mm_shuffle_epi8(hi, mask)}));
}
#endif

// Entry point for sub-vector partial loads.
#define GREX_PARTLOAD_SUB(KIND, BITS, PART, SIZE) \
  inline SubVector<KIND##BITS, PART, SIZE> load_part(const KIND##BITS* ptr, std::size_t size, \
                                                     TypeTag<SubVector<KIND##BITS, PART, SIZE>>) { \
    GREX_PARTLOAD_SUB_IMPL(KIND, BITS, PART, SIZE) \
  }
GREX_FOREACH_SUB(GREX_PARTLOAD_SUB)
} // namespace grex::backend

#include "grex/backend/shared/operations/load.hpp" // IWYU pragma: export

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_LOAD_HPP
