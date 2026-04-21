// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_MULTIBYTE_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_MULTIBYTE_HPP

#include <cstddef>

#include <immintrin.h>

#include "grex/backend/base.hpp"
#include "grex/backend/choosers.hpp"
#include "grex/backend/x86/instruction-sets.hpp" // IWYU pragma: keep
#include "grex/backend/x86/operations/load.hpp"
#include "grex/backend/x86/types.hpp"
#include "grex/base.hpp"

#if GREX_X86_64_LEVEL >= 4
#include "grex/backend/x86/operations/set.hpp"
#endif

// Shared definitions.
#include "grex/backend/shared/operations/multibyte.hpp" // IWYU pragma: keep

// Load integers consisting of M bytes into integers consisting of N = 2^B bytes,
// assuming N = std::bitceil(M).
// We call M the `src_bytes` and N the `dst_bytes`, while `size` denotes the number of values
// being converted.
// The underlying memory is assumed to be padded at the beginning and end by the number of bytes
// in the largest supported SIMD register.

namespace grex::backend {
// N == M: trivial case, just load and rewrap.
template<std::size_t tSrc, AnyVector TDst>
requires(!AnySuperNativeVector<TDst> && tSrc == sizeof(typename TDst::Value))
inline TDst load_multibyte(const u8* ptr, IndexTag<tSrc> /*src*/, TypeTag<TDst> /*dst*/) {
  const auto raw = load(ptr, type_tag<VectorFor<u8, tSrc * TDst::size>>).registr();
  return TDst{raw};
}

#if GREX_X86_64_LEVEL >= 2
// Generic SSSE3 path for 16-byte native registers using PSHUFB.
template<std::size_t tSrc, typename TDst>
requires(tSrc < sizeof(typename TDst::Value) && sizeof(typename TDst::Register) == 16)
inline TDst load_multibyte(const u8* ptr, IndexTag<tSrc> /*src*/, TypeTag<TDst> /*dst*/) {
  using Value = TDst::Value;
  static constexpr std::size_t size = TDst::size;
  static constexpr std::size_t full_size = sizeof(typename TDst::Register) / sizeof(Value);
  static constexpr auto idxs_arr = mb::shuffle_indices_128<tSrc, sizeof(Value), full_size, size>;
  const __m128i raw = load(ptr, type_tag<VectorFor<u8, sizeof(Value) * size>>).registr();
  const __m128i idxs =
    static_apply<16>([]<std::size_t... tIdxs> { return _mm_setr_epi8(idxs_arr[tIdxs]...); });
  return TDst{_mm_shuffle_epi8(raw, idxs)};
}

// Specialized SSSE3 path for loading 2 × 6-byte values into 2 × 8-byte u64.
inline u64x2 load_multibyte(const u8* ptr, IndexTag<6> /*src*/, TypeTag<u64x2> /*dst*/) {
  // ..000000|111111.. (load 16 bytes starting 2 bytes before `ptr`)
  const __m128i raw = load(ptr - 2, type_tag<u8x16>).r;
  // 000000..|111111.. (rotate lowest 16-bit words into position)
  const __m128i shu = _mm_shufflelo_epi16(raw, 0b11'10'01);
  // Zero out the high padding bytes in each 64-bit lane.
  return {.r = _mm_blend_epi16(shu, _mm_setzero_si128(), 0b10'00'10'00)};
}
#else
template<std::size_t tSrc>
requires(tSrc < 8)
inline u64x2 load_multibyte(const u8* ptr, IndexTag<tSrc> /*src*/, TypeTag<u64x2> /*dst*/) {
  // The comments assume M == 5; M == 6 and 7 are analogous.
  // offset = dst_bytes - src_bytes = 8 - 5 = 3
  constexpr std::size_t offset = 8 - tSrc;
  // ...00000|11111... (raw bytes padded around the two 5-byte integers)
  const __m128i raw = load(ptr - offset, type_tag<u8x16>).r;

  const __m128i v0 = raw;
  // ···...00|···11111| (shift each 64-bit lane left so that the lower integer becomes top-aligned)
  const __m128i v1 = _mm_slli_epi64(raw, 8 * offset);

  // ...00000|···11111 (move low 64-bit from v1 to high 64-bit of v0)
  const __m128d mix = _mm_move_sd(_mm_castsi128_pd(v1), _mm_castsi128_pd(v0));
  // 00000···|11111··· (shift right to bottom-align both integers and drop the padding)
  return {.r = _mm_srli_epi64(_mm_castpd_si128(mix), 8 * offset)};
}

// Expand 4 × 3-byte values to 4 × 4-byte u32 using only SSE2.
inline u32x4 load_multibyte(const u8* ptr, IndexTag<3> /*src*/, TypeTag<u32x4> /*dst*/) {
  // ..00|0111|2223|33..
  const __m128i raw = load(ptr - 2, type_tag<u8x16>).r;

  // Isolate and align bytes for elements 0 and 1:
  // v0:  ···.|.000|···1|1122 (move element 0 into position)
  const __m128i v0 = _mm_slli_epi64(raw, 24);
  // v1:  ..00|0111|2223|33.. (original positions)
  const __m128i v1 = raw;
  // v01: ···.|..00|.000|0111 (interleave elements 0 and 1)
  const __m128i v01 = _mm_unpacklo_epi32(v0, v1);

  // Isolate and align bytes for elements 2 and 3:
  // v2:  ····|·..0|····|·222 (move element 2 into position)
  const __m128i v2 = _mm_slli_epi64(raw, 40);
  // v3:  ··..|0001|··22|2333 (move element 3 into position)
  const __m128i v3 = _mm_slli_epi64(raw, 16);
  // v23: ····|··22|·222|2333 (interleave elements 2 and 3)
  const __m128i v23 = _mm_unpackhi_epi32(v2, v3);

  // Combine and drop the padding byte per element:
  return {.r = _mm_srli_epi32(_mm_unpackhi_epi64(v01, v23), 8)};
}

// SSE2 version for partially filled u32x4 (only first 2 lanes valid).
inline SubVector<u32, 2, 4> load_multibyte(const u8* ptr, IndexTag<3> /*src*/,
                                           TypeTag<SubVector<u32, 2, 4>> /*dst*/) {
  // .000|111.|....|....
  const __m128i raw = load(ptr - 1, type_tag<SubVector<u8, 8, 16>>).registr();

  const __m128i v0 = raw;
  // ·.00|·111|·...|·... (shift left so that the high byte of each element is in place)
  const __m128i v1 = _mm_slli_epi32(raw, 8);

  // .000|·111|·...|·... (move low 32 bits of v0 into high 32 bits of v1 to pack the two elements)
  const __m128 mix = _mm_move_ss(_mm_castsi128_ps(v1), _mm_castsi128_ps(v0));
  return SubVector<u32, 2, 4>{_mm_srli_epi32(_mm_castps_si128(mix), 8)};
}
#endif

#if GREX_X86_64_LEVEL >= 3
// AVX2: load `tSize` elements of size `tSrc` into 32 bytes (256 bits).
template<std::size_t tSrc, typename TDst, std::size_t tSize>
requires(tSrc < sizeof(TDst) && (sizeof(TDst) * tSize) == 32)
inline NativeVector<TDst, tSize> load_multibyte(const u8* ptr, IndexTag<tSrc> /*src*/,
                                                TypeTag<NativeVector<TDst, tSize>> /*dst*/) {
  static constexpr std::size_t src_bytes = tSrc;
  static constexpr std::size_t dst_bytes = sizeof(TDst);
  // Zero padding per element.
  static constexpr std::size_t offset = dst_bytes - src_bytes;

  // Load a contiguous block that contains all elements plus their leading padding.
  const auto raw =
    load(ptr - (tSize / 2) * offset, type_tag<NativeVector<u8, dst_bytes * tSize>>).r;

  // Build PSHUFB indices:
  // - Bytes belonging to real data map to their source position.
  // - Padding is mapped to -1 (zero).
  const auto idxs = static_apply<dst_bytes * tSize>([&]<std::size_t... tIdxs>() {
    return _mm256_setr_epi8(
      ((tIdxs % dst_bytes < src_bytes)
         ? i8(tIdxs % dst_bytes + (tIdxs / dst_bytes) * src_bytes + (tSize / 2) * offset)
         : i8(-1))...);
  });
  return {.r = _mm256_shuffle_epi8(raw, idxs)};
}
#endif

#if GREX_X86_64_LEVEL >= 4
// AVX-512: load `tSize` elements of size `tSrc` into 64 bytes (512 bits).
template<std::size_t tSrc, typename TDst, std::size_t tSize>
requires(tSrc < sizeof(TDst) && (sizeof(TDst) * tSize) == 64)
inline NativeVector<TDst, tSize> load_multibyte(const u8* ptr, IndexTag<tSrc> /*src*/,
                                                TypeTag<NativeVector<TDst, tSize>> /*dst*/) {
  // The comments are based on M == 5; M == 6 and 7 are analogous.

  static constexpr std::size_t src_bytes = tSrc;
  static constexpr std::size_t dst_bytes = sizeof(TDst);
  // Total padding across all elements; always a multiple of 8.
  // offset = (dst_bytes - src_bytes) * tSize = (8 - 5) * 8 = 24
  static constexpr std::size_t offset = (dst_bytes - src_bytes) * tSize;

  // Load with an offset of half the total padding so that each half of the elements ends up in the
  // correct 256-bit half of the 512-bit register.
  // ........|....0000|01111122|22233333|44444555|55666667|7777....|........
  __m512i out = load(ptr - offset / 2, type_tag<NativeVector<u8, dst_bytes * tSize>>).r;

  // First, permute 32-bit chunks into the correct 128-bit lanes.
  //
  // - The two middle 128-bit lanes stay where they are.
  // - The outer lanes are moved by offset / 2 bytes to the very beginning/end.
  //
  // After this, lanes 0 and 2 (even indices) are bottom-aligned, while lanes 1 and 3
  // are top-aligned within their 128-bit lanes.
  // 00000111|11222223|01111122|22233333|44444555|55666667|45555566|66677777
  const __m512i idxs32 = static_apply<16>([]<std::size_t... tIdxs>() {
    return set(type_tag<NativeVector<i32, 16>>,
               i32{(tIdxs < 4) ? (tIdxs + offset / 8)
                               : ((tIdxs >= 12) ? (tIdxs - offset / 8) : tIdxs)}...)
      .r;
  });
  out = _mm512_permutexvar_epi32(idxs32, out);

  // Then perform a byte-wise shuffle within each 128-bit lane.
  //
  // For each output byte index i in [0, 64), we:
  // 1. Check if it is inside the padding; if so, use -1 (zero).
  // 2. Otherwise compute the source byte index as a sum of:
  //    (a) index within the element (`i % dst_bytes`, bounded by src_bytes),
  //    (b) element offset within the 128-bit lane,
  //    (c) additional offset for top-aligned lanes (odd lane indices).
  //
  // In the example (M == 5, N == 8, tSize == 8), these components look like:
  // (a) 01234...|01234...|01234...|01234...|01234...|01234...|01234...|01234...
  // (b) 00000...|55555...|00000...|55555...|00000...|55555...|00000...|55555...
  // (c) 00000...|00000...|66666...|66666...|00000...|00000...|66666...|66666...
  const __m512i idxs8 = static_apply<64>([]<std::size_t... tIdxs>() {
    return set(type_tag<NativeVector<i8, 64>>,
               ((tIdxs % dst_bytes < src_bytes)
                  ? i8{tIdxs % dst_bytes + ((tIdxs % 16) / dst_bytes) * src_bytes +
                       ((tIdxs / 16) % 2) * (offset / 4)}
                  : i8{-1})...)
      .r;
  });
  return {.r = _mm512_shuffle_epi8(out, idxs8)};
}
#endif
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_MULTIBYTE_HPP
