// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_SHUFFLE_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_SHUFFLE_HPP

#include <cstddef>

#include <immintrin.h>

#include "grex/backend/base.hpp"
#include "grex/backend/choosers.hpp"
#include "grex/backend/x86/instruction-sets.hpp"
#include "grex/backend/x86/operations/compare.hpp"
#include "grex/backend/x86/operations/convert.hpp"
#include "grex/backend/x86/operations/merge.hpp"
#include "grex/backend/x86/operations/reinterpret.hpp"
#include "grex/backend/x86/operations/set.hpp"
#include "grex/backend/x86/operations/shift.hpp"
#include "grex/backend/x86/operations/shrink.hpp"
#include "grex/backend/x86/operations/split.hpp"
#include "grex/base.hpp"

#if GREX_X86_64_LEVEL > 1
#include <algorithm>
#include <array>
#include <bit>
#include <limits>
#include <utility>

#include "grex/backend/macros/for-each.hpp"
#include "grex/backend/macros/math.hpp"
#include "grex/backend/x86/macros/for-each.hpp"
#include "grex/backend/x86/macros/types.hpp"
#include "grex/backend/x86/operations/arithmetic.hpp"
#include "grex/backend/x86/operations/expand.hpp"
#include "grex/backend/x86/operations/load.hpp"
#include "grex/backend/x86/types.hpp"
#else
#include <concepts>

#include "grex/backend/x86/operations/blend.hpp"
#endif

#if GREX_X86_64_LEVEL >= 4
#include "grex/backend/macros/base.hpp"
#include "grex/backend/x86/macros/intrinsics.hpp"
#endif

namespace grex::backend {
#if GREX_X86_64_LEVEL > 1
// Thin wrappers to abstract over 128/256-bit byte-wise shuffles.
GREX_ALWAYS_INLINE inline __m128i shuffle_epi8(__m128i a, __m128i b) {
  return _mm_shuffle_epi8(a, b);
}
#if GREX_X86_64_LEVEL >= 3
GREX_ALWAYS_INLINE inline __m256i shuffle_epi8(__m256i a, __m256i b) {
  return _mm256_shuffle_epi8(a, b);
}
#endif

// Compute indices for a shuffle operation acting on tDstBytes-sized chunks using a table with
// tValueBytes-sized values.
//
// TIdxs:       vector of indices in units of tValueBytes.
// tDstBytes:   shuffling chunks size in bytes (here 1 → byte indices).
// tValueBytes: size of a single table element in bytes.
template<UnsignedIntVector TIdxs, std::size_t tDstBytes, std::size_t tValueBytes>
requires(sizeof(ValueOf<TIdxs>) <= tValueBytes && tDstBytes == 1 && tValueBytes != 1)
GREX_ALWAYS_INLINE inline NativeVector<UnsignedInt<tDstBytes>,
                                       size_of<TIdxs> * tValueBytes / tDstBytes>
shuffle_indices(TIdxs idxs, IndexTag<tDstBytes> /*dst_bytes*/,
                IndexTag<tValueBytes> /*value_bytes*/) {
  using Src = ValueOf<TIdxs>;
  constexpr std::size_t dst_bytes = size_of<TIdxs> * tValueBytes;
  constexpr std::size_t dst_size = dst_bytes / tDstBytes;

  // For each destination byte, select the starting byte of the source table element.
  constexpr auto shuf = grex::static_apply<dst_size>([]<std::size_t... tI>() {
    return std::array<u8, dst_size>{(tI / tValueBytes) * sizeof(Src)...};
  });
  // Per-byte offsets within each value.
  constexpr auto offs = grex::static_apply<dst_size>(
    []<std::size_t... tI>() { return std::array<u8, dst_size>{tI % tValueBytes...}; });

  using UnIntVec = NativeVector<u8, dst_bytes>;
  const auto vshuf = load(shuf.data(), type_tag<UnIntVec>);
  const auto voffs = load(offs.data(), type_tag<UnIntVec>);

  // Scale element indices by tValueBytes in bytes → shift by log2(tValueBytes).
  const auto rscaled = shift_left(idxs, index_tag<std::size_t{std::bit_width(tValueBytes)} - 1>);
  // If the underlying native register is smaller than dst_bytes, duplicate it to fill.
  const auto scaled = [rscaled] {
    if constexpr (sizeof(typename TIdxs::Register) < dst_bytes) {
      return repeat<2>(rscaled.native());
    } else {
      return rscaled;
    }
  }();
  // Byte indices = shuffled, scaled element indices + per-byte offsets.
  return add(UnIntVec{.r = shuffle_epi8(scaled.registr(), vshuf.r)}, voffs);
}

// Widen indices larger than tValueBytes to tValueBytes-sized integers, then delegate to other
// overloads.
template<UnsignedIntVector TIdxs, std::size_t tDstBytes, std::size_t tValueBytes>
requires(sizeof(ValueOf<TIdxs>) > tValueBytes && tDstBytes != tValueBytes)
GREX_ALWAYS_INLINE inline NativeVector<UnsignedInt<tDstBytes>,
                                       TIdxs::size * tValueBytes / tDstBytes>
shuffle_indices(TIdxs idxs, IndexTag<tDstBytes> dst_bytes, IndexTag<tValueBytes> value_bytes) {
  return shuffle_indices(convert<UnsignedInt<tValueBytes>>(idxs), dst_bytes, value_bytes);
}

// tDstBytes and tValueBytes are identical: simply convert the indices.
template<UnsignedIntVector TIdxs, std::size_t tDstBytes, std::size_t tValueBytes>
requires(tDstBytes == tValueBytes)
GREX_ALWAYS_INLINE inline NativeVector<UnsignedInt<tDstBytes>, TIdxs::size>
shuffle_indices(TIdxs idxs, IndexTag<tDstBytes> /*dst_bytes*/,
                IndexTag<tValueBytes> /*value_bytes*/) {
  return convert<UnsignedInt<tValueBytes>>(idxs);
}

#if GREX_X86_64_LEVEL >= 3
// Specialized computations for 256-bit shuffles with 32-bit chunks (vpermd) and 64-bit values

// 4×u64 → 8×u32: replicate the lower 32 bits into the upper 32 bits, double each index, and offset
// by [0, 1].
GREX_ALWAYS_INLINE inline u32x8 shuffle_indices(u64x4 idxs, IndexTag<4> /*dst_bytes*/,
                                                IndexTag<8> /*value_bytes*/) {
  const auto idxs32 = _mm256_slli_epi64(_mm256_shuffle_epi32(idxs.r, 0b10100000), 1);
  return {.r = _mm256_add_epi32(idxs32, _mm256_setr_epi32(0, 1, 0, 1, 0, 1, 0, 1))};
}

// 4×u32 → 8×u32: replicate each 4-byte index to 8 adjacent bytes, double each index, and offset by
// [0, 1].
GREX_ALWAYS_INLINE inline u32x8 shuffle_indices(u32x4 idxs, IndexTag<4> /*dst_bytes*/,
                                                IndexTag<8> /*value_bytes*/) {
  const auto lo32 = _mm_shuffle_epi32(idxs.r, 0b01010000);
  const auto hi32 = _mm_shuffle_epi32(idxs.r, 0b11111010);
  const auto idxs32 = _mm256_slli_epi64(_mm256_setr_m128i(lo32, hi32), 1);
  return {.r = _mm256_add_epi32(idxs32, _mm256_setr_epi32(0, 1, 0, 1, 0, 1, 0, 1))};
}

// 4×u16 → 8×u32: replicate each 2-byte index to 8 adjacent bytes, double each index, and offset by
// [0, 1].
GREX_ALWAYS_INLINE inline u32x8 shuffle_indices(SubVector<u16, 4> idxs, IndexTag<4> /*dst_bytes*/,
                                                IndexTag<8> /*value_bytes*/) {
  const auto lo64 = _mm_shufflelo_epi16(idxs.full.r, 0b01010000);
  const auto hi64 = _mm_shufflelo_epi16(idxs.full.r, 0b11111010);
  const auto lo128 = _mm_unpacklo_epi64(lo64, hi64);
  const auto lo32 = _mm_shuffle_epi32(lo128, 0b01010000);
  const auto hi32 = _mm_shuffle_epi32(lo128, 0b11111010);
  const auto idxs32 = _mm256_slli_epi64(_mm256_setr_m128i(lo32, hi32), 1);
  return {.r = _mm256_add_epi32(idxs32, _mm256_setr_epi32(0, 1, 0, 1, 0, 1, 0, 1))};
}

// 4×u8 → 8×u32: replicate each 1-byte index to 8 adjacent bytes, double each index, and offset by
// [0, 1].
GREX_ALWAYS_INLINE inline u32x8 shuffle_indices(SubVector<u8, 4> idxs, IndexTag<4> /*dst_bytes*/,
                                                IndexTag<8> /*value_bytes*/) {
  const __m256i bcidxs = _mm256_broadcastd_epi32(idxs.registr());
  const std::array<u8, 32> shuf{0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                                2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3};
  const auto shidxs = _mm256_shuffle_epi8(bcidxs, load(shuf.data(), type_tag<u8x32>).r);
  const auto idxs32 = _mm256_add_epi64(shidxs, shidxs);
  return {.r = _mm256_add_epi32(idxs32, _mm256_setr_epi32(0, 1, 0, 1, 0, 1, 0, 1))};
}
#endif

// Macros for native-size shuffles.
//
// All of these macros assume:
//  - table is a native SIMD vector of KIND##BITS.
//  - idxs is a same-sized vector of unsigned indices (u8/u16/u32/u64).
//  - index_ub is the (exclusive) upper bound for indices.
//  - index_offset is a logical starting index (for split tables).

// SSSE3: byte-wise shuffle in a 128-bit lane using pshufb.
#define GREX_SHFL_PSHUFB(KIND, BITS, IDXBITS, SIZE, REGISTERBITS, BITPREFIX) \
  auto idxs8 = shuffle_indices(idxs, index_tag<1>, index_tag<GREX_DIVIDE(BITS, 8)>).r; \
  if constexpr (index_ub * BITS > 128) { \
    /* Clear high bit so elements are not inadvertently zeroed. */ \
    idxs8 = _mm_and_si128(idxs8, _mm_set1_epi8(127)); \
  } \
  const auto shuf8 = _mm_shuffle_epi8(as<u8>(table).r, idxs8); \
  return as<KIND##BITS>(GREX_VECTOR_TYPE(u, 8, GREX_DIVIDE(REGISTERBITS, 8)){shuf8});

// AVX2 64×2: use vpermilpd.
#define GREX_SHFL_VPERMILPD(KIND, BITS, IDXBITS, SIZE, REGISTERBITS, BITPREFIX) \
  const auto idxs64 = shuffle_indices(idxs, index_tag<8>, index_tag<GREX_DIVIDE(BITS, 8)>); \
  const auto shuf64 = _mm_permutevar_pd(as<f64>(table).r, shift_left(idxs64, index_tag<1>).r); \
  return as<KIND##BITS>(f64x2{shuf64});

// AVX2 32×4: use vpermilps.
#define GREX_SHFL_VPERMILPS(KIND, BITS, IDXBITS, SIZE, REGISTERBITS, BITPREFIX) \
  const auto idxs32 = shuffle_indices(idxs, index_tag<4>, index_tag<GREX_DIVIDE(BITS, 8)>).r; \
  const auto shuf32 = _mm_permutevar_ps(as<f32>(table).r, idxs32); \
  return as<KIND##BITS>(f32x4{shuf32});

// AVX2 32×8: use vpermd.
#define GREX_SHFL_VPERMD(KIND, BITS, IDXBITS, SIZE, REGISTERBITS, BITPREFIX) \
  const auto idxs32 = shuffle_indices(idxs, index_tag<4>, index_tag<GREX_DIVIDE(BITS, 8)>).r; \
  const auto shuf32 = _mm256_permutevar8x32_epi32(as<u32>(table).r, idxs32); \
  return as<KIND##BITS>(u32x8{shuf32});

// AVX2 8-bit: two independent 128-bit pshufb shuffles + blend across 256-bit lanes.
#define GREX_SHFL_PSHUFBx2(KIND, BITS, IDXBITS, SIZE, REGISTERBITS, BITPREFIX) \
  auto idxs8 = shuffle_indices(idxs, index_tag<1>, index_tag<GREX_DIVIDE(BITS, 8)>).r; \
  if constexpr (index_ub * BITS > 128) { \
    /* Clear high bit so elements are not inadvertently zeroed. */ \
    idxs8 = _mm256_and_si256(idxs8, _mm256_set1_epi8(127)); \
  } \
  /* Table with 128-bit lanes flipped (low ↔ high). */ \
  const u8x32 rtable{_mm256_permute4x64_epi64(as<u8>(table).r, 0x4E)}; \
  const auto shuf0 = _mm256_shuffle_epi8(as<u8>(table).r, idxs8); \
  const auto shuf1 = _mm256_shuffle_epi8(as<u8>(rtable).r, idxs8); \
  /* Bit 4 in the index selects which 128-bit lane the element comes from. */ \
  const auto bitmask = _mm256_set1_epi8(16); \
  const auto bitref = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 16, 16, \
                                       16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16); \
  const auto blendmask = _mm256_cmpeq_epi8(_mm256_and_si256(idxs8, bitmask), bitref); \
  const auto shuf8 = _mm256_blendv_epi8(shuf1, shuf0, blendmask); \
  return as<KIND##BITS>(GREX_VECTOR_TYPE(u, 8, GREX_DIVIDE(REGISTERBITS, 8)){shuf8});

// AVX-512: use direct permutexvar intrinsics for full-width element permutes.
#define GREX_SHFL_AVX512(KIND, BITS, IDXBITS, SIZE, REGISTERBITS, BITPREFIX) \
  const auto vidxs = \
    shuffle_indices(idxs, index_tag<GREX_DIVIDE(BITS, 8)>, index_tag<GREX_DIVIDE(BITS, 8)>).r; \
  const auto shuf = BITPREFIX##_permutexvar_epi##BITS(vidxs, as<u##BITS>(table).registr()); \
  return as<KIND##BITS>(GREX_VECTOR_TYPE(u, BITS, SIZE){shuf});

// AVX-512, 2×native table: use the vpermi2/permutex2var family for shuffles across two tables.
#define GREX_SHFL2_AVX512(KIND, BITS, IDXBITS, SIZE) \
  GREX_ALWAYS_INLINE inline VectorFor<KIND##BITS, GREX_DIVIDE(SIZE, 2)> shuffle( \
    VectorFor<KIND##BITS, SIZE> table, VectorFor<u##IDXBITS, GREX_DIVIDE(SIZE, 2)> idxs, \
    AnyIndexTag auto /*index_ub*/, AnyIndexTag auto /*index_offset*/) { \
    using UnIntVec = VectorFor<u##BITS, GREX_DIVIDE(SIZE, 2)>; \
    const auto itag = index_tag<GREX_DIVIDE(BITS, 8)>; \
    const auto vidxs = shuffle_indices(idxs, itag, itag).r; \
    const auto tbllo = as<u##BITS>(table).lower.r; \
    const auto tblhi = as<u##BITS>(table).upper.r; \
    return as<KIND##BITS>(UnIntVec{_mm512_permutex2var_epi##BITS(tbllo, vidxs, tblhi)}); \
  }

#if GREX_X86_64_LEVEL >= 4
// Map element width × element count to an implementation macro.
#define GREX_SHFL_IMPL_64x2 GREX_SHFL_VPERMILPD
#define GREX_SHFL_IMPL_32x4 GREX_SHFL_VPERMILPS
#define GREX_SHFL_IMPL_16x8 GREX_SHFL_AVX512
#define GREX_SHFL_IMPL_8x16 GREX_SHFL_PSHUFB
#define GREX_SHFL_IMPL_64x4 GREX_SHFL_AVX512
#define GREX_SHFL_IMPL_32x8 GREX_SHFL_VPERMD
#define GREX_SHFL_IMPL_16x16 GREX_SHFL_AVX512
#if GREX_HAS_AVX512VBMI
#define GREX_SHFL_IMPL_8x32 GREX_SHFL_AVX512
#else
#define GREX_SHFL_IMPL_8x32 GREX_SHFL_PSHUFBx2
#endif
#define GREX_SHFL_IMPL_64x8 GREX_SHFL_AVX512
#define GREX_SHFL_IMPL_32x16 GREX_SHFL_AVX512
#define GREX_SHFL_IMPL_16x32 GREX_SHFL_AVX512

// Super-native tables: use vermi2 with varying index widths.
#define GREX_SHFL2_AVX512_IDXS(KIND, BITS, SIZE) \
  GREX_SHFL2_AVX512(KIND, BITS, 64, SIZE) \
  GREX_SHFL2_AVX512(KIND, BITS, 32, SIZE) \
  GREX_SHFL2_AVX512(KIND, BITS, 16, SIZE) \
  GREX_SHFL2_AVX512(KIND, BITS, 8, SIZE)
GREX_SHFL2_AVX512_IDXS(f, 64, 16)
GREX_SHFL2_AVX512_IDXS(i, 64, 16)
GREX_SHFL2_AVX512_IDXS(u, 64, 16)
GREX_SHFL2_AVX512_IDXS(f, 32, 32)
GREX_SHFL2_AVX512_IDXS(i, 32, 32)
GREX_SHFL2_AVX512_IDXS(u, 32, 32)
GREX_SHFL2_AVX512_IDXS(i, 16, 64)
GREX_SHFL2_AVX512_IDXS(u, 16, 64)
#if GREX_HAS_AVX512VBMI
#define GREX_SHFL_IMPL_8x64 GREX_SHFL_AVX512
GREX_SHFL2_AVX512_IDXS(i, 8, 128)
GREX_SHFL2_AVX512_IDXS(u, 8, 128)
#else
// AVX-512 without VBMI: emulate 8-bit 512-bit shuffles by broadcasting 128-bit lanes.
#define GREX_SHFL_IMPL_8x64(KIND, BITS, IDXBITS, SIZE, REGISTERBITS, BITPREFIX) \
  auto vidxs = \
    shuffle_indices(idxs, index_tag<GREX_DIVIDE(BITS, 8)>, index_tag<GREX_DIVIDE(BITS, 8)>); \
  if constexpr (index_ub > 64) { \
    /* Mask indices to ensure the 128-bit lane selection works. */ \
    vidxs = {.r = _mm512_and_si512(vidxs.r, _mm512_set1_epi8(63))}; \
  } \
  /* Broadcast each 128-bit lane to a full 512-bit register. */ \
  const __m512i lane0 = _mm512_broadcast_i32x4(_mm512_castsi512_si128(table.r)); \
  const __m512i lane1 = _mm512_shuffle_i64x2(table.r, table.r, 0x55); \
  const __m512i lane2 = _mm512_shuffle_i64x2(table.r, table.r, 0xAA); \
  const __m512i lane3 = _mm512_shuffle_i64x2(table.r, table.r, 0xFF); \
  /* Upper nibble of index selects the lane (0..3). */ \
  const auto laneselect = shift_right(vidxs, index_tag<4>).r; \
  /* Perform lane-local shuffle and select the right lane via masks. */ \
  const auto dat0 = _mm512_maskz_shuffle_epi8( \
    _mm512_cmpeq_epi8_mask(laneselect, _mm512_set1_epi8(0)), lane0, vidxs.r); \
  const auto dat1 = _mm512_mask_shuffle_epi8( \
    dat0, _mm512_cmpeq_epi8_mask(laneselect, _mm512_set1_epi8(1)), lane1, vidxs.r); \
  const auto dat2 = _mm512_maskz_shuffle_epi8( \
    _mm512_cmpeq_epi8_mask(laneselect, _mm512_set1_epi8(2)), lane2, vidxs.r); \
  const auto dat3 = _mm512_mask_shuffle_epi8( \
    dat2, _mm512_cmpeq_epi8_mask(laneselect, _mm512_set1_epi8(3)), lane3, vidxs.r); \
  return {.r = _mm512_or_si512(dat1, dat3)};
#endif

// Generic “pick implementation macro” for native-size shuffles.
#define GREX_SHFL_IMPL(KIND, BITS, IDXBITS, SIZE, REGISTERBITS, BITPREFIX) \
  GREX_SHFL_IMPL_##BITS##x##SIZE(KIND, BITS, IDXBITS, SIZE, REGISTERBITS, BITPREFIX)
#elif GREX_X86_64_LEVEL == 3
// AVX2-only implementation matrix.
#define GREX_SHFL_IMPL_64x2 GREX_SHFL_VPERMILPD
#define GREX_SHFL_IMPL_32x4 GREX_SHFL_VPERMILPS
#define GREX_SHFL_IMPL_16x8 GREX_SHFL_PSHUFB
#define GREX_SHFL_IMPL_8x16 GREX_SHFL_PSHUFB
#define GREX_SHFL_IMPL_64x4 GREX_SHFL_VPERMD
#define GREX_SHFL_IMPL_32x8 GREX_SHFL_VPERMD
#define GREX_SHFL_IMPL_16x16 GREX_SHFL_PSHUFBx2
#define GREX_SHFL_IMPL_8x32 GREX_SHFL_PSHUFBx2
#define GREX_SHFL_IMPL(KIND, BITS, IDXBITS, SIZE, REGISTERBITS, BITPREFIX) \
  GREX_SHFL_IMPL_##BITS##x##SIZE(KIND, BITS, IDXBITS, SIZE, REGISTERBITS, BITPREFIX)
#elif GREX_X86_64_LEVEL == 2
// SSSE3-only implementation: pshufb on 128-bit lanes.
#define GREX_SHFL_IMPL GREX_SHFL_PSHUFB
#endif

// Generate “native table, native-sized output” shuffle overloads for all type/size combos.
//
// KIND:         f/i/u
// BITS:         element width
// IDXBITS:      index element width
// SIZE:         element count in vector
// REGISTERBITS: bit-width of the underlying SIMD register
// BITPREFIX:    intrinsic name prefix (e.g. _mm256)
#define GREX_SHFL(KIND, BITS, IDXBITS, SIZE, REGISTERBITS, BITPREFIX) \
  GREX_ALWAYS_INLINE inline NativeVector<KIND##BITS, SIZE> shuffle( \
    NativeVector<KIND##BITS, SIZE> table, VectorFor<u##IDXBITS, SIZE> idxs, \
    [[maybe_unused]] AnyIndexTag auto index_ub, [[maybe_unused]] AnyIndexTag auto index_offset) { \
    GREX_SHFL_IMPL(KIND, BITS, IDXBITS, SIZE, REGISTERBITS, BITPREFIX) \
  }

// Instantiate shuffles for all index sizes for a given element type/size.
#define GREX_SHFL_IDXS(KIND, BITS, SIZE, REGISTERBITS, BITPREFIX) \
  GREX_SHFL(KIND, BITS, 64, SIZE, REGISTERBITS, BITPREFIX) \
  GREX_SHFL(KIND, BITS, 32, SIZE, REGISTERBITS, BITPREFIX) \
  GREX_SHFL(KIND, BITS, 16, SIZE, REGISTERBITS, BITPREFIX) \
  GREX_SHFL(KIND, BITS, 8, SIZE, REGISTERBITS, BITPREFIX)

// Instantiate for all supported element types at a given register width.
#define GREX_SHFL_ALL(REGISTERBITS, BITPREFIX) \
  GREX_FOREACH_TYPE_R(GREX_SHFL_IDXS, REGISTERBITS, REGISTERBITS, BITPREFIX)
GREX_FOREACH_X86_64_LEVEL(GREX_SHFL_ALL)

// Define multi-shuffles
//
// These handle the case where the output size is larger than a single native table:
// - repeat the table REPS times
// - perform a native shuffle
// - reinterpret back to the requested type.
#if GREX_X86_64_LEVEL >= 4
#define GREX_SHFL_MULTI_BASE(REPS, KIND, WORKKIND, BITS, DSTSIZE, SRCSIZE, IDXBITS, IMPL) \
  GREX_ALWAYS_INLINE inline KIND##BITS##x##DSTSIZE shuffle( \
    KIND##BITS##x##SRCSIZE table, VectorFor<u##IDXBITS, DSTSIZE> idxs, \
    [[maybe_unused]] AnyIndexTag auto index_ub, [[maybe_unused]] AnyIndexTag auto index_offset) { \
    constexpr auto tag = index_tag<GREX_DIVIDE(BITS, 8)>; \
    const auto vidxs = shuffle_indices(idxs, tag, tag).r; \
    /* Work on a repeated table in WORKKIND representation. */ \
    const auto xtable = as<WORKKIND##BITS>(repeat<REPS>(table)).r; \
    const auto shuf = IMPL; \
    return as<KIND##BITS>(WORKKIND##BITS##x##DSTSIZE{shuf}); \
  }

// Floating multi-permutes: always work via f32/f64 intrinsics.
#define GREX_SHFL_FPERMUTE_f32(REGISTERBITS) _mm##REGISTERBITS##_permutevar_ps(xtable, vidxs)
#define GREX_SHFL_FPERMUTE_f64(REGISTERBITS) \
  _mm##REGISTERBITS##_permutevar_pd(xtable, _mm##REGISTERBITS##_slli_epi64(vidxs, 1))
#define GREX_SHFL_FPERMUTE(BITS, REGISTERBITS) GREX_SHFL_FPERMUTE_f##BITS(REGISTERBITS)

// 4× replicated small table, permuted via floating permutes.
#define GREX_SHFL4_I(KIND, BITS, DSTSIZE, SRCSIZE, IDXBITS, REGISTERBITS) \
  GREX_SHFL_MULTI_BASE(4, KIND, f, BITS, DSTSIZE, SRCSIZE, IDXBITS, \
                       GREX_SHFL_FPERMUTE(BITS, REGISTERBITS))
#define GREX_SHFL4(KIND, BITS, SIZE, IDXBITS) \
  GREX_SHFL4_I(KIND, BITS, GREX_MULTIPLY(SIZE, 4), SIZE, IDXBITS, \
               GREX_MULTIPLY(GREX_MULTIPLY(BITS, SIZE), 4))

// 2× replicated table: 256- or 512-bit permutes depending on REGISTERBITS.
#define GREX_SHFL2_PERMUTE_256(BITS) GREX_SHFL_FPERMUTE(BITS, 256)
#define GREX_SHFL2_PERMUTE_512(BITS) \
  GREX_CAT(_mm512_permutexvar_, GREX_EPI_SUFFIX(f, BITS))(vidxs, xtable)
#define GREX_SHFL2_II(KIND, BITS, DSTSIZE, SRCSIZE, IDXBITS, REGISTERBITS) \
  GREX_SHFL_MULTI_BASE(2, KIND, f, BITS, DSTSIZE, SRCSIZE, IDXBITS, \
                       GREX_SHFL2_PERMUTE_##REGISTERBITS(BITS))
#define GREX_SHFL2_I(KIND, BITS, DSTSIZE, SRCSIZE, IDXBITS, REGISTERBITS) \
  GREX_SHFL2_II(KIND, BITS, DSTSIZE, SRCSIZE, IDXBITS, REGISTERBITS)
#define GREX_SHFL2(KIND, BITS, SIZE, IDXBITS) \
  GREX_SHFL2_I(KIND, BITS, GREX_MULTIPLY(SIZE, 2), SIZE, IDXBITS, \
               GREX_MULTIPLY(GREX_MULTIPLY(BITS, SIZE), 2))

// Multi-shuffle instantiations for “big” element sizes.
#define GREX_SHFL_MULTI_BIG(KIND, BITS, IDXBITS) \
  GREX_SHFL2(KIND, BITS, GREX_DIVIDE(128, BITS), IDXBITS) \
  GREX_SHFL2(KIND, BITS, GREX_DIVIDE(256, BITS), IDXBITS) \
  GREX_SHFL4(KIND, BITS, GREX_DIVIDE(128, BITS), IDXBITS)
#define GREX_SHFL_MULTI_64 GREX_SHFL_MULTI_BIG
#define GREX_SHFL_MULTI_32 GREX_SHFL_MULTI_BIG

// Special-cased 16-bit multi-shuffles using epi16 permutes.
#define GREX_SHFL_MULTI_16(KIND, BITS, IDXBITS) \
  GREX_SHFL_MULTI_BASE(2, KIND, KIND, 16, 16, 8, IDXBITS, _mm256_permutexvar_epi16(vidxs, xtable)) \
  GREX_SHFL_MULTI_BASE(2, KIND, KIND, 16, 32, 16, IDXBITS, \
                       _mm512_permutexvar_epi16(vidxs, xtable)) \
  GREX_SHFL_MULTI_BASE(4, KIND, KIND, 16, 32, 8, IDXBITS, _mm512_permutexvar_epi16(vidxs, xtable))

// Special-cased 8-bit multi-shuffles using byte-wise shuffles.
#define GREX_SHFL_MULTI_8(KIND, BITS, IDXBITS) \
  GREX_SHFL_MULTI_BASE(2, KIND, KIND, 8, 32, 16, IDXBITS, _mm256_shuffle_epi8(xtable, vidxs)) \
  GREX_SHFL_MULTI_BASE(2, KIND, KIND, 8, 64, 32, IDXBITS, \
                       shuffle(u8x64{xtable}, u8x64{vidxs}, index_ub, index_offset).r) \
  GREX_SHFL_MULTI_BASE(4, KIND, KIND, 8, 64, 16, IDXBITS, _mm512_shuffle_epi8(xtable, vidxs))

// Instantiate multi-shuffles for all index sizes for a (KIND, BITS) pair.
#define GREX_SHFL_MULTI(MACRO, KIND, BITS) \
  MACRO(KIND, BITS, 64) \
  MACRO(KIND, BITS, 32) \
  MACRO(KIND, BITS, 16) \
  MACRO(KIND, BITS, 8)

// Resolver for all vector sizes of a given element type.
#define GREX_SHFL_MULTI_RESOLVER(KIND, BITS, SIZE) \
  GREX_SHFL_MULTI(GREX_SHFL_MULTI_##BITS, KIND, BITS)
GREX_FOREACH_TYPE(GREX_SHFL_MULTI_RESOLVER, 128)
#elif GREX_X86_64_LEVEL == 3
// AVX2-only “multi” shuffles (output > native table size).

#define GREX_SHFL_MULTI_VPERMILPD(KIND, BITS, IDXBITS, SIZE) \
  GREX_ALWAYS_INLINE inline VectorFor<KIND##BITS, SIZE> shuffle( \
    VectorFor<KIND##BITS, GREX_DIVIDE(SIZE, 2)> table, VectorFor<u##IDXBITS, SIZE> idxs, \
    AnyIndexTag auto /*index_ub*/, AnyIndexTag auto /*index_offset*/) { \
    const auto idxs64 = shuffle_indices(idxs, index_tag<8>, index_tag<GREX_DIVIDE(BITS, 8)>); \
    const auto xtable = repeat<2>(as<f64>(table)); \
    const auto shuf64 = _mm256_permutevar_pd(xtable.r, shift_left(idxs64, index_tag<1>).r); \
    return as<KIND##BITS>(f64x4{shuf64}); \
  }

#define GREX_SHFL_MULTI_VPERMILPS(KIND, BITS, IDXBITS, SIZE) \
  GREX_ALWAYS_INLINE inline VectorFor<KIND##BITS, SIZE> shuffle( \
    VectorFor<KIND##BITS, GREX_DIVIDE(SIZE, 2)> table, VectorFor<u##IDXBITS, SIZE> idxs, \
    AnyIndexTag auto /*index_ub*/, AnyIndexTag auto /*index_offset*/) { \
    const auto idxs32 = shuffle_indices(idxs, index_tag<4>, index_tag<GREX_DIVIDE(BITS, 8)>).r; \
    const auto shuf32 = _mm256_permutevar_ps(repeat<2>(as<f32>(table)).r, idxs32); \
    return as<KIND##BITS>(f32x8{shuf32}); \
  }

#define GREX_SHFL_MULTI_PSHUFB(KIND, BITS, IDXBITS, SIZE) \
  GREX_ALWAYS_INLINE inline VectorFor<KIND##BITS, SIZE> shuffle( \
    VectorFor<KIND##BITS, GREX_DIVIDE(SIZE, 2)> table, VectorFor<u##IDXBITS, SIZE> idxs, \
    AnyIndexTag auto /*index_ub*/, AnyIndexTag auto /*index_offset*/) { \
    const auto idxs8 = shuffle_indices(idxs, index_tag<1>, index_tag<GREX_DIVIDE(BITS, 8)>).r; \
    const auto shuf8 = _mm256_shuffle_epi8(repeat<2>(as<u8>(table)).r, idxs8); \
    return as<KIND##BITS>(u8x32{shuf8}); \
  }

// Instantiate AVX2 multi-shuffles for all index sizes.
#define GREX_SHFL_MULTI(MACRO, KIND, BITS, SIZE) \
  MACRO(KIND, BITS, 64, SIZE) \
  MACRO(KIND, BITS, 32, SIZE) \
  MACRO(KIND, BITS, 16, SIZE) \
  MACRO(KIND, BITS, 8, SIZE)
GREX_SHFL_MULTI(GREX_SHFL_MULTI_VPERMILPD, f, 64, 4)
GREX_SHFL_MULTI(GREX_SHFL_MULTI_VPERMILPD, i, 64, 4)
GREX_SHFL_MULTI(GREX_SHFL_MULTI_VPERMILPD, u, 64, 4)
GREX_SHFL_MULTI(GREX_SHFL_MULTI_VPERMILPS, f, 32, 8)
GREX_SHFL_MULTI(GREX_SHFL_MULTI_VPERMILPS, i, 32, 8)
GREX_SHFL_MULTI(GREX_SHFL_MULTI_VPERMILPS, u, 32, 8)
GREX_SHFL_MULTI(GREX_SHFL_MULTI_PSHUFB, i, 16, 16)
GREX_SHFL_MULTI(GREX_SHFL_MULTI_PSHUFB, u, 16, 16)
GREX_SHFL_MULTI(GREX_SHFL_MULTI_PSHUFB, i, 8, 32)
GREX_SHFL_MULTI(GREX_SHFL_MULTI_PSHUFB, u, 8, 32)
#endif

// Binary16: delegate to `u16`.
template<Float16Vector TTable, UnsignedIntVector TIdxs>
inline VectorFor<f16, size_of<TIdxs>> shuffle(TTable table, TIdxs idxs, AnyIndexTag auto index_ub,
                                              AnyIndexTag auto index_offset) {
  return as<f16>(shuffle(as<u16>(table), idxs, index_ub, index_offset));
}

// Generic entry point for x86-64-v2+.
//
// The overload set above covers many nice cases (native tables/indices).
// This function resolves the remaining cases by splitting/expanding/merging into forms that are
// handled by the specialized overloads.
template<AnyVector TTable, UnsignedIntVector TIdxs>
GREX_ALWAYS_INLINE inline VectorFor<ValueOf<TTable>, size_of<TIdxs>>
shuffle(TTable table, TIdxs idxs, AnyIndexTag auto index_ub, AnyIndexTag auto index_offset) {
  using Value = ValueOf<TTable>;
  using ValueIndex = UnsignedInt<sizeof(Value)>;
  constexpr std::size_t table_size = size_of<TTable>;
  using Index = ValueOf<TIdxs>;
  constexpr std::size_t index_size = size_of<TIdxs>;
  constexpr std::size_t max_index = std::numeric_limits<Index>::max();

  if constexpr (sizeof(Index) < 8 && table_size > max_index + 1) {
    // Index type cannot address the full table → logically shrink the table.
    // This only occurs in practice for u8 indices.
    return shuffle(shrink<max_index + 1>(table), idxs, index_tag<max_index + 1>, index_offset);
  } else if constexpr (is_supernative<Value, index_size>) {
    // Output vector does not fit in a single native register → split indices and merge results.
    return merge(shuffle(table, get_low(idxs), index_ub, index_offset),
                 shuffle(table, get_high(idxs), index_ub, index_offset));
  }
#if GREX_X86_64_LEVEL >= 4
  else if constexpr (
#if GREX_HAS_AVX512VBMI
    sizeof(Value) * table_size == 128 && !is_supernative<Value, index_size>
#else
    sizeof(Value) > 1 && sizeof(Value) * table_size == 128 && !is_supernative<Value, index_size>
#endif
  ) {
    // 1024-bit table and at most native-sized output → use permutex2var on a 2×native table.
    // 8-bit elements require AVX-512VBMI for direct support.
    const auto xidxs = expand_any(convert(idxs, type_tag<ValueIndex>), index_tag<table_size / 2>);
    return shrink(shuffle(table, xidxs, index_ub, index_offset), index_tag<index_size>);
  }
#endif
  else if constexpr (is_supernative<Value, table_size>) {
    // Table exceeds native width → split table into two halves and blend based on index range.
    const auto lo = shuffle(table.lower, idxs, index_ub, index_offset);
    const auto hi = shuffle(table.upper, idxs, index_ub, index_tag<index_offset + table_size / 2>);
    const auto mask =
      compare_lt(idxs, broadcast(Index{index_offset + table_size / 2}, type_tag<TIdxs>));
    return blend(convert<Value>(mask), hi, lo);
  } else if constexpr (index_size < table_size && is_supernative<Index, table_size>) {
    // Fewer indices than table entries, and expanded indices would exceed native width:
    // convert indices to a smaller integer type.
    return shuffle(table, convert<ValueIndex>(idxs), index_ub,
                   index_tag<index_offset * sizeof(Index) / sizeof(ValueIndex)>);
  } else if constexpr (table_size > index_size && !is_supernative<Index, table_size>) {
    // Table larger than index vector, and indices can be expanded natively:
    // expand indices, then shrink result back.
    return shrink<index_size>(shuffle(table, expand_any<table_size>(idxs), index_ub, index_offset));
  } else if constexpr (is_subnative<Value, table_size> && is_subnative<Index, index_size>) {
    // Both table and indices are sub-native: expand them so that at least
    // one becomes native, perform a native shuffle, then shrink back.
    constexpr auto factor =
      std::min(min_native_size<Value> / table_size, min_native_size<Index> / index_size);
    const auto shuf =
      shuffle(VectorFor<Value, factor * table_size>{table.full},
              VectorFor<Index, factor * index_size>{idxs.full}, index_ub, index_offset);
    return shrink<index_size>(shuf);
  } else if constexpr (is_subnative<Value, table_size> && !is_subnative<Index, index_size>) {
    // Table sub-native, indices native/super-native → operate on the full table.
    return shuffle(table.full, idxs, index_ub, index_offset);
  } else {
    static_assert(false, "Unsupported shuffle!");
    std::unreachable();
  }
}

// Convenience overload: full table range, zero offset.
template<AnyVector TTable, UnsignedIntVector TIdxs>
GREX_ALWAYS_INLINE inline VectorFor<ValueOf<TTable>, size_of<TIdxs>> shuffle(TTable table,
                                                                             TIdxs idxs) {
  return shuffle(table, idxs, index_tag<TTable::size>, index_tag<0>);
}
#else // GREX_X86_64_LEVEL <= 1
// Fallback implementations built from basic shuffles, shifts and blends.

#define GREX_SHFL_64(KIND) \
  GREX_ALWAYS_INLINE inline KIND##64x2 shuffle(KIND##64x2 table, u64x2 idxs, \
                                               AnyIndexTag auto /*index_offset*/) { \
    /* Index bit 0 selects element 0 or 1. */ \
    const auto select0 = \
      bf64x2{.r = compare_eq(shift_left(idxs, index_tag<63>), zeros<u64x2>()).r}; \
    const auto tab = as<f64>(table).r; \
    const auto bc0 = f64x2{.r = _mm_unpacklo_pd(tab, tab)}; \
    const auto bc1 = f64x2{.r = _mm_unpackhi_pd(tab, tab)}; \
    return as<KIND##64>(blend(select0, bc1, bc0)); \
  }

#define GREX_SHFL_32(KIND) \
  GREX_ALWAYS_INLINE inline KIND##32x4 shuffle(KIND##32x4 table, u32x4 idxs, \
                                               AnyIndexTag auto /*index_offset*/) { \
    /* Decode two lowest index bits using masks. */ \
    const auto select0 = \
      bf32x4{.r = compare_eq(shift_left(idxs, index_tag<31>), zeros<u32x4>()).r}; \
    const auto bit1 = shift_right(shift_left(idxs, index_tag<30>), index_tag<31>); \
    const auto select1 = bf32x4{.r = compare_eq(bit1, zeros<u32x4>()).r}; \
\
    const auto tab = as<f32>(table).r; \
    const auto bc0 = f32x4{.r = _mm_shuffle_ps(tab, tab, 0b00000000)}; \
    const auto bc1 = f32x4{.r = _mm_shuffle_ps(tab, tab, 0b01010101)}; \
    const auto bc01 = blend(select0, bc1, bc0); \
    const auto bc2 = f32x4{.r = _mm_shuffle_ps(tab, tab, 0b10101010)}; \
    const auto bc3 = f32x4{.r = _mm_shuffle_ps(tab, tab, 0b11111111)}; \
    const auto bc23 = blend(select0, bc3, bc2); \
    return as<KIND##32>(blend(select1, bc23, bc01)); \
  } \
  GREX_ALWAYS_INLINE inline KIND##32x4 shuffle(SubVector<KIND##32, 2> table, u32x4 idxs, \
                                               AnyIndexTag auto /*index_offset*/) { \
    const auto select0 = \
      bf32x4{.r = compare_eq(shift_left(idxs, index_tag<31>), zeros<u32x4>()).r}; \
    const auto tab = as<f32>(table.full).r; \
    const auto bc0 = f32x4{.r = _mm_shuffle_ps(tab, tab, 0b00000000)}; \
    const auto bc1 = f32x4{.r = _mm_shuffle_ps(tab, tab, 0b01010101)}; \
    return as<KIND##32>(blend(select0, bc1, bc0)); \
  }

#define GREX_SHFL_16(KIND) \
  GREX_ALWAYS_INLINE inline KIND##16x8 shuffle(KIND##16x8 table, u16x8 idxs, \
                                               AnyIndexTag auto /*index_offset*/) { \
    using Vec = NativeVector<KIND##16, 8>; \
    using Msk = NativeMask<KIND##16, 8>; \
\
    /* Decode three lowest index bits. */ \
    const auto select0 = Msk{.r = compare_eq(shift_left(idxs, index_tag<15>), zeros<u16x8>()).r}; \
    const auto bit1 = shift_right(shift_left(idxs, index_tag<14>), index_tag<15>); \
    const auto select1 = Msk{.r = compare_eq(bit1, zeros<u16x8>()).r}; \
    const auto bit2 = shift_right(shift_left(idxs, index_tag<13>), index_tag<15>); \
    const auto select2 = Msk{.r = compare_eq(bit2, zeros<u16x8>()).r}; \
\
    auto duplo = [](__m128i v) { return _mm_unpacklo_epi64(v, v); }; \
    const auto bc0 = Vec{.r = duplo(_mm_shufflelo_epi16(table.r, 0b00000000))}; \
    const auto bc1 = Vec{.r = duplo(_mm_shufflelo_epi16(table.r, 0b01010101))}; \
    const auto bc01 = blend(select0, bc1, bc0); \
    const auto bc2 = Vec{.r = duplo(_mm_shufflelo_epi16(table.r, 0b10101010))}; \
    const auto bc3 = Vec{.r = duplo(_mm_shufflelo_epi16(table.r, 0b11111111))}; \
    const auto bc23 = blend(select0, bc3, bc2); \
    const auto bc0123 = blend(select1, bc23, bc01); \
\
    auto duphi = [](__m128i v) { return _mm_unpackhi_epi64(v, v); }; \
    const auto bc4 = Vec{.r = duphi(_mm_shufflehi_epi16(table.r, 0b00000000))}; \
    const auto bc5 = Vec{.r = duphi(_mm_shufflehi_epi16(table.r, 0b01010101))}; \
    const auto bc45 = blend(select0, bc5, bc4); \
    const auto bc6 = Vec{.r = duphi(_mm_shufflehi_epi16(table.r, 0b10101010))}; \
    const auto bc7 = Vec{.r = duphi(_mm_shufflehi_epi16(table.r, 0b11111111))}; \
    const auto bc67 = blend(select0, bc7, bc6); \
    const auto bc4567 = blend(select1, bc67, bc45); \
\
    return blend(select2, bc4567, bc0123); \
  } \
  GREX_ALWAYS_INLINE inline KIND##16x8 shuffle(SubVector<KIND##16, 4> table, u16x8 idxs, \
                                               AnyIndexTag auto /*index_offset*/) { \
    using Vec = NativeVector<KIND##16, 8>; \
    using Msk = NativeMask<KIND##16, 8>; \
\
    const auto select0 = Msk{.r = compare_eq(shift_left(idxs, index_tag<15>), zeros<u16x8>()).r}; \
    const auto bit1 = shift_right(shift_left(idxs, index_tag<14>), index_tag<15>); \
    const auto select1 = Msk{.r = compare_eq(bit1, zeros<u16x8>()).r}; \
\
    auto duplo = [](__m128i v) { return _mm_unpacklo_epi64(v, v); }; \
    const auto bc0 = Vec{.r = duplo(_mm_shufflelo_epi16(table.full.r, 0b00000000))}; \
    const auto bc1 = Vec{.r = duplo(_mm_shufflelo_epi16(table.full.r, 0b01010101))}; \
    const auto bc01 = blend(select0, bc1, bc0); \
    const auto bc2 = Vec{.r = duplo(_mm_shufflelo_epi16(table.full.r, 0b10101010))}; \
    const auto bc3 = Vec{.r = duplo(_mm_shufflelo_epi16(table.full.r, 0b11111111))}; \
    const auto bc23 = blend(select0, bc3, bc2); \
    return blend(select1, bc23, bc01); \
  } \
  GREX_ALWAYS_INLINE inline KIND##16x8 shuffle(SubVector<KIND##16, 2> table, u16x8 idxs, \
                                               AnyIndexTag auto /*index_offset*/) { \
    using Vec = NativeVector<KIND##16, 8>; \
    using Msk = NativeMask<KIND##16, 8>; \
\
    const auto select0 = Msk{.r = compare_eq(shift_left(idxs, index_tag<15>), zeros<u16x8>()).r}; \
    auto duplo = [](__m128i v) { return _mm_unpacklo_epi64(v, v); }; \
    const auto bc0 = Vec{.r = duplo(_mm_shufflelo_epi16(table.full.r, 0b00000000))}; \
    const auto bc1 = Vec{.r = duplo(_mm_shufflelo_epi16(table.full.r, 0b01010101))}; \
    return blend(select0, bc1, bc0); \
  }

#define GREX_SHFL_8(KIND) \
  GREX_ALWAYS_INLINE inline KIND##8x16 shuffle(KIND##8x16 table, u8x16 idxs, \
                                               AnyIndexTag auto /*index_offset*/) { \
    using Vec = NativeVector<KIND##8, 16>; \
    using Msk = NativeMask<KIND##8, 16>; \
\
    /* Decode four lowest bits of the index to select 1-of-16 elements. */ \
    const auto select0 = Msk{.r = compare_eq(shift_left(idxs, index_tag<7>), zeros<u8x16>()).r}; \
    const auto bit1 = shift_right(shift_left(idxs, index_tag<6>), index_tag<7>); \
    const auto select1 = Msk{.r = compare_eq(bit1, zeros<u8x16>()).r}; \
    const auto bit2 = shift_right(shift_left(idxs, index_tag<5>), index_tag<7>); \
    const auto select2 = Msk{.r = compare_eq(bit2, zeros<u8x16>()).r}; \
    const auto bit3 = shift_right(shift_left(idxs, index_tag<4>), index_tag<7>); \
    const auto select3 = Msk{.r = compare_eq(bit3, zeros<u8x16>()).r}; \
\
    const auto tablo = _mm_unpacklo_epi8(table.r, table.r); \
    const auto tabhi = _mm_unpackhi_epi8(table.r, table.r); \
    auto duplo = [](__m128i v) { return _mm_unpacklo_epi64(v, v); }; \
    auto duphi = [](__m128i v) { return _mm_unpackhi_epi64(v, v); }; \
\
    const auto bc0 = Vec{.r = duplo(_mm_shufflelo_epi16(tablo, 0b00000000))}; \
    const auto bc1 = Vec{.r = duplo(_mm_shufflelo_epi16(tablo, 0b01010101))}; \
    const auto bc01 = blend(select0, bc1, bc0); \
    const auto bc2 = Vec{.r = duplo(_mm_shufflelo_epi16(tablo, 0b10101010))}; \
    const auto bc3 = Vec{.r = duplo(_mm_shufflelo_epi16(tablo, 0b11111111))}; \
    const auto bc23 = blend(select0, bc3, bc2); \
    const auto bc0123 = blend(select1, bc23, bc01); \
\
    const auto bc4 = Vec{.r = duphi(_mm_shufflehi_epi16(tablo, 0b00000000))}; \
    const auto bc5 = Vec{.r = duphi(_mm_shufflehi_epi16(tablo, 0b01010101))}; \
    const auto bc45 = blend(select0, bc5, bc4); \
    const auto bc6 = Vec{.r = duphi(_mm_shufflehi_epi16(tablo, 0b10101010))}; \
    const auto bc7 = Vec{.r = duphi(_mm_shufflehi_epi16(tablo, 0b11111111))}; \
    const auto bc67 = blend(select0, bc7, bc6); \
    const auto bc4567 = blend(select1, bc67, bc45); \
    const auto bc01234567 = blend(select2, bc4567, bc0123); \
\
    const auto bc8 = Vec{.r = duplo(_mm_shufflelo_epi16(tabhi, 0b00000000))}; \
    const auto bc9 = Vec{.r = duplo(_mm_shufflelo_epi16(tabhi, 0b01010101))}; \
    const auto bc89 = blend(select0, bc9, bc8); \
    const auto bca = Vec{.r = duplo(_mm_shufflelo_epi16(tabhi, 0b10101010))}; \
    const auto bcb = Vec{.r = duplo(_mm_shufflelo_epi16(tabhi, 0b11111111))}; \
    const auto bcab = blend(select0, bcb, bca); \
    const auto bc89ab = blend(select1, bcab, bc89); \
\
    const auto bcc = Vec{.r = duphi(_mm_shufflehi_epi16(tabhi, 0b00000000))}; \
    const auto bcd = Vec{.r = duphi(_mm_shufflehi_epi16(tabhi, 0b01010101))}; \
    const auto bccd = blend(select0, bcd, bcc); \
    const auto bce = Vec{.r = duphi(_mm_shufflehi_epi16(tabhi, 0b10101010))}; \
    const auto bcf = Vec{.r = duphi(_mm_shufflehi_epi16(tabhi, 0b11111111))}; \
    const auto bcef = blend(select0, bcf, bce); \
    const auto bccdef = blend(select1, bcef, bccd); \
    const auto bc89abcdef = blend(select2, bccdef, bc89ab); \
\
    return blend(select3, bc89abcdef, bc01234567); \
  } \
  GREX_ALWAYS_INLINE inline KIND##8x16 shuffle(SubVector<KIND##8, 8> table, u8x16 idxs, \
                                               AnyIndexTag auto /*index_offset*/) { \
    using Vec = NativeVector<KIND##8, 16>; \
    using Msk = NativeMask<KIND##8, 16>; \
\
    const auto select0 = Msk{.r = compare_eq(shift_left(idxs, index_tag<7>), zeros<u8x16>()).r}; \
    const auto bit1 = shift_right(shift_left(idxs, index_tag<6>), index_tag<7>); \
    const auto select1 = Msk{.r = compare_eq(bit1, zeros<u8x16>()).r}; \
    const auto bit2 = shift_right(shift_left(idxs, index_tag<5>), index_tag<7>); \
    const auto select2 = Msk{.r = compare_eq(bit2, zeros<u8x16>()).r}; \
    const auto tablo = _mm_unpacklo_epi8(table.full.r, table.full.r); \
\
    auto duplo = [](__m128i v) { return _mm_unpacklo_epi64(v, v); }; \
    const auto bc0 = Vec{.r = duplo(_mm_shufflelo_epi16(tablo, 0b00000000))}; \
    const auto bc1 = Vec{.r = duplo(_mm_shufflelo_epi16(tablo, 0b01010101))}; \
    const auto bc01 = blend(select0, bc1, bc0); \
    const auto bc2 = Vec{.r = duplo(_mm_shufflelo_epi16(tablo, 0b10101010))}; \
    const auto bc3 = Vec{.r = duplo(_mm_shufflelo_epi16(tablo, 0b11111111))}; \
    const auto bc23 = blend(select0, bc3, bc2); \
    const auto bc0123 = blend(select1, bc23, bc01); \
\
    auto duphi = [](__m128i v) { return _mm_unpackhi_epi64(v, v); }; \
    const auto bc4 = Vec{.r = duphi(_mm_shufflehi_epi16(tablo, 0b00000000))}; \
    const auto bc5 = Vec{.r = duphi(_mm_shufflehi_epi16(tablo, 0b01010101))}; \
    const auto bc45 = blend(select0, bc5, bc4); \
    const auto bc6 = Vec{.r = duphi(_mm_shufflehi_epi16(tablo, 0b10101010))}; \
    const auto bc7 = Vec{.r = duphi(_mm_shufflehi_epi16(tablo, 0b11111111))}; \
    const auto bc67 = blend(select0, bc7, bc6); \
    const auto bc4567 = blend(select1, bc67, bc45); \
    return blend(select2, bc4567, bc0123); \
  } \
  GREX_ALWAYS_INLINE inline KIND##8x16 shuffle(SubVector<KIND##8, 4> table, u8x16 idxs, \
                                               AnyIndexTag auto /*index_offset*/) { \
    using Vec = NativeVector<KIND##8, 16>; \
    using Msk = NativeMask<KIND##8, 16>; \
\
    const auto select0 = Msk{.r = compare_eq(shift_left(idxs, index_tag<7>), zeros<u8x16>()).r}; \
    const auto bit1 = shift_right(shift_left(idxs, index_tag<6>), index_tag<7>); \
    const auto select1 = Msk{.r = compare_eq(bit1, zeros<u8x16>()).r}; \
\
    const auto tablo = _mm_unpacklo_epi8(table.full.r, table.full.r); \
    auto duplo = [](__m128i v) { return _mm_unpacklo_epi64(v, v); }; \
    const auto bc0 = Vec{.r = duplo(_mm_shufflelo_epi16(tablo, 0b00000000))}; \
    const auto bc1 = Vec{.r = duplo(_mm_shufflelo_epi16(tablo, 0b01010101))}; \
    const auto bc01 = blend(select0, bc1, bc0); \
    const auto bc2 = Vec{.r = duplo(_mm_shufflelo_epi16(tablo, 0b10101010))}; \
    const auto bc3 = Vec{.r = duplo(_mm_shufflelo_epi16(tablo, 0b11111111))}; \
    const auto bc23 = blend(select0, bc3, bc2); \
    return blend(select1, bc23, bc01); \
  } \
  GREX_ALWAYS_INLINE inline KIND##8x16 shuffle(SubVector<KIND##8, 2> table, u8x16 idxs, \
                                               AnyIndexTag auto /*index_offset*/) { \
    using Vec = NativeVector<KIND##8, 16>; \
    using Msk = NativeMask<KIND##8, 16>; \
\
    const auto select0 = Msk{.r = compare_eq(shift_left(idxs, index_tag<7>), zeros<u8x16>()).r}; \
    const auto tablo = _mm_unpacklo_epi8(table.full.r, table.full.r); \
    auto duplo = [](__m128i v) { return _mm_unpacklo_epi64(v, v); }; \
    const auto bc0 = Vec{.r = duplo(_mm_shufflelo_epi16(tablo, 0b00000000))}; \
    const auto bc1 = Vec{.r = duplo(_mm_shufflelo_epi16(tablo, 0b01010101))}; \
    return blend(select0, bc1, bc0); \
  }

GREX_SHFL_64(f)
GREX_SHFL_64(i)
GREX_SHFL_64(u)
GREX_SHFL_32(f)
GREX_SHFL_32(i)
GREX_SHFL_32(u)
GREX_SHFL_16(i)
GREX_SHFL_16(u)
GREX_SHFL_8(i)
GREX_SHFL_8(u)

// Binary16: delegate to `u16`.
template<Float16Vector TTable, UnsignedIntVector TIdxs>
inline VectorFor<f16, size_of<TIdxs>> shuffle(TTable table, TIdxs idxs,
                                              AnyIndexTag auto index_offset) {
  return as<f16>(shuffle(as<u16>(table), idxs, index_offset));
}

// Generic entry point for x86-64-v1.
//
// Similar strategy as the x86-64-v2+ variant, but with fewer cases because only very small vector
// sizes are natively supported.
template<AnyVector TTable, UnsignedIntVector TIdxs>
GREX_ALWAYS_INLINE inline VectorFor<ValueOf<TTable>, size_of<TIdxs>>
shuffle(TTable table, TIdxs idxs, AnyIndexTag auto index_offset) {
  using Value = ValueOf<TTable>;
  using ValueIndex = UnsignedInt<sizeof(Value)>;
  constexpr std::size_t table_size = size_of<TTable>;
  using Index = ValueOf<TIdxs>;
  constexpr std::size_t index_size = size_of<TIdxs>;
  constexpr std::size_t max_index = std::numeric_limits<Index>::max();

  if constexpr (sizeof(Index) < 8 && table_size > max_index + 1) {
    // Index type cannot address the full table → logically shrink the table.
    // This only occurs in practice for u8 indices.
    return shuffle(shrink<max_index + 1>(table), idxs, index_offset);
  } else if constexpr (is_supernative<Value, index_size>) {
    // Output vector does not fit in a single native register → split indices and merge results.
    return merge(shuffle(table, get_low(idxs), index_offset),
                 shuffle(table, get_high(idxs), index_offset));
  } else if constexpr (is_supernative<Value, table_size>) {
    // Table exceeds native width → split table into two halves and blend based on index range.
    const auto lo = shuffle(table.lower, idxs, index_offset);
    const auto hi = shuffle(table.upper, idxs, index_tag<index_offset + table_size / 2>);
    const auto mask =
      compare_lt(idxs, broadcast(Index{index_offset + table_size / 2}, type_tag<TIdxs>));
    return blend(convert<Value>(mask), hi, lo);
  } else if constexpr (!std::same_as<Index, ValueIndex>) {
    // Index type differs from element size → convert indices to value-sized integers.
    return shuffle(table, convert<ValueIndex>(idxs),
                   index_tag<index_offset * sizeof(Index) / sizeof(ValueIndex)>);
  } else if constexpr (is_subnative<Index, index_size>) {
    // Indices are sub-native → operate on expanded full view and shrink result.
    return shrink<index_size>(shuffle(table, idxs.full, index_offset));
  } else {
    static_assert(false, "Unsupported shuffle!");
    std::unreachable();
  }
}

// Convenience overload: zero offset.
template<AnyVector TTable, UnsignedIntVector TIdxs>
GREX_ALWAYS_INLINE inline VectorFor<ValueOf<TTable>, size_of<TIdxs>> shuffle(TTable table,
                                                                             TIdxs idxs) {
  return shuffle(table, idxs, index_tag<0>);
}
#endif
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_SHUFFLE_HPP
