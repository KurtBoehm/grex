// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_NEON_OPERATIONS_SHUFFLE_HPP
#define INCLUDE_GREX_BACKEND_NEON_OPERATIONS_SHUFFLE_HPP

#include <array>
#include <concepts>
#include <cstddef>
#include <limits>
#include <utility>

#include <arm_neon.h>

#include "grex/backend/base.hpp"
#include "grex/backend/choosers.hpp"
#include "grex/backend/neon/operations/blend.hpp"
#include "grex/backend/neon/operations/compare.hpp"
#include "grex/backend/neon/operations/convert.hpp"
#include "grex/backend/neon/operations/merge.hpp"
#include "grex/backend/neon/operations/reinterpret.hpp"
#include "grex/backend/neon/operations/shrink.hpp"
#include "grex/backend/neon/operations/split.hpp"
#include "grex/backend/neon/types.hpp"
#include "grex/backend/shared/operations/set.hpp"
#include "grex/base.hpp"

namespace grex::backend {
// 1-byte elements, u8×16 indices: indices are already byte indices.
inline u8x16 shuffle_indices(u8x16 idxs, IndexTag<1> /*value_bytes*/) {
  return idxs;
}

// 2-byte elements, u8×8 indices: replicate each 1-byte index to two consecutive bytes,
// multiply by 2, then add the per-element offsets [0, 1].
inline u8x16 shuffle_indices(SubVector<u8, 8> idxs, IndexTag<2> /*value_bytes*/) {
  const auto zip = vzip1q_u8(idxs.full.r, idxs.full.r);
  const auto shift = vaddq_u8(zip, zip);
  return {.r = vorrq_u8(shift, as<u8>(vdupq_n_u16(0x0100)))};
}

// 4-byte elements, u8×4 indices: multiply by 4, replicate each 1-byte index to four
// adjacent bytes, then add the per-element offsets [0, 1, 2, 3].
inline u8x16 shuffle_indices(SubVector<u8, 4> idxs, IndexTag<4> /*value_bytes*/) {
  constexpr std::array<u8, 16> shuf_arr{0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};
  const auto shift = vshlq_n_u8(idxs.full.r, 2);
  const auto shuf = vqtbl1q_u8(shift, vld1q_u8(shuf_arr.data()));
  // OR with base byte offsets [0, 1, 2, 3] repeated.
  return {.r = vorrq_u8(shuf, as<u8>(vdupq_n_u32(0x03020100)))};
}

// 8-byte elements, u8×2 indices: multiply by 8, replicate each 1-byte index to eight
// adjacent bytes, then add the per-element offsets [0, 1, 2, 3, 4, 5, 6, 7].
inline u8x16 shuffle_indices(SubVector<u8, 2> idxs, IndexTag<8> /*value_bytes*/) {
  constexpr std::array<u8, 16> shuf_arr{0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1};
  const auto shift = vshlq_n_u8(idxs.full.r, 3);
  const auto shuf = vqtbl1q_u8(shift, vld1q_u8(shuf_arr.data()));
  // OR with base byte offsets [0, 1, 2, 3, 4, 5, 6, 7] repeated.
  return {.r = vorrq_u8(shuf, as<u8>(vdupq_n_u64(0x0706050403020100)))};
}

// 1-byte elements, u16×16 indices: take the low byte of each index.
inline u8x16 shuffle_indices(VectorFor<u16, 16> idxs, IndexTag<1> /*value_bytes*/) {
  return {.r = vuzp1q_u8(as<u8>(idxs.lower.r), as<u8>(idxs.upper.r))};
}

// 1-byte elements, u16×8 indices: take the low byte of each index.
inline SubVector<u8, 8> shuffle_indices(u16x8 idxs, IndexTag<1> /*value_bytes*/) {
  return SubVector<u8, 8>{{.r = vuzp1q_u8(as<u8>(idxs.r), as<u8>(idxs.r))}};
}

// 2-byte elements, u16×8 indices: replicate the low byte into the high byte, multiply by 2,
// then add the per-element offsets [0, 1].
inline u8x16 shuffle_indices(u16x8 idxs, IndexTag<2> /*value_bytes*/) {
  const auto trn = vtrn1q_u8(as<u8>(idxs.r), as<u8>(idxs.r));
  const auto shift = vaddq_u8(trn, trn);
  return {.r = vorrq_u8(shift, as<u8>(vdupq_n_u16(0x0100)))};
}

// 4-byte elements, u16×4 indices: widen to u32, multiply by u32(u8×4{4, 4, 4, 4}) to obtain
// byte indices in one step, then add the per-element offsets [0, 1, 2, 3].
inline u8x16 shuffle_indices(SubVector<u16, 4> idxs, IndexTag<4> /*value_bytes*/) {
  const auto idxs32 = vmovl_u16(vget_low_u16(idxs.full.r));
  const auto mul = as<u8>(vmulq_u32(idxs32, vdupq_n_u32(0x04040404)));
  return {.r = vorrq_u8(mul, as<u8>(vdupq_n_u32(0x03020100)))};
}

// 8-byte elements, u16×2 indices: multiply by 8, replicate each low byte across 8 bytes,
// then add the per-element offsets [0, 1, 2, 3, 4, 5, 6, 7].
inline u8x16 shuffle_indices(SubVector<u16, 2> idxs, IndexTag<8> /*value_bytes*/) {
  constexpr std::array<u8, 16> shuf_arr{0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2};
  const auto shift = vshlq_n_u8(as<u8>(idxs.full.r), 3);
  const auto shuf = vqtbl1q_u8(shift, vld1q_u8(shuf_arr.data()));
  return {.r = vorrq_u8(shuf, as<u8>(vdupq_n_u64(0x0706050403020100)))};
}

// 1-byte elements, u32×16 indices: copy the lower 16 bits of each index into the upper 16 bits
// and delegate to the u16 path.
inline u8x16 shuffle_indices(VectorFor<u32, 16> idxs, IndexTag<1> value_bytes) {
  const auto idxs16 = as<u16>(idxs);
  const VectorFor<u16, 16> uzp{
    .lower = {.r = vuzp1q_u16(idxs16.lower.lower.r, idxs16.lower.upper.r)},
    .upper = {.r = vuzp1q_u16(idxs16.upper.lower.r, idxs16.upper.upper.r)},
  };
  return shuffle_indices(uzp, value_bytes);
}

// 1-byte elements, u32×8 indices: copy the lower 16 bits of each index into the upper 16 bits
// and delegate to the u16 path.
inline SubVector<u8, 8> shuffle_indices(VectorFor<u32, 8> idxs, IndexTag<1> value_bytes) {
  const auto uzp = vuzp1q_u16(as<u16>(idxs.lower.r), as<u16>(idxs.upper.r));
  return shuffle_indices(u16x8{.r = uzp}, value_bytes);
}

// 1-byte elements, u32×4 indices: copy the lower 16 bits of each index into the upper 16 bits
// and delegate to the u16 path.
inline SubVector<u8, 4> shuffle_indices(u32x4 idxs, IndexTag<1> value_bytes) {
  const auto uzp = vuzp1q_u16(as<u16>(idxs.r), as<u16>(idxs.r));
  return SubVector<u8, 4>{shuffle_indices(u16x8{.r = uzp}, value_bytes).full};
}

// 2-byte elements, u32×8 indices: copy the lower 16 bits of each index into the upper 16 bits
// and delegate to the u16 path.
inline u8x16 shuffle_indices(VectorFor<u32, 8> idxs, IndexTag<2> value_bytes) {
  const auto uzp = vuzp1q_u16(as<u16>(idxs.lower.r), as<u16>(idxs.upper.r));
  return shuffle_indices(u16x8{.r = uzp}, value_bytes);
}

// 2-byte elements, u32×4 indices: copy the lower 16 bits of each index into the upper 16 bits
// and delegate to the u16 path.
inline SubVector<u8, 8> shuffle_indices(u32x4 idxs, IndexTag<2> value_bytes) {
  const auto uzp = vuzp1q_u16(as<u16>(idxs.r), as<u16>(idxs.r));
  return SubVector<u8, 8>{shuffle_indices(u16x8{.r = uzp}, value_bytes)};
}

// 4-byte elements, u32×4 indices: multiply by u32(u8×4{4, 4, 4, 4}) to obtain byte indices,
// then add the per-element offsets [0, 1, 2, 3].
inline u8x16 shuffle_indices(u32x4 idxs, IndexTag<4> /*value_bytes*/) {
  const auto mul = as<u8>(vmulq_u32(idxs.r, vdupq_n_u32(0x04040404)));
  return {.r = vorrq_u8(mul, as<u8>(vdupq_n_u32(0x03020100)))};
}

// 8-byte elements, u32×2 indices: multiply by 8, replicate each low byte across 8 bytes,
// then add the per-element offsets [0, 1, 2, 3, 4, 5, 6, 7].
inline u8x16 shuffle_indices(SubVector<u32, 2> idxs, IndexTag<8> /*value_bytes*/) {
  constexpr std::array<u8, 16> shuf_arr{0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4};
  const auto shift = vshlq_n_u8(as<u8>(idxs.full.r), 3);
  const auto shuf = vqtbl1q_u8(shift, vld1q_u8(shuf_arr.data()));
  return {.r = vorrq_u8(shuf, as<u8>(vdupq_n_u64(0x0706050403020100)))};
}

// Replicate the lower 32 bits of each 64-bit index lane into the upper 32 bits.
inline SubVector<u32, 2> compress64(u64x2 idxs) {
  return SubVector<u32, 2>{{.r = vuzp1q_u32(as<u32>(idxs.r), as<u32>(idxs.r))}};
}

// Replicate the lower 32 bits of each 64-bit index lane into the upper 32 bits.
inline u32x4 compress64(VectorFor<u64, 4> idxs) {
  return {.r = vuzp1q_u32(as<u32>(idxs.lower.r), as<u32>(idxs.upper.r))};
}

// Super-native variants: recursively compress each half.
template<AnySuperNativeVector TVec>
inline VectorFor<u32, size_of<TVec>> compress64(TVec v) {
  return {.lower = compress64(v.lower), .upper = compress64(v.upper)};
}

// All u64 index variants reuse the u32 index path via compress64.

inline u8x16 shuffle_indices(VectorFor<u64, 16> idxs, IndexTag<1> value_bytes) {
  return shuffle_indices(compress64(idxs), value_bytes);
}
inline SubVector<u8, 8> shuffle_indices(VectorFor<u64, 8> idxs, IndexTag<1> value_bytes) {
  return shuffle_indices(compress64(idxs), value_bytes);
}
inline SubVector<u8, 4> shuffle_indices(VectorFor<u64, 4> idxs, IndexTag<1> value_bytes) {
  return shuffle_indices(compress64(idxs), value_bytes);
}
inline SubVector<u8, 2> shuffle_indices(u64x2 idxs, IndexTag<1> value_bytes) {
  return SubVector<u8, 2>{shuffle_indices(compress64(idxs).full, value_bytes).full};
}

inline u8x16 shuffle_indices(VectorFor<u64, 8> idxs, IndexTag<2> value_bytes) {
  return shuffle_indices(compress64(idxs), value_bytes);
}
inline SubVector<u8, 8> shuffle_indices(VectorFor<u64, 4> idxs, IndexTag<2> value_bytes) {
  return shuffle_indices(compress64(idxs), value_bytes);
}
inline SubVector<u8, 4> shuffle_indices(u64x2 idxs, IndexTag<2> value_bytes) {
  return SubVector<u8, 4>{shuffle_indices(compress64(idxs).full, value_bytes).full};
}

inline u8x16 shuffle_indices(VectorFor<u64, 4> idxs, IndexTag<4> value_bytes) {
  return shuffle_indices(compress64(idxs), value_bytes);
}
inline SubVector<u8, 8> shuffle_indices(u64x2 idxs, IndexTag<4> value_bytes) {
  return SubVector<u8, 8>{shuffle_indices(compress64(idxs).full, value_bytes)};
}

// 8-byte elements, u64×2 indices: multiply by 8, replicate each low byte across the
// corresponding 8 bytes, then add the per-element offsets [0, 1, 2, 3, 4, 5, 6, 7].
inline u8x16 shuffle_indices(u64x2 idxs, IndexTag<8> /*value_bytes*/) {
  constexpr std::array<u8, 16> shuf_arr{0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8};
  const auto shift = vshlq_n_u8(as<u8>(idxs.r), 3);
  const auto shuf = vqtbl1q_u8(shift, vld1q_u8(shuf_arr.data()));
  return {.r = vorrq_u8(shuf, as<u8>(vdupq_n_u64(0x0706050403020100)))};
}

// Sub-native indices: expand the index vector to twice its width, compute byte indices,
// then shrink back to the original vector width.
template<Vectorizable T, std::size_t tSize, std::size_t tValueBytes>
inline VectorFor<u8, tSize * tValueBytes> shuffle_indices(SubVector<T, tSize> idxs,
                                                          IndexTag<tValueBytes> value_bytes) {
  return shrink<tSize * tValueBytes>(shuffle_indices(expand_any<2 * tSize>(idxs), value_bytes));
}

/////////////////////////////
// Neon 8-bit tbl shuffles //
/////////////////////////////

// Table size 16: vqtbl1q_u8.
// If logical index range (index_ub) exceeds table size, mask indices down to [0, 15].
inline u8x16 shuffle(u8x16 table, u8x16 idxs, AnyIndexTag auto index_ub,
                     AnyIndexTag auto /*index_offset*/) {
  uint8x16_t vidxs = idxs.r;
  if constexpr (index_ub > 16) {
    vidxs = vandq_u8(vidxs, vdupq_n_u8(0x0F));
  }
  return {.r = vqtbl1q_u8(table.r, vidxs)};
}

// Table size 32: vqtbl2q_u8.
// If logical index range (index_ub) exceeds table size, mask indices down to [0, 31].
inline u8x16 shuffle(VectorFor<u8, 32> table, u8x16 idxs, AnyIndexTag auto index_ub,
                     AnyIndexTag auto /*index_offset*/) {
  uint8x16_t vidxs = idxs.r;
  if constexpr (index_ub > 32) {
    vidxs = vandq_u8(vidxs, vdupq_n_u8(0x1F));
  }
  return {.r = vqtbl2q_u8(uint8x16x2_t{table.lower.r, table.upper.r}, vidxs)};
}

// Table size 64: vqtbl4q_u8.
// If logical index range (index_ub) exceeds table size, mask indices down to [0, 63].
inline u8x16 shuffle(VectorFor<u8, 64> table, u8x16 idxs, AnyIndexTag auto index_ub,
                     AnyIndexTag auto /*index_offset*/) {
  uint8x16_t vidxs = idxs.r;
  if constexpr (index_ub > 64) {
    vidxs = vandq_u8(vidxs, vdupq_n_u8(0x3F));
  }
  const uint8x16x4_t vtable{
    table.lower.lower.r,
    table.lower.upper.r,
    table.upper.lower.r,
    table.upper.upper.r,
  };
  return {.r = vqtbl4q_u8(vtable, vidxs)};
}

/////////////////////////////////////
// Generic typed shuffle front-end //
/////////////////////////////////////

// TTable:       table of values to select from
// TIdxs:        unsigned integer indices into the table
// index_ub:     compile-time upper bound on idx values (exclusive, in elements)
// index_offset: base offset that has already been applied to the table (in elements)
template<AnyVector TTable, UnsignedIntVector TIdxs>
inline VectorFor<ValueOf<TTable>, size_of<TIdxs>>
shuffle(TTable table, TIdxs idxs, AnyIndexTag auto index_ub, AnyIndexTag auto index_offset) {
  using Value = ValueOf<TTable>;
  constexpr std::size_t table_size = size_of<TTable>;
  using Index = ValueOf<TIdxs>;
  constexpr std::size_t index_size = size_of<TIdxs>;
  constexpr std::size_t max_index = std::numeric_limits<Index>::max();

  if constexpr (sizeof(Index) < 8 && table_size > max_index + 1) {
    // Index type cannot address the full table → logically shrink the table.
    // Realistically, this only occurs for u8 indices.
    return shuffle(shrink<max_index + 1>(table), idxs, index_tag<max_index + 1>, index_offset);
  } else if constexpr (is_supernative<Value, index_size>) {
    // Result vector spans multiple native registers → process low/high halves separately.
    return merge(shuffle(table, get_low(idxs), index_ub, index_offset),
                 shuffle(table, get_high(idxs), index_ub, index_offset));
  } else if constexpr (!std::same_as<Value, u8> || !std::same_as<Index, u8>) {
    // Non-byte element and/or index type: implement as a byte-wise shuffle.
    // Expand element indices to byte indices, shuffle the u8 table, then cast back.
    const auto idxs8 = shuffle_indices(idxs, index_tag<sizeof(Value)>);
    const auto shuf = shuffle(as<u8>(table), idxs8, index_tag<index_ub * sizeof(Value)>,
                              index_tag<index_offset * sizeof(Value)>);
    return as<Value>(shuf);
  } else if constexpr (index_size < 16) {
    // Sub-native index vector → widen to full native width, then shrink result.
    return shrink<index_size>(shuffle(table, idxs.full, index_ub, index_offset));
  } else if constexpr (table_size < 16) {
    // Sub-native table → widen table to full native width.
    return shuffle(table.full, idxs, index_ub, index_offset);
  } else if constexpr (table_size > 64) {
    // Table too large for a single tbl (max 64 bytes) → split into two halves.
    const auto lo = shuffle(table.lower, idxs, index_ub, index_offset);
    const auto hi = shuffle(table.upper, idxs, index_ub, index_tag<index_offset + table_size / 2>);
    const auto mask = compare_lt(idxs, broadcast<TIdxs>(Index{index_offset + table_size / 2}));
    // Select from low or upper half depending on index range.
    return blend(convert<Value>(mask), hi, lo);
  } else {
    static_assert(false, "Unsupported shuffle!");
    std::unreachable();
  }
}

// Convenience overload: full table range, zero base offset.
template<AnyVector TTable, AnyVector TIdxs>
inline VectorFor<typename TTable::Value, TIdxs::size> shuffle(TTable table, TIdxs idxs) {
  return shuffle(table, idxs, index_tag<TTable::size>, index_tag<0>);
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_SHUFFLE_HPP
