// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_NEON_OPERATIONS_SHUFFLE_HPP
#define INCLUDE_GREX_BACKEND_NEON_OPERATIONS_SHUFFLE_HPP

#include <array>
#include <cstddef>

#include <arm_neon.h>

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

namespace grex::backend {
inline u8x16 shuffle_indices(u8x16 idxs, IndexTag<1> /*value_bytes*/) {
  return idxs;
}
inline u8x16 shuffle_indices(SubVector<u8, 8, 16> idxs, IndexTag<2> /*value_bytes*/) {
  const auto zip = vzip1q_u8(idxs.full.r, idxs.full.r);
  const auto shift = vaddq_u8(zip, zip);
  return {.r = vorrq_u8(shift, as<u8>(vdupq_n_u16(0x0100)))};
}
inline u8x16 shuffle_indices(SubVector<u8, 4, 16> idxs, IndexTag<4> /*value_bytes*/) {
  constexpr std::array<u8, 16> shuf_arr{0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};
  const auto shift = vshlq_n_u8(idxs.full.r, 2);
  const auto shuf = vqtbl1q_u8(shift, vld1q_u8(shuf_arr.data()));
  return {.r = vorrq_u8(shuf, as<u8>(vdupq_n_u32(0x03020100)))};
}
inline u8x16 shuffle_indices(SubVector<u8, 2, 16> idxs, IndexTag<8> /*value_bytes*/) {
  constexpr std::array<u8, 16> shuf_arr{0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1};
  const auto shift = vshlq_n_u8(idxs.full.r, 3);
  const auto shuf = vqtbl1q_u8(shift, vld1q_u8(shuf_arr.data()));
  return {.r = vorrq_u8(shuf, as<u8>(vdupq_n_u64(0x0706050403020100)))};
}

inline u8x16 shuffle_indices(VectorFor<u16, 16> idxs, IndexTag<1> /*value_bytes*/) {
  return {.r = vuzp1q_u8(as<u8>(idxs.lower.r), as<u8>(idxs.upper.r))};
}
inline SubVector<u8, 8, 16> shuffle_indices(u16x8 idxs, IndexTag<1> /*value_bytes*/) {
  return SubVector<u8, 8, 16>{{.r = vuzp1q_u8(as<u8>(idxs.r), as<u8>(idxs.r))}};
}
inline u8x16 shuffle_indices(u16x8 idxs, IndexTag<2> /*value_bytes*/) {
  const auto trn = vtrn1q_u8(as<u8>(idxs.r), as<u8>(idxs.r));
  const auto shift = vaddq_u8(trn, trn);
  return {.r = vorrq_u8(shift, as<u8>(vdupq_n_u16(0x0100)))};
}
inline u8x16 shuffle_indices(SubVector<u16, 4, 8> idxs, IndexTag<4> /*value_bytes*/) {
  const auto idxs32 = vmovl_u16(vget_low_u16(idxs.full.r));
  const auto mul = as<u8>(vmulq_u32(idxs32, vdupq_n_u32(0x04040404)));
  return {.r = vorrq_u8(mul, as<u8>(vdupq_n_u32(0x03020100)))};
}
inline u8x16 shuffle_indices(SubVector<u16, 2, 8> idxs, IndexTag<8> /*value_bytes*/) {
  constexpr std::array<u8, 16> shuf_arr{0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2};
  const auto shift = vshlq_n_u8(as<u8>(idxs.full.r), 3);
  const auto shuf = vqtbl1q_u8(shift, vld1q_u8(shuf_arr.data()));
  return {.r = vorrq_u8(shuf, as<u8>(vdupq_n_u64(0x0706050403020100)))};
}

inline u8x16 shuffle_indices(VectorFor<u32, 16> idxs, IndexTag<1> value_bytes) {
  const auto idxs16 = as<u16>(idxs);
  const VectorFor<u16, 16> uzp{
    .lower = {.r = vuzp1q_u16(idxs16.lower.lower.r, idxs16.lower.upper.r)},
    .upper = {.r = vuzp1q_u16(idxs16.upper.lower.r, idxs16.upper.upper.r)},
  };
  return shuffle_indices(uzp, value_bytes);
}
inline SubVector<u8, 8, 16> shuffle_indices(VectorFor<u32, 8> idxs, IndexTag<1> value_bytes) {
  const auto uzp = vuzp1q_u16(as<u16>(idxs.lower.r), as<u16>(idxs.upper.r));
  return shuffle_indices(u16x8{.r = uzp}, value_bytes);
}
inline SubVector<u8, 4, 16> shuffle_indices(u32x4 idxs, IndexTag<1> value_bytes) {
  const auto uzp = vuzp1q_u16(as<u16>(idxs.r), as<u16>(idxs.r));
  return SubVector<u8, 4, 16>{shuffle_indices(u16x8{.r = uzp}, value_bytes).full};
}
inline u8x16 shuffle_indices(VectorFor<u32, 8> idxs, IndexTag<2> value_bytes) {
  const auto uzp = vuzp1q_u16(as<u16>(idxs.lower.r), as<u16>(idxs.upper.r));
  return shuffle_indices(u16x8{.r = uzp}, value_bytes);
}
inline SubVector<u8, 8, 16> shuffle_indices(u32x4 idxs, IndexTag<2> value_bytes) {
  const auto uzp = vuzp1q_u16(as<u16>(idxs.r), as<u16>(idxs.r));
  return SubVector<u8, 8, 16>{shuffle_indices(u16x8{.r = uzp}, value_bytes)};
}
inline u8x16 shuffle_indices(u32x4 idxs, IndexTag<4> /*value_bytes*/) {
  const auto mul = as<u8>(vmulq_u32(idxs.r, vdupq_n_u32(0x04040404)));
  return {.r = vorrq_u8(mul, as<u8>(vdupq_n_u32(0x03020100)))};
}
inline u8x16 shuffle_indices(SubVector<u32, 2, 4> idxs, IndexTag<8> /*value_bytes*/) {
  constexpr std::array<u8, 16> shuf_arr{0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4};
  const auto shift = vshlq_n_u8(as<u8>(idxs.full.r), 3);
  const auto shuf = vqtbl1q_u8(shift, vld1q_u8(shuf_arr.data()));
  return {.r = vorrq_u8(shuf, as<u8>(vdupq_n_u64(0x0706050403020100)))};
}

inline SubVector<u32, 2, 4> compress64(u64x2 idxs) {
  return SubVector<u32, 2, 4>{{.r = vuzp1q_u32(as<u32>(idxs.r), as<u32>(idxs.r))}};
}
inline u32x4 compress64(VectorFor<u64, 4> idxs) {
  return {.r = vuzp1q_u32(as<u32>(idxs.lower.r), as<u32>(idxs.upper.r))};
}
template<AnySuperNativeVector TVec>
inline VectorFor<u32, TVec::size> compress64(TVec v) {
  return {.lower = compress64(v.lower), .upper = compress64(v.upper)};
}

inline u8x16 shuffle_indices(VectorFor<u64, 16> idxs, IndexTag<1> value_bytes) {
  return shuffle_indices(compress64(idxs), value_bytes);
}
inline SubVector<u8, 8, 16> shuffle_indices(VectorFor<u64, 8> idxs, IndexTag<1> value_bytes) {
  return shuffle_indices(compress64(idxs), value_bytes);
}
inline SubVector<u8, 4, 16> shuffle_indices(VectorFor<u64, 4> idxs, IndexTag<1> value_bytes) {
  return shuffle_indices(compress64(idxs), value_bytes);
}
inline SubVector<u8, 2, 16> shuffle_indices(u64x2 idxs, IndexTag<1> value_bytes) {
  return SubVector<u8, 2, 16>{shuffle_indices(compress64(idxs).full, value_bytes).full};
}

inline u8x16 shuffle_indices(VectorFor<u64, 8> idxs, IndexTag<2> value_bytes) {
  return shuffle_indices(compress64(idxs), value_bytes);
}
inline SubVector<u8, 8, 16> shuffle_indices(VectorFor<u64, 4> idxs, IndexTag<2> value_bytes) {
  return shuffle_indices(compress64(idxs), value_bytes);
}
inline SubVector<u8, 4, 16> shuffle_indices(u64x2 idxs, IndexTag<2> value_bytes) {
  return SubVector<u8, 4, 16>{shuffle_indices(compress64(idxs).full, value_bytes).full};
}

inline u8x16 shuffle_indices(VectorFor<u64, 4> idxs, IndexTag<4> value_bytes) {
  return shuffle_indices(compress64(idxs), value_bytes);
}
inline SubVector<u8, 8, 16> shuffle_indices(u64x2 idxs, IndexTag<4> value_bytes) {
  return SubVector<u8, 8, 16>{shuffle_indices(compress64(idxs).full, value_bytes)};
}

inline u8x16 shuffle_indices(u64x2 idxs, IndexTag<8> /*value_bytes*/) {
  constexpr std::array<u8, 16> shuf_arr{0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8};
  const auto shift = vshlq_n_u8(as<u8>(idxs.r), 3);
  const auto shuf = vqtbl1q_u8(shift, vld1q_u8(shuf_arr.data()));
  return {.r = vorrq_u8(shuf, as<u8>(vdupq_n_u64(0x0706050403020100)))};
}

template<Vectorizable T, std::size_t tPart, std::size_t tSize, std::size_t tValueBytes>
inline VectorFor<u8, tPart * tValueBytes> shuffle_indices(SubVector<T, tPart, tSize> idxs,
                                                          IndexTag<tValueBytes> value_bytes) {
  return shrink<tPart * tValueBytes>(shuffle_indices(expand_any<2 * tPart>(idxs), value_bytes));
}

inline u8x16 shuffle(u8x16 table, u8x16 idxs, AnyIndexTag auto index_ub,
                     AnyIndexTag auto /*index_offset*/) {
  uint8x16_t vidxs = idxs.r;
  if constexpr (index_ub > 16) {
    vidxs = vandq_u8(vidxs, vdupq_n_u8(0x0F));
  }
  return {.r = vqtbl1q_u8(table.r, vidxs)};
}
inline u8x16 shuffle(VectorFor<u8, 32> table, u8x16 idxs, AnyIndexTag auto index_ub,
                     AnyIndexTag auto /*index_offset*/) {
  uint8x16_t vidxs = idxs.r;
  if constexpr (index_ub > 32) {
    vidxs = vandq_u8(vidxs, vdupq_n_u8(0x1F));
  }
  return {.r = vqtbl2q_u8(uint8x16x2_t{table.lower.r, table.upper.r}, vidxs)};
}
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

template<AnyVector TTable, AnyVector TIdxs>
inline VectorFor<typename TTable::Value, TIdxs::size>
shuffle(TTable table, TIdxs idxs, AnyIndexTag auto index_ub, AnyIndexTag auto index_offset) {
  using Value = TTable::Value;
  constexpr std::size_t table_size = TTable::size;
  using Index = TIdxs::Value;
  constexpr std::size_t index_size = TIdxs::size;
  constexpr std::size_t max_index = std::numeric_limits<Index>::max();

  if constexpr (sizeof(Index) < 8 && table_size > max_index + 1) {
    // the table is bigger than what the index type can address → reduce table size
    // this can realistically only happen if `Index` is `u8`
    return shuffle(shrink<max_index + 1>(table), idxs, index_tag<max_index + 1>, index_offset);
  } else if constexpr (is_supernative<Value, index_size>) {
    // the output is super-native → split the indices
    return merge(shuffle(table, get_low(idxs), index_ub, index_offset),
                 shuffle(table, get_high(idxs), index_ub, index_offset));
  } else if constexpr (!std::same_as<Value, u8> || !std::same_as<Index, u8>) {
    // the table or the indices are not `u8` → convert both to `u8`
    const auto idxs8 = shuffle_indices(idxs, index_tag<sizeof(Value)>);
    const auto shuf = shuffle(as<u8>(table), idxs8, index_tag<index_ub * sizeof(Value)>,
                              index_tag<index_offset * sizeof(Value)>);
    return as<Value>(shuf);
  } else if constexpr (index_size < 16) {
    // the indices are sub-native → use full indices
    return shrink<index_size>(shuffle(table, idxs.full, index_ub, index_offset));
  } else if constexpr (table_size < 16) {
    // the table is sub-native → use full table
    return shuffle(table.full, idxs, index_ub, index_offset);
  } else if constexpr (table_size > 64) {
    // table is too big for the shuffle instructions → split table
    const auto lo = shuffle(table.lower, idxs, index_ub, index_offset);
    const auto hi = shuffle(table.upper, idxs, index_ub, index_tag<index_offset + table_size / 2>);
    const auto mask = compare_lt(idxs, broadcast<TIdxs>(Index{index_offset + table_size / 2}));
    return blend(convert<Value>(mask), hi, lo);
  } else {
    static_assert(false, "Unsupported shuffle!");
    std::unreachable();
  }
}

template<AnyVector TTable, AnyVector TIdxs>
inline VectorFor<typename TTable::Value, TIdxs::size> shuffle(TTable table, TIdxs idxs) {
  return shuffle(table, idxs, index_tag<TTable::size>, index_tag<0>);
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_SHUFFLE_HPP
