// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_NEON_OPERATIONS_MULTIBYTE_HPP
#define INCLUDE_GREX_BACKEND_NEON_OPERATIONS_MULTIBYTE_HPP

#include <cstddef>

#include <arm_neon.h>

#include "grex/backend/choosers.hpp"
#include "grex/backend/defs.hpp" // IWYU pragma: keep
#include "grex/backend/neon/operations/load.hpp" // IWYU pragma: keep
#include "grex/backend/neon/types.hpp"
#include "grex/base.hpp"

// shared definitions
#include "grex/backend/shared/operations/multibyte.hpp" // IWYU pragma: keep

// Load integers consisting of M bytes into integers containing N = 2^B bytes,
// assuming N = std::bitceil(M).
// We call M the `src_bytes` and N the `dst_bytes`, while `size` denotes the number of values
// being converted.
// One basic assumption is that the underlying memory is padded at the beginning and end
// by the numnber of bytes in the largest supported SIMD register

namespace grex::backend {
// N == M: simply load
template<std::size_t tSrc, AnyVector TDst>
requires(!AnySuperNativeVector<TDst> && tSrc == sizeof(typename TDst::Value))
inline TDst load_multibyte(const u8* ptr, IndexTag<tSrc> /*src*/, TypeTag<TDst> /*dst*/) {
  const auto raw = load(ptr, type_tag<VectorFor<u8, tSrc * TDst::size>>).registr();
  return TDst{as<typename TDst::Value>(raw)};
}

template<std::size_t tSrc>
requires(tSrc < 8)
inline u64x2 load_multibyte(const u8* ptr, IndexTag<tSrc> /*src*/, TypeTag<u64x2> /*dst*/) {
  // the comments are based on M == 5, but are quite analogous for M == 6 and M == 7
  // offset = 8 - 5 = 3
  constexpr std::size_t offset = 8 - tSrc;
  // |---00000|11111---|
  const uint64x2_t raw = vreinterpretq_u64_u8(vld1q_u8(ptr - offset));
  // |zzz---00|zzz11111|
  const uint64x2_t v1 = vshlq_n_u64(raw, 8 * offset);
  // |---00000|zzz11111|
  const uint64x2_t mix = vcopyq_lane_u64(raw, 1, vget_high_u64(v1), 0);
  // |00000zzz|11111zzz|
  return {.r = vshrq_n_u64(mix, 8 * offset)};
}
inline u32x4 load_multibyte(const u8* ptr, IndexTag<3> /*src*/, TypeTag<u32x4> /*dst*/) {
  constexpr std::array<u8, 16> idxs{0, 1, 2, 255, 3, 4, 5, 255, 6, 7, 8, 255, 9, 10, 11, 255};
  // |0001|1122|2333|----|
  const uint8x16_t raw = vld1q_u8(ptr);
  // |000z|111z|222z|333z|
  return {.r = vreinterpretq_u32_u8(vqtbl1q_u8(raw, vld1q_u8(idxs.data())))};
}
inline SubVector<u32, 2, 4> load_multibyte(const u8* ptr, IndexTag<3> src,
                                           TypeTag<SubVector<u32, 2, 4>> /*dst*/) {
  return SubVector<u32, 2, 4>{load_multibyte(ptr, src, type_tag<u32x4>)};
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_MULTIBYTE_HPP
