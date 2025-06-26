// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_MULTIBYTE_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_MULTIBYTE_HPP

#include <array>
#include <cstddef>

#include <immintrin.h>

#include "grex/backend/choosers.hpp"
#include "grex/backend/defs.hpp"
#include "grex/backend/x86/instruction-sets.hpp"
#include "grex/backend/x86/operations/load.hpp"
#include "grex/backend/x86/types.hpp"
#include "grex/base/defs.hpp"

#if GREX_X86_64_LEVEL >= 3
#include "grex/backend/x86/operations/set.hpp"
#endif

// Load integers consisting of M bytes into integers containing N = 2^B bytes,
// assuming N = std::bitceil(M).
// We call M the `src_bytes` and N the `dst_bytes`, while `size` denotes the number of values
// being converted.

namespace grex::backend {
namespace mb {
template<std::size_t tSrc, std::size_t tDst, std::size_t tSize, std::size_t tPart = tSize>
requires((tDst * tSize) == 16)
inline constexpr auto shuffle_indices_128 = static_apply<tSize * tDst>([]<std::size_t... tIdxs>() {
  auto op = []<std::size_t tIdx>(IndexTag<tIdx> /*idx*/) {
    constexpr std::size_t j = tIdx % tDst;
    constexpr std::size_t k = tIdx / tDst;
    return (j < tSrc && k < tPart) ? i8(j + tSrc * k) : i8(-1);
  };
  return std::array{op(index_tag<tIdxs>)...};
});
} // namespace mb

// N == M: simply load
template<std::size_t tSrc, AnyVector TDst>
requires(!AnySuperNativeVector<TDst> && tSrc == sizeof(typename TDst::Value))
inline TDst load_multibyte(const u8* ptr, IndexTag<tSrc> /*src*/, TypeTag<TDst> /*dst*/) {
  const auto raw = load(ptr, type_tag<VectorFor<u8, tSrc * TDst::size>>).registr();
  return TDst{raw};
}

#if GREX_X86_64_LEVEL >= 2
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
// TODO This could be extended to level 1 by using AND with a constant mask,
// but that requires a move instruction from static memory
inline u64x2 load_multibyte(const u8* ptr, IndexTag<6> /*src*/, TypeTag<u64x2> /*dst*/) {
  // |-000|111-|
  const __m128i raw = load(ptr - 2, type_tag<u8x16>).r;
  // |000-|111-|
  const __m128i shu = _mm_shufflelo_epi16(raw, 0b11'10'01);
  return {.r = _mm_blend_epi16(shu, _mm_setzero_si128(), 0b10'00'10'00)};
}
#else
template<std::size_t tSrc>
requires(tSrc < 8)
inline u64x2 load_multibyte(const u8* ptr, IndexTag<tSrc> /*src*/, TypeTag<u64x2> /*dst*/) {
  // the comments are based on M == 5, but are quite analogous for M == 6 and M == 7
  // offset = 8 - 5 = 3
  constexpr std::size_t offset = 8 - tSrc;
  // |---00000|11111---|
  const __m128i raw = load(ptr - offset, type_tag<u8x16>).r;
  // |zzz---00|zzz11111|
  const __m128i v1 = _mm_slli_epi64(raw, 8 * offset);
  // |---00000|zzz11111|
  const __m128d mix = _mm_move_sd(_mm_castsi128_pd(v1), _mm_castsi128_pd(raw));
  // |00000zzz|11111zzz|
  return {.r = _mm_srli_epi64(_mm_castpd_si128(mix), 8 * offset)};
}
inline u32x4 load_multibyte(const u8* ptr, IndexTag<3> /*src*/, TypeTag<u32x4> /*dst*/) {
  // |----|0001|1122|2333|
  const __m128i raw = load(ptr - 4, type_tag<u8x16>).r;
  // |zzz-|---0|zzz1|1222|
  const __m128i v2 = _mm_slli_epi64(raw, 24);
  // |zz00|0111|2223|3300|
  const __m128i v1 = _mm_bsrli_si128(raw, 2);
  // |z---|-000|z112|2233|
  const __m128i v0 = _mm_slli_epi64(raw, 8);
  // |zzz1|1122|1222|2333|
  const __m128i v23 = _mm_unpackhi_epi32(v2, raw);
  // |z---|zz00|-000|0111|
  const __m128i v01 = _mm_unpacklo_epi32(v0, v1);
  const __m128i mix = _mm_unpackhi_epi64(v01, v23);
  return {.r = _mm_srli_epi32(mix, 8)};
}
inline SubVector<u32, 2, 4> load_multibyte(const u8* ptr, IndexTag<3> /*src*/,
                                           TypeTag<SubVector<u32, 2, 4>> /*dst*/) {
  // |--00|0111|----|----|
  const __m128i raw = load(ptr - 2, type_tag<SubVector<u8, 8, 16>>).registr();
  // |-000|111z|----|----|
  const __m128i v0 = _mm_srli_epi64(raw, 8);
  // |-000|0111|----|----|
  const __m128 mix = _mm_move_ss(_mm_castsi128_ps(raw), _mm_castsi128_ps(v0));
  return SubVector<u32, 2, 4>{_mm_srli_epi32(_mm_castps_si128(mix), 8)};
}
#endif

#if GREX_X86_64_LEVEL >= 3
template<std::size_t tSrc, typename TDst, std::size_t tSize>
requires(tSrc < sizeof(TDst) && (sizeof(TDst) * tSize) == 32)
inline Vector<TDst, tSize> load_multibyte(const u8* ptr, IndexTag<tSrc> /*src*/,
                                          TypeTag<Vector<TDst, tSize>> /*dst*/) {
  static constexpr std::size_t src_bytes = tSrc;
  static constexpr std::size_t dst_bytes = sizeof(TDst);
  static constexpr std::size_t offset = dst_bytes - src_bytes;
  const auto raw = load(ptr - (tSize / 2) * offset, type_tag<Vector<u8, dst_bytes * tSize>>).r;
  const auto idxs = static_apply<dst_bytes * tSize>([&]<std::size_t... tIdxs>() {
    return _mm256_setr_epi8(
      ((tIdxs % dst_bytes < src_bytes)
         ? (tIdxs % dst_bytes + (tIdxs / dst_bytes) * src_bytes + (tSize / 2) * offset)
         : i8(-1))...);
  });
  return {.r = _mm256_shuffle_epi8(raw, idxs)};
}
#endif
#if GREX_X86_64_LEVEL >= 4
template<std::size_t tSrc, typename TDst, std::size_t tSize>
requires(tSrc < sizeof(TDst) && (sizeof(TDst) * tSize) == 64)
inline Vector<TDst, tSize> load_multibyte(const u8* ptr, IndexTag<tSrc> /*src*/,
                                          TypeTag<Vector<TDst, tSize>> /*dst*/) {
  static constexpr std::size_t src_bytes = tSrc;
  static constexpr std::size_t dst_bytes = sizeof(TDst);
  static constexpr std::size_t offset = (dst_bytes - src_bytes) * tSize;
  __m512i out = load(ptr - offset / 2, type_tag<Vector<u8, dst_bytes * tSize>>).r;
  const __m512i idxs32 = static_apply<16>([]<std::size_t... tIdxs>() {
    return set(type_tag<Vector<i32, 16>>,
               i32{(tIdxs < 4) ? (tIdxs + offset / 8)
                               : ((tIdxs >= 12) ? (tIdxs - offset / 8) : tIdxs)}...)
      .r;
  });
  out = _mm512_permutexvar_epi32(idxs32, out);
  const __m512i idxs8 = static_apply<64>([]<std::size_t... tIdxs>() {
    return set(type_tag<Vector<i8, 64>>,
               ((tIdxs % dst_bytes < src_bytes)
                  ? i8{tIdxs % dst_bytes + ((tIdxs % 16) / dst_bytes) * src_bytes +
                       ((tIdxs / 16) % 2) * (offset / 4)}
                  : i8{-1})...)
      .r;
  });
  return {.r = _mm512_shuffle_epi8(out, idxs8)};
}
#endif

// super-native
template<std::size_t tSrc, typename THalf>
inline SuperVector<THalf> load_multibyte(const u8* ptr, IndexTag<tSrc> /*src*/,
                                         TypeTag<SuperVector<THalf>> /*dst*/) {
  return {
    .lower = load_multibyte(ptr, index_tag<tSrc>, type_tag<THalf>),
    .upper = load_multibyte(ptr + tSrc * THalf::size, index_tag<tSrc>, type_tag<THalf>),
  };
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_MULTIBYTE_HPP
