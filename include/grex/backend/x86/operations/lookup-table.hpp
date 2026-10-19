// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_LOOKUP_TABLE_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_LOOKUP_TABLE_HPP

#include <array>
#include <concepts>
#include <cstddef>

#include "grex/backend/base.hpp"
#include "grex/backend/choosers.hpp"
#include "grex/backend/shared/operations/lookup-table.hpp"
#include "grex/backend/x86/operations/load.hpp"
#include "grex/backend/x86/operations/reinterpret.hpp"
#include "grex/backend/x86/operations/split.hpp"
#include "grex/backend/x86/types.hpp"
#include "grex/base.hpp"

namespace grex::backend {
template<Vectorizable T, std::size_t N>
struct LookupTable {
  static constexpr bool is_enabled = false;
};

/** If the table is at most twice the native size, use `shuffle`. */
template<Vectorizable T, std::size_t N>
requires(N <= 2 * max_native_size<T>)
struct LookupTable<T, N> : ShuffleLookupTable<T, N> {
  using Index = u8;
  static constexpr bool is_enabled = true;

  using ShuffleLookupTable<T, N>::ShuffleLookupTable;
};

#if GREX_X86_64_LEVEL == 3
/** A specialized implementation for a 16×64-bit lookup table. */
template<Vectorizable T, std::size_t N>
requires(sizeof(T) == 8 && N == 16)
struct LookupTable<T, N> {
  using Index = u8;
  static constexpr bool is_enabled = true;

  explicit LookupTable(const std::array<T, N>& table) : table_{make_table(table)} {}

  [[nodiscard, GREX_ALWAYS_INLINE]] VectorFor<T, 16> lookup(u8x16 idxs) const {
    const auto idxs2 = _mm_broadcastsi128_si256(idxs.r);

    // Each byte plane is shuffled according to `idxs`. What remains is to re-assemble
    // the bytes.
    const auto b04 = _mm256_shuffle_epi8(table_[0].r, idxs2);
    const auto b15 = _mm256_shuffle_epi8(table_[1].r, idxs2);
    const auto b26 = _mm256_shuffle_epi8(table_[2].r, idxs2);
    const auto b37 = _mm256_shuffle_epi8(table_[3].r, idxs2);

    // Merge two adjacent byte planes into 16-bit planes; the first two vectors span
    // the lower half of the output vector, the latter two the upper half.
    const auto s0h01_45 = _mm256_unpacklo_epi8(b04, b15);
    const auto s0h23_67 = _mm256_unpacklo_epi8(b26, b37);
    const auto s1h01_45 = _mm256_unpackhi_epi8(b04, b15);
    const auto s1h23_67 = _mm256_unpackhi_epi8(b26, b37);

    // Merge two adjacent 16-bit planes into 32-bit planes; each vector now spans one
    // of the four registers making up the output vector, with the lower 32-bit
    // plane of each entry in the lower half and the upper 32-bit planes in the upper
    // half.
    const auto s0w = _mm256_unpacklo_epi16(s0h01_45, s0h23_67);
    const auto s1w = _mm256_unpackhi_epi16(s0h01_45, s0h23_67);
    const auto s2w = _mm256_unpacklo_epi16(s1h01_45, s1h23_67);
    const auto s3w = _mm256_unpackhi_epi16(s1h01_45, s1h23_67);

    // Permute so that each 128-bit lane contains the lower 32-bit plane for both
    // components that end up in this lane in the lower 64 bits and the upper 32-bit
    // plane of those components in the upper 64 bits.
    const auto s0mix64 = _mm256_permute4x64_epi64(s0w, 0b11'01'10'00);
    const auto s1mix64 = _mm256_permute4x64_epi64(s1w, 0b11'01'10'00);
    const auto s2mix64 = _mm256_permute4x64_epi64(s2w, 0b11'01'10'00);
    const auto s3mix64 = _mm256_permute4x64_epi64(s3w, 0b11'01'10'00);

    // Permute each 128-bit lane to merge the 32-bit planes.
    const auto d0 = _mm256_shuffle_epi32(s0mix64, 0b11'01'10'00);
    const auto d1 = _mm256_shuffle_epi32(s1mix64, 0b11'01'10'00);
    const auto d2 = _mm256_shuffle_epi32(s2mix64, 0b11'01'10'00);
    const auto d3 = _mm256_shuffle_epi32(s3mix64, 0b11'01'10'00);

    // Assemble the super-native vector as `u64` and re-interpret later.
    const VectorFor<u64, 16> dst{
      .lower = {.lower = {.r = d0}, .upper = {.r = d1}},
      .upper = {.lower = {.r = d2}, .upper = {.r = d3}},
    };

    return as<T>(dst);
  }

  [[nodiscard, GREX_ALWAYS_INLINE]] VectorFor<T, 32> lookup(u8x32 idxs) const {
    return {.lower = lookup(get_low(idxs)), .upper = lookup(get_high(idxs))};
  }

  template<AnySuperNativeVector Idxs>
  requires(std::same_as<ValueOf<Idxs>, u8>)
  [[nodiscard, GREX_ALWAYS_INLINE]] VectorFor<T, size_of<Idxs>> lookup(Idxs idxs) const {
    return {.lower = lookup(get_low(idxs)), .upper = lookup(get_high(idxs))};
  }

  [[nodiscard, GREX_ALWAYS_INLINE]] VectorFor<T, 8> lookup(SubVector<u8, 8> idxs) const {
    return lookup(idxs.full).lower;
  }

  [[nodiscard, GREX_ALWAYS_INLINE]] VectorFor<T, 4> lookup(SubVector<u8, 4> idxs) const {
    return lookup(SubVector<u8, 8>{idxs.full}).lower;
  }

  [[nodiscard, GREX_ALWAYS_INLINE]] VectorFor<T, 2> lookup(SubVector<u8, 2> idxs) const {
    const __m256i ext = as<u64>(lookup(SubVector<u8, 4>{idxs.full})).r;
    return as<T>(u64x2{.r = _mm256_castsi256_si128(ext)});
  }

private:
  static std::array<u64x4, 4> make_table(const std::array<T, N>& table) {
    const VectorFor<u64, 16> vtable = as<u64>(load(table.data(), type_tag<VectorFor<T, 16>>));
    const auto zero = _mm256_setzero_si256();

    // Maps the `u64x4` `[a, b, c, d]` to `[a, c, b, d]`.
    const auto swap64 = [](__m256i v) { return _mm256_permute4x64_epi64(v, 0b11'01'10'00); };
    // Maps the `u128x2` `a` and `b` to `[a[0], b[0]]`.
    const auto merge128lo = [](__m256i a, __m256i b) {
      return _mm256_inserti128_si256(a, _mm256_castsi256_si128(b), 1);
    };
    // Maps the `u128x2` `a` and `b` to `[a[1], b[1]]`.
    const auto merge128hi = [](__m256i a, __m256i b) {
      return _mm256_permute2x128_si256(a, b, 0x31);
    };

    const auto t0 = vtable.lower.lower.r;
    const auto t1 = vtable.lower.upper.r;
    const auto t2 = vtable.upper.lower.r;
    const auto t3 = vtable.upper.upper.r;

    // For the later discussion, `tN` corresponds to lane `N` for `table`.

    // Move the lower 32 bits of each entry into the lower 64 bits of each 128-bit lane and vice
    // versa. As 32-bit vectors: `a0 = [t0 t2 : t1 t3 | t4 t6 : t5 t7]`.
    const auto a0 = _mm256_shuffle_epi32(t0, 0b11'01'10'00);
    const auto a1 = _mm256_shuffle_epi32(t1, 0b11'01'10'00);
    const auto a2 = _mm256_shuffle_epi32(t2, 0b11'01'10'00);
    const auto a3 = _mm256_shuffle_epi32(t3, 0b11'01'10'00);

    // Move the lower 32 bits of each entry into the lower 128-bit lane and vice versa. As 32-bit
    // vectors: `b0 = [t0 t2 : t4 t6 | t1 t3 : t5 t7]`
    const auto b0 = swap64(a0);
    const auto b1 = swap64(a1);
    const auto b2 = swap64(a2);
    const auto b3 = swap64(a3);

    // Swap around so that `c0` contains the lower 32 bits of each entry in `t0` and `t1`, `c1` the
    // corresponding upper 32 bits, and analogously for `t2`/`t3` and `c2`/`c3`. As 32-bit vectors:
    // `c0 = [t0 t2 : t4 t6 | t8 t10 : t12 t14]`
    const auto c0 = merge128lo(b0, b1);
    const auto c1 = merge128hi(b0, b1);
    const auto c2 = merge128lo(b2, b3);
    const auto c3 = merge128hi(b2, b3);

    // Mask out the lower and upper 16 bits in each 32-bit block. As 16-bit vectors:
    // `d0lo = [t0 0 t4 0 : t8 0 t12 0 | …]`
    // `d0hi = [0 t1 0 t5 : 0 t9 0 t13 | …]`
    const auto d0lo = _mm256_blend_epi16(zero, c0, 0b01010101);
    const auto d0hi = _mm256_srli_epi32(c0, 16);
    const auto d1lo = _mm256_blend_epi16(zero, c1, 0b01010101);
    const auto d1hi = _mm256_srli_epi32(c1, 16);
    const auto d2lo = _mm256_blend_epi16(zero, c2, 0b01010101);
    const auto d2hi = _mm256_srli_epi32(c2, 16);
    const auto d3lo = _mm256_blend_epi16(zero, c3, 0b01010101);
    const auto d3hi = _mm256_srli_epi32(c3, 16);

    // Merge the lower 16 bits of each 128-bit lane in `c0` in the lower 64 bits of that lane in
    // `e0` with the upper 16 bits in the upper 64 bits and analogously for the others. As 16-bit
    // vectors:
    // `e0 = [t0 t4 t8 t12 : t1 t5 t9 t13 | t16 t20 t24 t28 : t17 t21 t25 t29]`
    const auto e0 = _mm256_packus_epi32(d0lo, d0hi);
    const auto e1 = _mm256_packus_epi32(d1lo, d1hi);
    const auto e2 = _mm256_packus_epi32(d2lo, d2hi);
    const auto e3 = _mm256_packus_epi32(d3lo, d3hi);

    // Move the lower 16-bit planes into the lower 128 bits and vice versa again. As 16-bit vectors:
    // `f0 = [t0 t4 t8 t12 : t16 t20 t24 t28 | t1 t5 t9 t13 : t17 t21 t25 t29]`.
    // This corresponds to planes 0 and 1 in the lower half and lanes 2 and 3 in the upper for `t0`
    // and `t1`; `f2` has the other lanes of `t0` and `t1` while `f2` and `f3` are the analogous
    // vectors for `t2` and `t3`.
    const auto f0 = swap64(e0);
    const auto f1 = swap64(e1);
    const auto f2 = swap64(e2);
    const auto f3 = swap64(e3);

    // Swap around so that `g0` contains bytes 0 and 1 of `t0` and `t1` in the lower half and bytes
    // 4 and 5 in the upper while `g1` has bytes 2/3 and 6/7; `g2` and `g3` are analogous for `t2`
    // and `t3`. As 16-bit vectors:
    // `g0 = [t0 t4 t8 t12 : t16 t20 t24 t28 | t2 t6 t10 t14 : t18 t22 t26 t30]`.
    const auto g0 = merge128lo(f0, f1);
    const auto g1 = merge128hi(f0, f1);
    const auto g2 = merge128lo(f2, f3);
    const auto g3 = merge128hi(f2, f3);

    // Mask out the lower and upper 8 bits in each 16-bit block. As 8-bit vectors:
    // `h0lo = [t0 0 t8 0 t16 0 t24 0 : … | t4 0 t12 0 t20 0 t28 0 : …]`
    // `h0hi = [0 t1 0 t9 0 t17 0 t25 : … | t5 0 t13 0 t21 0 t29 0 : …]`
    const auto h0lo = _mm256_srli_epi16(_mm256_slli_epi16(g0, 8), 8);
    const auto h0hi = _mm256_srli_epi16(g0, 8);
    const auto h1lo = _mm256_srli_epi16(_mm256_slli_epi16(g1, 8), 8);
    const auto h1hi = _mm256_srli_epi16(g1, 8);
    const auto h2lo = _mm256_srli_epi16(_mm256_slli_epi16(g2, 8), 8);
    const auto h2hi = _mm256_srli_epi16(g2, 8);
    const auto h3lo = _mm256_srli_epi16(_mm256_slli_epi16(g3, 8), 8);
    const auto h3hi = _mm256_srli_epi16(g3, 8);

    // Merge the lower 8 bits of each 128-bit lane in `g0` in the lower 64 bits of that lane in
    // `i0` with the upper 8 bits in the upper 64 bits and analogously for the others. As 8-bit
    // vectors:
    // `i0 = [t0 t8 t16 t24 … : t1 t9 t17 t25 … | t4 t12 t20 t28 … : t5 t13 t21 t29 …]`
    // This corresponds to bytes planes 0, 1, 4, and 5 in each 64-bit lane for `t0` and `t1`, with
    // `i1` containing lanes 2, 3, 6, and 7 and `i2`/`i3` representing `t2` and `t3`.
    const auto i0 = _mm256_packus_epi16(h0lo, h0hi);
    const auto i1 = _mm256_packus_epi16(h1lo, h1hi);
    const auto i2 = _mm256_packus_epi16(h2lo, h2hi);
    const auto i3 = _mm256_packus_epi16(h3lo, h3hi);

    // Merge the planes into the corresponding 128-bit lanes; for instance, plane 0 is spread over
    // the lower 64 bits of `i0` and `i2`, which is why they can be merged using 64-bit operations.
    // As 8-bit vectors:
    // `j0 = [t0 t8 t16 t24 … : t64 t72 t80 t88 … | t4 t12 t20 t28 … : t68 t76 t84 t92 …]`
    const auto j0 = _mm256_unpacklo_epi64(i0, i2);
    const auto j1 = _mm256_unpackhi_epi64(i0, i2);
    const auto j2 = _mm256_unpacklo_epi64(i1, i3);
    const auto j3 = _mm256_unpackhi_epi64(i1, i3);

    return std::array{u64x4{j0}, u64x4{j1}, u64x4{j2}, u64x4{j3}};
  }

  std::array<u64x4, 4> table_;
};
#endif
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_LOOKUP_TABLE_HPP
