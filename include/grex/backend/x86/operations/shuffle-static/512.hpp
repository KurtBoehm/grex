// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_SHUFFLE_STATIC_512_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_SHUFFLE_STATIC_512_HPP

#include "grex/backend/x86/instruction-sets.hpp"

#if GREX_X86_64_LEVEL >= 4
#include <algorithm>
#include <array>
#include <cstddef>
#include <optional>
#include <utility>

#include "grex/backend/defs.hpp"
#include "grex/backend/shared/defs.hpp"
#include "grex/backend/shared/operations/shuffle-static.hpp"
#include "grex/backend/x86/operations/bitwise.hpp"
#include "grex/backend/x86/operations/blend-zero-static.hpp"
#include "grex/backend/x86/operations/load.hpp"
#include "grex/backend/x86/operations/shuffle-static/shared.hpp"
#include "grex/backend/x86/types.hpp"
#include "grex/base/defs.hpp"

// TODO These could be improved by using more maskz intrinsics instead of zero blending.

namespace grex::backend {
struct ShufflerShuffle128x4 : public BaseExpensiveOp {
  template<AnyShuffleIndices auto tSh>
  static constexpr bool is_applicable(AutoTag<tSh> /*tag*/) {
    constexpr std::optional<ShuffleIndices<16, 4>> idxs128 = convert<16>(tSh);
    constexpr std::optional<ShuffleIndices<4, 16>> idxs32 = convert<4>(tSh);
    return idxs128.has_value() && !idxs32->subzero;
  }
  template<AnyVector TVec, ShuffleIndicesFor<TVec> tSh>
  static TVec apply(TVec vec, AutoTag<tSh> /*tag*/) {
    using Value = TVec::Value;
    static constexpr int imm8 = convert<16>(tSh).value().imm8();
    static constexpr ShuffleIndices<4, 16> idxs32 = convert<4>(tSh).value();
    static_assert(!idxs32.subzero);

    const i32x16 ivec = reinterpret(vec, type_tag<i32>);
    if constexpr (idxs32.requires_zeroing()) {
      const auto shuf = _mm512_maskz_shuffle_i32x4(idxs32.mask().r, ivec.r, ivec.r, imm8);
      return reinterpret(i32x16{shuf}, type_tag<Value>);
    } else {
      const auto shuf = _mm512_shuffle_i32x4(ivec.r, ivec.r, imm8);
      return reinterpret(i32x16{shuf}, type_tag<Value>);
    }
  }
  template<AnyShuffleIndices auto tSh>
  static constexpr std::pair<f64, f64> cost(AutoTag<tSh> /*idxs*/) {
    return {0.5, 3};
  }
};

struct ShufflerShuffle8x64 : public BaseExpensiveOp {
  template<AnyShuffleIndices auto tSh>
  static constexpr bool is_applicable(AutoTag<tSh> /*tag*/) {
    return tSh.is_lane_local();
  }
  template<AnyVector TVec, ShuffleIndicesFor<TVec> tSh>
  static TVec apply(TVec vec, AutoTag<tSh> /*tag*/) {
    using Value = TVec::Value;
    static constexpr ShuffleIndices<1, 64> shi = convert<1>(tSh).value();
    static constexpr auto idxs = shi.laned_indices().value();

    const i8x64 ivec = reinterpret(vec, type_tag<i8>);
    const auto shuf = _mm512_shuffle_epi8(ivec.r, load(idxs.data(), type_tag<i8x64>).r);
    return reinterpret(i8x64{shuf}, type_tag<Value>);
  }
  template<AnyShuffleIndices auto tSh>
  static constexpr std::pair<f64, f64> cost(AutoTag<tSh> /*idxs*/) {
    return {0.5, 4};
  }
};

struct ShufflerShuffle32x16 : public BaseExpensiveOp {
  template<AnyShuffleIndices auto tSh>
  static constexpr bool is_applicable(AutoTag<tSh> /*tag*/) {
    const std::optional<ShuffleIndices<4, 16>> base = convert<4>(tSh);
    return base.has_value() && base->single_lane().has_value();
  }
  template<AnyVector TVec, ShuffleIndicesFor<TVec> tSh>
  static TVec apply(TVec vec, AutoTag<tSh> /*tag*/) {
    using Value = TVec::Value;
    static constexpr ShuffleIndices<4, 16> bidxs = convert<4>(tSh).value();
    static constexpr int imm8 = bidxs.single_lane().value().imm8();

    const i32x16 ivec = reinterpret(vec, type_tag<i32>);
    const auto shuf = _mm512_shuffle_epi32(ivec.r, _MM_PERM_ENUM(imm8));
    const TVec shufr = reinterpret(i32x16{shuf}, type_tag<Value>);
    return ZeroBlender<tSh.blend_zeros()>::apply(shufr, auto_tag<tSh.blend_zeros()>);
  }
  template<AnyShuffleIndices auto tSh>
  static constexpr std::pair<f64, f64> cost(AutoTag<tSh> /*idxs*/) {
    const auto [c0, c1] = ZeroBlender<tSh.blend_zeros()>::cost(auto_tag<tSh>);
    return {0.5 + c0, std::max<f64>(c1, 1)};
  }
};

struct ShufflerPermutex64x8 : public BaseExpensiveOp {
  template<AnyShuffleIndices auto tSh>
  static constexpr bool is_applicable(AutoTag<tSh> /*tag*/) {
    const std::optional<ShuffleIndices<8, 8>> base = convert<8>(tSh);
    return base.has_value() && base->double_lane().has_value();
  }
  template<AnyVector TVec, ShuffleIndicesFor<TVec> tSh>
  static TVec apply(TVec vec, AutoTag<tSh> /*tag*/) {
    using Value = TVec::Value;
    static constexpr ShuffleIndices<8, 8> bidxs = convert<8>(tSh).value();
    static constexpr int imm8 = bidxs.double_lane().value().imm8();

    const i64x8 ivec = reinterpret(vec, type_tag<i64>);
    const TVec shuffled = reinterpret(i64x8{_mm512_permutex_epi64(ivec.r, imm8)}, type_tag<Value>);
    return ZeroBlender<tSh.blend_zeros()>::apply(shuffled, auto_tag<tSh.blend_zeros()>);
  }
  template<AnyShuffleIndices auto tSh>
  static constexpr std::pair<f64, f64> cost(AutoTag<tSh> /*idxs*/) {
    const auto [c0, c1] = ZeroBlender<tSh.blend_zeros()>::cost(auto_tag<tSh>);
    return {1.0 + c0, std::max<f64>(4, c1)};
  }
};

#define GREX_SHUFFLE_PERMUTEX_VAR(BITS, BYTES, SIZE, COST2) \
  struct ShufflerPermutexVar##BITS##x##SIZE : public BaseExpensiveOp { \
    template<AnyShuffleIndices auto tSh> \
    static constexpr bool is_applicable(AutoTag<tSh> /*tag*/) { \
      const std::optional<ShuffleIndices<BYTES, SIZE>> base = convert<BYTES>(tSh); \
      return base.has_value() && !base->subzero; \
    } \
    template<AnyVector TVec, ShuffleIndicesFor<TVec> tSh> \
    static TVec apply(TVec vec, AutoTag<tSh> /*tag*/) { \
      using Value = TVec::Value; \
      static constexpr ShuffleIndices<BYTES, SIZE> idxs = convert<BYTES>(tSh).value(); \
      static_assert(!idxs.subzero); \
\
      const i##BITS##x##SIZE ivec = reinterpret(vec, type_tag<i##BITS>); \
      const __m512i vidxs = idxs.vector().r; \
      if constexpr (idxs.requires_zeroing()) { \
        const i##BITS##x##SIZE shuf{ \
          _mm512_permutex2var_epi##BITS(ivec.r, vidxs, _mm512_setzero_si512())}; \
        return reinterpret(shuf, type_tag<Value>); \
      } else { \
        return reinterpret(i##BITS##x##SIZE{_mm512_permutexvar_epi##BITS(vidxs, ivec.r)}, \
                           type_tag<Value>); \
      } \
    } \
    template<AnyShuffleIndices auto tSh> \
    static constexpr std::pair<f64, f64> cost(AutoTag<tSh> /*idxs*/) { \
      return {1.0, COST2##.0}; \
    } \
  }; \
\
  struct PairShufflerPermutexVar##BITS##x##SIZE : public BaseExpensiveOp { \
    template<AnyShuffleIndices auto tSh> \
    static constexpr bool is_applicable(AutoTag<tSh> /*tag*/) { \
      const std::optional<ShuffleIndices<BYTES, SIZE>> base = convert<BYTES>(tSh); \
      return base.has_value() && !base->subzero; \
    } \
    template<AnyVector TVec, ShuffleIndicesFor<TVec> tSh> \
    static TVec apply(TVec a, TVec b, AutoTag<tSh> /*tag*/) { \
      using Value = TVec::Value; \
      static constexpr ShuffleIndices<BYTES, SIZE> idxs = convert<BYTES>(tSh).value(); \
      static_assert(!idxs.subzero); \
\
      const i##BITS##x##SIZE ia = reinterpret(a, type_tag<i##BITS>); \
      const i##BITS##x##SIZE ib = reinterpret(b, type_tag<i##BITS>); \
      if constexpr (idxs.requires_zeroing()) { \
        const i##BITS##x##SIZE shuf{ \
          _mm512_maskz_permutex2var_epi##BITS(idxs.mask().r, ia.r, idxs.vector().r, ib.r)}; \
        return reinterpret(shuf, type_tag<Value>); \
      } else { \
        const i##BITS##x##SIZE shuf{_mm512_permutex2var_epi##BITS(ia.r, idxs.vector().r, ib.r)}; \
        return reinterpret(shuf, type_tag<Value>); \
      } \
    } \
    template<AnyShuffleIndices auto tSh> \
    static constexpr std::pair<f64, f64> cost(AutoTag<tSh> /*idxs*/) { \
      return {1.0, COST2##.0}; \
    } \
  };

// TODO This 64-bit version does not seem to be faster than the 32-bit version on any architecture…
GREX_SHUFFLE_PERMUTEX_VAR(64, 8, 8, 5)

GREX_SHUFFLE_PERMUTEX_VAR(32, 4, 16, 5)

GREX_SHUFFLE_PERMUTEX_VAR(16, 2, 32, 7)

#if GREX_HAS_AVX512VBMI
GREX_SHUFFLE_PERMUTEX_VAR(8, 1, 64, 7)
#else
// TODO This version is not particularly efficient: Using as many 128-bit permutations as necessary
//      and then using intra-lane shuffling is more efficient, though that is more complicated
//      to implement.
struct ShufflerPermutexVar8x64 : public BaseExpensiveOp {
  template<AnyShuffleIndices auto tSh>
  static constexpr bool is_applicable(AutoTag<tSh> /*tag*/) {
    return true;
  }
  template<AnyVector TVec, ShuffleIndicesFor<TVec> tSh>
  static TVec apply(TVec vec, AutoTag<tSh> /*tag*/) {
    using Value = TVec::Value;
    static constexpr ShuffleIndices<1, 64> idxs = convert<1>(tSh).value();
    static_assert(!idxs.subzero);

    // cross-lane 16-bit shuffling
    auto f16 = [&](std::size_t i, ShuffleIndex sh) {
      return is_index(sh) ? i16(u8(sh) / 2) : i16(i);
    };
    // even indices
    constexpr auto idxs16a = static_apply<32>(
      [&]<std::size_t... tIdxs>() { return std::array{f16(tIdxs, tSh[2 * tIdxs])...}; });
    // odd indices
    constexpr auto idxs16b = static_apply<32>(
      [&]<std::size_t... tIdxs>() { return std::array{f16(tIdxs, tSh[2 * tIdxs + 1])...}; });

    // per-lane 8-bit shuffling
    auto f8 = [&](std::size_t i, bool even) {
      const auto sh = tSh[i];
      return ((i % 2 == 0) == even && is_index(sh)) ? i8(i / 2 * 2 + u8(sh) % 2) : i8(-1);
    };
    // even indices
    constexpr auto idxs8a =
      static_apply<64>([&]<std::size_t... tIdxs>() { return std::array{f8(tIdxs, true)...}; });
    // odd indices
    constexpr auto idxs8b =
      static_apply<64>([&]<std::size_t... tIdxs>() { return std::array{f8(tIdxs, false)...}; });

    const i8x64 ivec = reinterpret(vec, type_tag<i8>);
    const i16x32 shuf16a{
      _mm512_permutexvar_epi16(load(idxs16a.data(), type_tag<i16x32>).r, ivec.r)};
    const i16x32 shuf16b{
      _mm512_permutexvar_epi16(load(idxs16b.data(), type_tag<i16x32>).r, ivec.r)};
    const i8x64 shuf8a{_mm512_shuffle_epi8(shuf16a.r, load(idxs8a.data(), type_tag<i8x64>).r)};
    const i8x64 shuf8b{_mm512_shuffle_epi8(shuf16b.r, load(idxs8b.data(), type_tag<i8x64>).r)};
    return reinterpret(bitwise_or(shuf8a, shuf8b), type_tag<Value>);
  }
  template<AnyShuffleIndices auto tSh>
  static constexpr std::pair<f64, f64> cost(AutoTag<tSh> /*idxs*/) {
    return {4.0, 7.0};
  }
};
struct PairShufflerPermutexVar8x64 : public BaseExpensiveOp {
  template<AnyShuffleIndices auto tSh>
  static constexpr bool is_applicable(AutoTag<tSh> /*tag*/) {
    return true;
  }
  template<AnyVector TVec, ShuffleIndicesFor<TVec> tSh>
  static TVec apply(TVec a, TVec b, AutoTag<tSh> /*tag*/) {
    using Value = TVec::Value;
    static constexpr ShuffleIndices<1, 64> idxs = convert<1>(tSh).value();
    static_assert(!idxs.subzero);

    // cross-lane 16-bit shuffling
    auto f16 = [&](std::size_t i, ShuffleIndex sh) {
      return is_index(sh) ? i16(u8(sh) / 2) : i16(i);
    };
    // even indices
    constexpr auto idxs16a = static_apply<32>(
      [&]<std::size_t... tIdxs>() { return std::array{f16(tIdxs, tSh[2 * tIdxs])...}; });
    // odd indices
    constexpr auto idxs16b = static_apply<32>(
      [&]<std::size_t... tIdxs>() { return std::array{f16(tIdxs, tSh[2 * tIdxs + 1])...}; });

    // per-lane 8-bit shuffling
    auto f8 = [&](std::size_t i, bool even) {
      const auto sh = tSh[i];
      return ((i % 2 == 0) == even && is_index(sh)) ? i8(i / 2 * 2 + u8(sh) % 2) : i8(-1);
    };
    // even indices
    constexpr auto idxs8a =
      static_apply<64>([&]<std::size_t... tIdxs>() { return std::array{f8(tIdxs, true)...}; });
    // odd indices
    constexpr auto idxs8b =
      static_apply<64>([&]<std::size_t... tIdxs>() { return std::array{f8(tIdxs, false)...}; });

    const i8x64 ia = reinterpret(a, type_tag<i8>);
    const i8x64 ib = reinterpret(b, type_tag<i8>);
    const i16x32 shuf16a{
      _mm512_permutex2var_epi16(ia.r, load(idxs16a.data(), type_tag<i16x32>).r, ib.r)};
    const i16x32 shuf16b{
      _mm512_permutex2var_epi16(ia.r, load(idxs16b.data(), type_tag<i16x32>).r, ib.r)};
    const i8x64 shuf8a{_mm512_shuffle_epi8(shuf16a.r, load(idxs8a.data(), type_tag<i8x64>).r)};
    const i8x64 shuf8b{_mm512_shuffle_epi8(shuf16b.r, load(idxs8b.data(), type_tag<i8x64>).r)};
    return reinterpret(bitwise_or(shuf8a, shuf8b), type_tag<Value>);
  }
  template<AnyShuffleIndices auto tSh>
  static constexpr std::pair<f64, f64> cost(AutoTag<tSh> /*idxs*/) {
    return {4.0, 7.0};
  }
};
#endif

template<AnyShuffleIndices auto tSh>
requires((tSh.value_size * tSh.size == 64))
struct ShufflerTrait<tSh> {
  using Shuffler =
    CheapestType<tSh, ShufflerBlendZero, ShufflerShuffle128x4, ShufflerShuffle8x64,
                 ShufflerShuffle32x16, ShufflerPermutex64x8, ShufflerPermutexVar64x8,
                 ShufflerPermutexVar32x16, ShufflerPermutexVar16x32, ShufflerPermutexVar8x64>;
};
template<AnyShuffleIndices auto tSh>
requires((tSh.value_size * tSh.size == 64))
struct PairShufflerTrait<tSh> {
  // TODO With AVX-512, the vpermi2 family can be used to merge two permutations
  using Shuffler =
    CheapestType<tSh, PairShufflerSingle, PairShufflerPermutexVar64x8, PairShufflerPermutexVar32x16,
                 PairShufflerPermutexVar16x32, PairShufflerPermutexVar8x64, PairShufflerBlend>;
};
} // namespace grex::backend
#endif

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_SHUFFLE_STATIC_512_HPP
