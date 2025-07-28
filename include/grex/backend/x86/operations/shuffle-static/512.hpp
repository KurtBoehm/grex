// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_SHUFFLE_STATIC_512_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_SHUFFLE_STATIC_512_HPP

#include "grex/backend/x86/instruction-sets.hpp"

#if GREX_X86_64_LEVEL >= 4
#include <array>
#include <cstddef>
#include <optional>
#include <utility>

#include "grex/backend/defs.hpp"
#include "grex/backend/x86/defs.hpp"
#include "grex/backend/x86/operations/bitwise.hpp"
#include "grex/backend/x86/operations/blend-zero-static.hpp"
#include "grex/backend/x86/operations/load.hpp"
#include "grex/backend/x86/operations/shuffle-static/base.hpp"
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
    static constexpr ShuffleIndices<16, 4> idxs128 = convert<16>(tSh).value();
    static constexpr ShuffleIndices<4, 16> idxs32 = convert<4>(tSh).value();
    static_assert(!idxs32.subzero);

    const i32x16 ivec = reinterpret(vec, type_tag<i32>);
    if constexpr (idxs32.requires_zeroing()) {
      const auto shuf = _mm512_maskz_shuffle_i32x4(idxs32.mask().r, ivec.r, ivec.r, idxs128.imm8());
      return reinterpret(i32x16{shuf}, type_tag<Value>);
    } else {
      const auto shuf = _mm512_shuffle_i32x4(ivec.r, ivec.r, idxs128.imm8());
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
    static constexpr ShuffleIndices<4, 4> lidxs = bidxs.single_lane().value();

    const i32x16 ivec = reinterpret(vec, type_tag<i32>);
    const auto shuf = _mm512_shuffle_epi32(ivec.r, _MM_PERM_ENUM(lidxs.imm8()));
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
    static constexpr ShuffleIndices<8, 4> lidxs = bidxs.double_lane().value();

    const i64x8 ivec = reinterpret(vec, type_tag<i64>);
    const TVec shuffled =
      reinterpret(i64x8{_mm512_permutex_epi64(ivec.r, lidxs.imm8())}, type_tag<Value>);
    return ZeroBlender<tSh.blend_zeros()>::apply(shuffled, auto_tag<tSh.blend_zeros()>);
  }
  template<AnyShuffleIndices auto tSh>
  static constexpr std::pair<f64, f64> cost(AutoTag<tSh> /*idxs*/) {
    const auto [c0, c1] = ZeroBlender<tSh.blend_zeros()>::cost(auto_tag<tSh>);
    return {1.0 + c0, std::max<f64>(4, c1)};
  }
};

// TODO This 64-bit version does not seem to be faster than the 32-bit version on any architecture…
struct ShufflerPermutexVar64x8 : public BaseExpensiveOp {
  template<AnyShuffleIndices auto tSh>
  static constexpr bool is_applicable(AutoTag<tSh> /*tag*/) {
    const std::optional<ShuffleIndices<8, 8>> base = convert<8>(tSh);
    return base.has_value() && !base->subzero;
  }
  template<AnyVector TVec, ShuffleIndicesFor<TVec> tSh>
  static TVec apply(TVec vec, AutoTag<tSh> /*tag*/) {
    using Value = TVec::Value;
    static constexpr ShuffleIndices<8, 8> idxs = convert<8>(tSh).value();
    static_assert(!idxs.subzero);

    const i64x8 ivec = reinterpret(vec, type_tag<i64>);
    if constexpr (idxs.requires_zeroing()) {
      const i64x8 shuf{_mm512_permutex2var_epi64(idxs.vector().r, ivec.r, _mm512_setzero_si512())};
      return reinterpret(shuf, type_tag<Value>);
    } else {
      return reinterpret(i64x8{_mm512_permutexvar_epi64(idxs.vector().r, ivec.r)}, type_tag<Value>);
    }
  }
  template<AnyShuffleIndices auto tSh>
  static constexpr std::pair<f64, f64> cost(AutoTag<tSh> /*idxs*/) {
    return {1.0, 5.0};
  }
};
struct ShufflerPermutexVar32x16 : public BaseExpensiveOp {
  template<AnyShuffleIndices auto tSh>
  static constexpr bool is_applicable(AutoTag<tSh> /*tag*/) {
    const std::optional<ShuffleIndices<4, 16>> base = convert<4>(tSh);
    return base.has_value() && !base->subzero;
  }
  template<AnyVector TVec, ShuffleIndicesFor<TVec> tSh>
  static TVec apply(TVec vec, AutoTag<tSh> /*tag*/) {
    using Value = TVec::Value;
    static constexpr ShuffleIndices<4, 16> idxs = convert<4>(tSh).value();
    static_assert(!idxs.subzero);

    const i32x16 ivec = reinterpret(vec, type_tag<i32>);
    if constexpr (idxs.requires_zeroing()) {
      const i32x16 shuf{_mm512_permutex2var_epi32(idxs.vector().r, ivec.r, _mm512_setzero_si512())};
      return reinterpret(shuf, type_tag<Value>);
    } else {
      return reinterpret(i32x16{_mm512_permutexvar_epi32(idxs.vector().r, ivec.r)},
                         type_tag<Value>);
    }
  }
  template<AnyShuffleIndices auto tSh>
  static constexpr std::pair<f64, f64> cost(AutoTag<tSh> /*idxs*/) {
    return {1.0, 5.0};
  }
};
struct ShufflerPermutexVar16x32 : public BaseExpensiveOp {
  template<AnyShuffleIndices auto tSh>
  static constexpr bool is_applicable(AutoTag<tSh> /*tag*/) {
    const std::optional<ShuffleIndices<2, 32>> base = convert<2>(tSh);
    return base.has_value() && !base->subzero;
  }
  template<AnyVector TVec, ShuffleIndicesFor<TVec> tSh>
  static TVec apply(TVec vec, AutoTag<tSh> /*tag*/) {
    using Value = TVec::Value;
    static constexpr ShuffleIndices<2, 32> idxs = convert<2>(tSh).value();
    static_assert(!idxs.subzero);

    const i16x32 ivec = reinterpret(vec, type_tag<i16>);
    if constexpr (idxs.requires_zeroing()) {
      const i16x32 shuf{_mm512_permutex2var_epi16(idxs.vector().r, ivec.r, _mm512_setzero_si512())};
      return reinterpret(shuf, type_tag<Value>);
    } else {
      return reinterpret(i16x32{_mm512_permutexvar_epi16(idxs.vector().r, ivec.r)},
                         type_tag<Value>);
    }
  }
  template<AnyShuffleIndices auto tSh>
  static constexpr std::pair<f64, f64> cost(AutoTag<tSh> /*idxs*/) {
    // for some reason, this is especially slow on some Intel architectures (e.g. Tigerlake)…
    return {1.0, 7.0};
  }
};
#if GREX_HAS_AVX512VBMI
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

    const i8x64 ivec = reinterpret(vec, type_tag<i8>);
    if constexpr (idxs.requires_zeroing()) {
      const i8x64 shuf{_mm512_permutex2var_epi8(idxs.vector().r, ivec.r, _mm512_setzero_si512())};
      return reinterpret(shuf, type_tag<Value>);
    } else {
      return reinterpret(i8x64{_mm512_permutexvar_epi8(idxs.vector().r, ivec.r)}, type_tag<Value>);
    }
  }
  template<AnyShuffleIndices auto tSh>
  static constexpr std::pair<f64, f64> cost(AutoTag<tSh> /*idxs*/) {
    return {1.0, 6.0};
  }
};
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
    // even indices
    auto f8 = [&](std::size_t i, bool even) {
      const auto sh = tSh[i];
      return ((i % 2 == 0) == even && is_index(sh)) ? i8(i + u8(sh) % 2) : i8(-1);
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
#endif

template<AnyShuffleIndices auto tIdxs>
requires((tIdxs.value_size * tIdxs.size == 64))
struct ShufflerTrait<tIdxs> {
  using Shuffler = CheapestType<tIdxs, ShufflerBlendZero, ShufflerShuffle128x4, ShufflerShuffle8x64,
                                ShufflerShuffle32x16, ShufflerPermutex64x8, ShufflerPermutexVar64x8,
                                ShufflerPermutexVar32x16, ShufflerPermutexVar16x32,
                                ShufflerPermutexVar8x64, ShufflerExtractSet>;
};
} // namespace grex::backend
#endif

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_SHUFFLE_STATIC_512_HPP
