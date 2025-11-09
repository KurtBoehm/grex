// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_SHUFFLE_STATIC_256_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_SHUFFLE_STATIC_256_HPP

#include "grex/backend/x86/instruction-sets.hpp"

#if GREX_X86_64_LEVEL >= 3
#include <algorithm>
#include <optional>
#include <utility>

#include "grex/backend/defs.hpp"
#include "grex/backend/x86/defs.hpp"
#include "grex/backend/x86/operations/blend-zero-static.hpp"
#include "grex/backend/x86/operations/load.hpp"
#include "grex/backend/x86/operations/shuffle-static/base.hpp"
#include "grex/backend/x86/operations/shuffle-static/shared.hpp"
#include "grex/backend/x86/types.hpp"
#include "grex/base/defs.hpp"

namespace grex::backend {
struct ShufflerShuffle8x32 : public BaseExpensiveOp {
  template<AnyShuffleIndices auto tSh>
  static constexpr bool is_applicable(AutoTag<tSh> /*tag*/) {
    return tSh.is_lane_local();
  }
  template<AnyVector TVec, ShuffleIndicesFor<TVec> tSh>
  static TVec apply(TVec vec, AutoTag<tSh> /*tag*/) {
    using Value = TVec::Value;
    static constexpr ShuffleIndices<1, 32> shi = convert<1>(tSh).value();
    static constexpr auto idxs = shi.laned_indices().value();

    const i8x32 ivec = reinterpret(vec, type_tag<i8>);
    const auto shuf = _mm256_shuffle_epi8(ivec.r, load(idxs.data(), type_tag<i8x32>).r);
    return reinterpret(i8x32{shuf}, type_tag<Value>);
  }
  template<AnyShuffleIndices auto tSh>
  static constexpr std::pair<f64, f64> cost(AutoTag<tSh> /*idxs*/) {
    return {0.5, 4};
  }
};

struct ShufflerShuffle32x8 : public BaseExpensiveOp {
  template<AnyShuffleIndices auto tSh>
  static constexpr bool is_applicable(AutoTag<tSh> /*tag*/) {
    const std::optional<ShuffleIndices<4, 8>> base = convert<4>(tSh);
    if (!base.has_value()) {
      return false;
    }
    return base.value().single_lane().has_value();
  }
  template<AnyVector TVec, ShuffleIndicesFor<TVec> tSh>
  static TVec apply(TVec vec, AutoTag<tSh> /*tag*/) {
    using Value = TVec::Value;
    static constexpr int imm8 = convert<4>(tSh).value().single_lane().value().imm8();

    const i32x8 ivec = reinterpret(vec, type_tag<i32>);
    const TVec shuffled = reinterpret(i32x8{_mm256_shuffle_epi32(ivec.r, imm8)}, type_tag<Value>);
    return ZeroBlender<tSh.blend_zeros()>::apply(shuffled, auto_tag<tSh.blend_zeros()>);
  }
  template<AnyShuffleIndices auto tSh>
  static constexpr std::pair<f64, f64> cost(AutoTag<tSh> /*idxs*/) {
    const auto [c0, c1] = ZeroBlender<tSh.blend_zeros()>::cost(auto_tag<tSh>);
    return {0.5 + c0, std::max<f64>(c1, 1)};
  }
};

struct ShufflerPermute64x4 : public BaseExpensiveOp {
  template<AnyShuffleIndices auto tSh>
  static constexpr bool is_applicable(AutoTag<tSh> /*tag*/) {
    return convert<8>(tSh).has_value();
  }
  template<AnyVector TVec, ShuffleIndicesFor<TVec> tSh>
  static TVec apply(TVec vec, AutoTag<tSh> /*tag*/) {
    using Value = TVec::Value;
    static constexpr int imm8 = convert<8>(tSh).value().imm8();

    const i64x4 ivec = reinterpret(vec, type_tag<i64>);
    const TVec shuffled =
      reinterpret(i64x4{_mm256_permute4x64_epi64(ivec.r, imm8)}, type_tag<Value>);
    return ZeroBlender<tSh.blend_zeros()>::apply(shuffled, auto_tag<tSh.blend_zeros()>);
  }
  template<AnyShuffleIndices auto tSh>
  static constexpr std::pair<f64, f64> cost(AutoTag<tSh> /*idxs*/) {
    const auto [c0, c1] = ZeroBlender<tSh.blend_zeros()>::cost(auto_tag<tSh>);
    return {1.0 + c0, std::max<f64>(4, c1)};
  }
};

struct ShufflerShuffle8x32Ext : public BaseExpensiveOp {
  template<AnyShuffleIndices auto tSh>
  static constexpr bool is_applicable(AutoTag<tSh> /*tag*/) {
    return true;
  }
  template<AnyVector TVec, ShuffleIndicesFor<TVec> tSh>
  static TVec apply(TVec vec, AutoTag<tSh> /*tag*/) {
    using Value = TVec::Value;
    static constexpr ShuffleIndices<1, 32> shi = convert<1>(tSh).value();
    static constexpr auto idxs0 = shi.intralane_indices();
    static constexpr auto idxs1 = shi.extralane_indices();

    const i8x32 ivec = reinterpret(vec, type_tag<i8>);
    const auto rev = _mm256_permute4x64_epi64(ivec.r, 0b01001110);
    const auto shuf0 = _mm256_shuffle_epi8(ivec.r, load(idxs0.data(), type_tag<i8x32>).r);
    const auto shuf1 = _mm256_shuffle_epi8(rev, load(idxs1.data(), type_tag<i8x32>).r);
    return reinterpret(i8x32{_mm256_or_si256(shuf0, shuf1)}, type_tag<Value>);
  }
  template<AnyShuffleIndices auto tSh>
  static constexpr std::pair<f64, f64> cost(AutoTag<tSh> /*idxs*/) {
    return {2.0, 4};
  }
};

template<AnyShuffleIndices auto tIdxs>
requires((tIdxs.value_size * tIdxs.size == 32))
struct ShufflerTrait<tIdxs> {
  using Shuffler = CheapestType<tIdxs, ShufflerBlendZero, ShufflerShuffle8x32, ShufflerShuffle32x8,
                                ShufflerPermute64x4, ShufflerShuffle8x32Ext>;
};
} // namespace grex::backend
#endif

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_SHUFFLE_STATIC_256_HPP
