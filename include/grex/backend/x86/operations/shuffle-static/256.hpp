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

#include "grex/backend/base.hpp"
#include "grex/backend/shared/defs.hpp"
#include "grex/backend/shared/operations/shuffle-static.hpp"
#include "grex/backend/x86/operations/blend-zero-static.hpp"
#include "grex/backend/x86/operations/load.hpp"
#include "grex/backend/x86/types.hpp"
#include "grex/base.hpp"

namespace grex::backend {
struct ShufflerShuffle8x32 : public BaseExpensiveOp {
  template<AnyShuffleIndices auto SI>
  static constexpr bool is_applicable(AutoTag<SI> /*tag*/) {
    return SI.is_lane_local();
  }
  template<AnyVector Vec, ShuffleIndicesFor<Vec> SI>
  static Vec apply(Vec vec, AutoTag<SI> /*tag*/) {
    using Value = ValueOf<Vec>;
    static constexpr ShuffleIndices<1, 32> shi = convert<1>(SI).value();
    static constexpr auto idxs = shi.laned_indices().value();

    const i8x32 ivec = reinterpret(vec, type_tag<i8>);
    const auto shuf = _mm256_shuffle_epi8(ivec.r, load(idxs.data(), type_tag<i8x32>).r);
    return reinterpret(i8x32{shuf}, type_tag<Value>);
  }
  template<AnyShuffleIndices auto SI>
  static constexpr Cost cost(AutoTag<SI> /*idxs*/) {
    return {.inv_throughput = 0.5, .latency = 4};
  }
};

struct ShufflerShuffle32x8 : public BaseExpensiveOp {
  template<AnyShuffleIndices auto SI>
  static constexpr bool is_applicable(AutoTag<SI> /*tag*/) {
    const std::optional<ShuffleIndices<4, 8>> base = convert<4>(SI);
    if (!base.has_value()) {
      return false;
    }
    return base.value().single_lane().has_value();
  }
  template<AnyVector Vec, ShuffleIndicesFor<Vec> SI>
  static Vec apply(Vec vec, AutoTag<SI> /*tag*/) {
    using Value = ValueOf<Vec>;
    static constexpr int imm8 = convert<4>(SI).value().single_lane().value().imm8();

    const i32x8 ivec = reinterpret(vec, type_tag<i32>);
    const Vec shuffled = reinterpret(i32x8{_mm256_shuffle_epi32(ivec.r, imm8)}, type_tag<Value>);
    return ZeroBlender<SI.blend_zeros()>::apply(shuffled, auto_tag<SI.blend_zeros()>);
  }
  template<AnyShuffleIndices auto SI>
  static constexpr Cost cost(AutoTag<SI> /*idxs*/) {
    const auto [c0, c1] = ZeroBlender<SI.blend_zeros()>::cost(auto_tag<SI>);
    return {.inv_throughput = 0.5 + c0, .latency = std::max<f64>(c1, 1)};
  }
};

struct ShufflerPermute64x4 : public BaseExpensiveOp {
  template<AnyShuffleIndices auto SI>
  static constexpr bool is_applicable(AutoTag<SI> /*tag*/) {
    return convert<8>(SI).has_value();
  }
  template<AnyVector Vec, ShuffleIndicesFor<Vec> SI>
  static Vec apply(Vec vec, AutoTag<SI> /*tag*/) {
    using Value = ValueOf<Vec>;
    static constexpr int imm8 = convert<8>(SI).value().imm8();

    const i64x4 ivec = reinterpret(vec, type_tag<i64>);
    const Vec shuffled =
      reinterpret(i64x4{_mm256_permute4x64_epi64(ivec.r, imm8)}, type_tag<Value>);
    return ZeroBlender<SI.blend_zeros()>::apply(shuffled, auto_tag<SI.blend_zeros()>);
  }
  template<AnyShuffleIndices auto SI>
  static constexpr Cost cost(AutoTag<SI> /*idxs*/) {
    const auto [c0, c1] = ZeroBlender<SI.blend_zeros()>::cost(auto_tag<SI>);
    return {.inv_throughput = 1 + c0, .latency = std::max<f64>(4, c1)};
  }
};

struct ShufflerShuffle8x32Ext : public BaseExpensiveOp {
  template<AnyShuffleIndices auto SI>
  static constexpr bool is_applicable(AutoTag<SI> /*tag*/) {
    return true;
  }
  template<AnyVector Vec, ShuffleIndicesFor<Vec> SI>
  static Vec apply(Vec vec, AutoTag<SI> /*tag*/) {
    using Value = ValueOf<Vec>;
    static constexpr ShuffleIndices<1, 32> shi = convert<1>(SI).value();
    static constexpr auto idxs0 = shi.intralane_indices();
    static constexpr auto idxs1 = shi.extralane_indices();

    const i8x32 ivec = reinterpret(vec, type_tag<i8>);
    const auto rev = _mm256_permute4x64_epi64(ivec.r, 0b01001110);
    const auto shuf0 = _mm256_shuffle_epi8(ivec.r, load(idxs0.data(), type_tag<i8x32>).r);
    const auto shuf1 = _mm256_shuffle_epi8(rev, load(idxs1.data(), type_tag<i8x32>).r);
    return reinterpret(i8x32{_mm256_or_si256(shuf0, shuf1)}, type_tag<Value>);
  }
  template<AnyShuffleIndices auto SI>
  static constexpr Cost cost(AutoTag<SI> /*idxs*/) {
    return {.inv_throughput = 2, .latency = 4};
  }
};

template<AnyShuffleIndices auto I>
requires((I.value_size * I.size == 32))
struct ShufflerTrait<I> {
  using Shuffler = CheapestType<I, ShufflerBlendZero, ShufflerShuffle8x32, ShufflerShuffle32x8,
                                ShufflerPermute64x4, ShufflerShuffle8x32Ext>;
};
} // namespace grex::backend
#endif

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_SHUFFLE_STATIC_256_HPP
