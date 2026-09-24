// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_SHUFFLE_STATIC_128_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_SHUFFLE_STATIC_128_HPP

#include <algorithm>
#include <utility>

#include "grex/backend/base.hpp"
#include "grex/backend/shared/defs.hpp"
#include "grex/backend/shared/operations/blend-zero-static.hpp"
#include "grex/backend/shared/operations/shuffle-static.hpp"
#include "grex/backend/x86/instruction-sets.hpp"
#include "grex/backend/x86/operations/reinterpret.hpp"
#include "grex/backend/x86/operations/shuffle-static/shared.hpp"
#include "grex/backend/x86/types.hpp"
#include "grex/base.hpp"

// TODO Very inefficient for 8- and 16-bit values on level 1, but a more efficient implementation
//      would require more work than I am willing to invest in such old archtectures:
//      Level 2 is ubiquitous since 2013, even on the low end.
// TODO Some less general instructions are not included, such as shift instructions,
//      pshuflw/pshufhw, etc.

namespace grex::backend {
struct ShufflerShuffle8x16 : public BaseExpensiveOp {
  template<AnyShuffleIndices auto SI>
  static constexpr bool is_applicable(AutoTag<SI> /*tag*/) {
    return GREX_X86_64_LEVEL >= 2;
  }
  template<AnyVector Vec, ShuffleIndicesFor<Vec> SI>
  static Vec apply(Vec vec, AutoTag<SI> /*tag*/) {
    using Value = Vec::Value;
    static constexpr ShuffleIndices<1, 16> idxs = convert<1>(SI).value();

    const i8x16 ivec = reinterpret(vec, type_tag<i8>);
    return reinterpret(i8x16{_mm_shuffle_epi8(ivec.r, idxs.vector().r)}, type_tag<Value>);
  }
  template<AnyShuffleIndices auto SI>
  static constexpr Cost cost(AutoTag<SI> /*idxs*/) {
    return {.inv_throughput = 0.5, .latency = 4};
  }
};

struct ShufflerShuffle32x4 : public BaseExpensiveOp {
  template<AnyShuffleIndices auto SI>
  static constexpr bool is_applicable(AutoTag<SI> /*tag*/) {
    return convert<4>(SI).has_value();
  }
  template<AnyVector Vec, ShuffleIndicesFor<Vec> SI>
  static Vec apply(Vec vec, AutoTag<SI> /*tag*/) {
    using Value = Vec::Value;
    static constexpr int imm8 = convert<4>(SI).value().imm8();

    const i32x4 ivec = reinterpret(vec, type_tag<i32>);
    const Vec shuffled = reinterpret(i32x4{_mm_shuffle_epi32(ivec.r, imm8)}, type_tag<Value>);
    return ZeroBlender<SI.blend_zeros()>::apply(shuffled, auto_tag<SI.blend_zeros()>);
  }
  template<AnyShuffleIndices auto SI>
  static constexpr Cost cost(AutoTag<SI> /*idxs*/) {
    const auto [c0, c1] = ZeroBlender<SI.blend_zeros()>::cost(auto_tag<SI>);
    return {.inv_throughput = 0.5 + c0, .latency = std::max<f64>(c1, 1)};
  }
};

template<AnyShuffleIndices auto I>
requires((I.value_size * I.size == 16))
struct ShufflerTrait<I> {
  using Shuffler = CheapestType<I, ShufflerBlendZero, ShufflerShuffle8x16, ShufflerShuffle32x4,
                                ShufflerExtractSet>;
};
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_SHUFFLE_STATIC_128_HPP
