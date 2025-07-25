// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_BLEND_ZERO_STATIC_256_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_BLEND_ZERO_STATIC_256_HPP

// #define GREX_X86_64_LEVEL 3
#include "grex/backend/x86/instruction-sets.hpp"

// TODO This does not include some of more specialized operations such as movq
#if GREX_X86_64_LEVEL >= 3
#include <cstddef>
#include <utility>

#include "grex/backend/defs.hpp"
#include "grex/backend/x86/defs.hpp"
#include "grex/backend/x86/operations/blend-zero-static/base.hpp"
#include "grex/backend/x86/operations/blend-zero-static/shared.hpp"
#include "grex/backend/x86/operations/reinterpret.hpp"
#include "grex/backend/x86/types.hpp"
#include "grex/base/defs.hpp"

namespace grex::backend {
struct ZeroBlenderBlend32x8 : public BaseExpensiveOp {
  template<AnyBlendZeros auto tBzs>
  static constexpr bool is_applicable(AutoTag<tBzs> /*tag*/) {
    return convert<4>(tBzs).has_value();
  }
  template<AnyVector TVec, BlendZerosFor<TVec> tBzs>
  static TVec apply(TVec vec, AutoTag<tBzs> /*tag*/) {
    using Value = TVec::Value;
    static constexpr auto bzs = convert<4>(tBzs).value();

    const f32x8 fvec = reinterpret(vec, type_tag<f32>);
    static constexpr int imm8 = static_apply<8>([]<std::size_t... tIdxs>() {
      return (0 + ... + (int(bzs[tIdxs] == BlendZero::keep) << tIdxs));
    });
    return reinterpret(f32x8{_mm256_blend_ps(_mm256_setzero_ps(), fvec.r, imm8)}, type_tag<Value>);
  }
  static constexpr std::pair<f64, f64> cost(auto /*bzs*/) {
    return {0.5, 2};
  }
};

struct ZeroBlenderBlend16x16 : public BaseExpensiveOp {
  template<AnyBlendZeros auto tBzs>
  static constexpr bool is_applicable(AutoTag<tBzs> /*tag*/) {
    const auto base = convert<2>(tBzs);
    if (!base.has_value()) {
      return false;
    }
    return base.value().single_lane().has_value();
  }
  template<AnyVector TVec, BlendZerosFor<TVec> tBzs>
  static TVec apply(TVec vec, AutoTag<tBzs> /*tag*/) {
    using Value = TVec::Value;
    static constexpr auto bzs = convert<2>(tBzs).value().single_lane().value();

    const i16x16 ivec = reinterpret(vec, type_tag<i16>);
    static constexpr int imm8 = static_apply<8>([]<std::size_t... tIdxs>() {
      return (0 + ... + (int(bzs[tIdxs] == BlendZero::keep) << tIdxs));
    });
    return reinterpret(i16x16{_mm256_blend_epi16(_mm256_setzero_si256(), ivec.r, imm8)},
                       type_tag<Value>);
  }
  static constexpr std::pair<f64, f64> cost(auto /*bzs*/) {
    return {0.5, 2};
  }
};

template<AnyBlendZeros auto tBzs>
requires((tBzs.value_size * tBzs.size == 32))
struct ZeroBlenderTrait<tBzs> {
  using Type = CheapestType<tBzs, ZeroBlenderNoop, ZeroBlenderZero, ZeroBlenderBlend32x8,
                            ZeroBlenderBlend16x16, ZeroBlenderAnd>;
};
} // namespace grex::backend
#endif

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_BLEND_ZERO_STATIC_256_HPP
