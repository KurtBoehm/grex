// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_BLEND_ZERO_STATIC_256_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_BLEND_ZERO_STATIC_256_HPP

#include "grex/backend/x86/instruction-sets.hpp"

// TODO This does not include some of more specialized operations such as movq
#if GREX_X86_64_LEVEL >= 3
#include <cstddef>
#include <utility>

#include "grex/backend/base.hpp"
#include "grex/backend/shared/defs.hpp"
#include "grex/backend/shared/operations/blend-zero-static.hpp"
#include "grex/backend/x86/operations/blend-zero-static/shared.hpp"
#include "grex/backend/x86/operations/reinterpret.hpp"
#include "grex/backend/x86/types.hpp"
#include "grex/base.hpp"

namespace grex::backend {
struct ZeroBlenderBlend32x8 : public BaseExpensiveOp {
  template<AnyBlendZeroSelectors auto BZS>
  static constexpr bool is_applicable(AutoTag<BZS> /*tag*/) {
    return convert<4>(BZS).has_value();
  }
  template<AnyVector Vec, BlendZeroSelectorsFor<Vec> BZS>
  static Vec apply(Vec vec, AutoTag<BZS> /*tag*/) {
    using Value = Vec::Value;
    static constexpr int imm8 = convert<4>(BZS).value().imm8();
    const f32x8 fvec = reinterpret(vec, type_tag<f32>);
    return reinterpret(f32x8{_mm256_blend_ps(_mm256_setzero_ps(), fvec.r, imm8)}, type_tag<Value>);
  }
  static constexpr Cost cost(auto /*bzs*/) {
    return {.inv_throughput = 0.5, .latency = 2};
  }
};

struct ZeroBlenderBlend16x16 : public BaseExpensiveOp {
  template<AnyBlendZeroSelectors auto BZS>
  static constexpr bool is_applicable(AutoTag<BZS> /*tag*/) {
    const auto base = convert<2>(BZS);
    if (!base.has_value()) {
      return false;
    }
    return base.value().single_lane().has_value();
  }
  template<AnyVector Vec, BlendZeroSelectorsFor<Vec> BZS>
  static Vec apply(Vec vec, AutoTag<BZS> /*tag*/) {
    using Value = Vec::Value;
    static constexpr int imm8 = convert<2>(BZS).value().single_lane().value().imm8();
    const i16x16 ivec = reinterpret(vec, type_tag<i16>);
    return reinterpret(i16x16{_mm256_blend_epi16(_mm256_setzero_si256(), ivec.r, imm8)},
                       type_tag<Value>);
  }
  static constexpr Cost cost(auto /*bzs*/) {
    return {.inv_throughput = 0.5, .latency = 2};
  }
};

template<AnyBlendZeroSelectors auto BZS>
requires((BZS.value_size * BZS.size == 32))
struct ZeroBlenderTrait<BZS> {
  using Type = CheapestType<BZS, ZeroBlenderNoop, ZeroBlenderZero, ZeroBlenderBlend32x8,
                            ZeroBlenderBlend16x16, ZeroBlenderAnd>;
};
} // namespace grex::backend
#endif

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_BLEND_ZERO_STATIC_256_HPP
