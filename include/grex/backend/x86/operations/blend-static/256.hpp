// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_BLEND_STATIC_256_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_BLEND_STATIC_256_HPP

#include "grex/backend/x86/instruction-sets.hpp"

#if GREX_X86_64_LEVEL >= 3
#include <cstddef>
#include <utility>

#include "grex/backend/base.hpp"
#include "grex/backend/shared/defs.hpp"
#include "grex/backend/shared/operations/blend-static.hpp"
#include "grex/backend/x86/operations/blend-static/shared.hpp"
#include "grex/backend/x86/operations/reinterpret.hpp"
#include "grex/backend/x86/types.hpp"
#include "grex/base.hpp"

namespace grex::backend {
struct BlenderBlend32x8 : public BaseExpensiveOp {
  template<AnyBlendSelectors auto tBls>
  static constexpr bool is_applicable(AutoTag<tBls> /*tag*/) {
    return convert<4>(tBls).has_value();
  }
  template<AnyVector TVec, BlendSelectorsFor<TVec> tBls>
  static TVec apply(TVec a, TVec b, AutoTag<tBls> /*tag*/) {
    using Value = TVec::Value;
    static constexpr int imm8 = convert<4>(tBls).value().imm8();
    const f32x8 fa = reinterpret(a, type_tag<f32>);
    const f32x8 fb = reinterpret(b, type_tag<f32>);
    return reinterpret(f32x8{_mm256_blend_ps(fa.r, fb.r, imm8)}, type_tag<Value>);
  }
  static constexpr std::pair<f64, f64> cost(auto /*bzs*/) {
    return {0.5, 1};
  }
};

struct BlenderBlend16x16 : public BaseExpensiveOp {
  template<AnyBlendSelectors auto tBls>
  static constexpr bool is_applicable(AutoTag<tBls> /*tag*/) {
    const auto base = convert<2>(tBls);
    if (!base.has_value()) {
      return false;
    }
    return base.value().single_lane().has_value();
  }
  template<AnyVector TVec, BlendSelectorsFor<TVec> tBls>
  static TVec apply(TVec a, TVec b, AutoTag<tBls> /*tag*/) {
    using Value = TVec::Value;
    static constexpr int imm8 = convert<2>(tBls).value().single_lane().value().imm8();
    const i16x16 ia = reinterpret(a, type_tag<i16>);
    const i16x16 ib = reinterpret(b, type_tag<i16>);
    return reinterpret(i16x16{_mm256_blend_epi16(ia.r, ib.r, imm8)}, type_tag<Value>);
  }
  static constexpr std::pair<f64, f64> cost(auto /*bzs*/) {
    return {0.5, 1};
  }
};

template<AnyBlendSelectors auto tBls>
requires((tBls.value_size * tBls.size == 32))
struct BlenderTrait<tBls> {
  using Type =
    CheapestType<tBls, BlenderConstant, BlenderBlend32x8, BlenderBlend16x16, BlenderVariable>;
};
} // namespace grex::backend
#endif

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_BLEND_STATIC_256_HPP
