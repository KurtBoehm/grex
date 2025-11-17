// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_BLEND_STATIC_128_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_BLEND_STATIC_128_HPP

#include <utility>

#include "grex/backend/defs.hpp"
#include "grex/backend/shared/defs.hpp"
#include "grex/backend/shared/operations/blend-static.hpp"
#include "grex/backend/x86/instruction-sets.hpp"
#include "grex/backend/x86/operations/blend-static/shared.hpp"
#include "grex/backend/x86/operations/reinterpret.hpp"
#include "grex/backend/x86/types.hpp"
#include "grex/base/defs.hpp"

// TODO There are no optimized implementations at level 1, as those would require a lot of work
//      for very old architectures.
// TODO Add insertps?

namespace grex::backend {
struct BlenderBlend32x4 : public BaseExpensiveOp {
  template<AnyBlendSelectors auto tBls>
  static constexpr bool is_applicable(AutoTag<tBls> /*tag*/) {
#if GREX_X86_64_LEVEL >= 2
    return convert<4>(tBls).has_value();
#else
    return false;
#endif
  }
  template<AnyVector TVec, BlendSelectorsFor<TVec> tBls>
  static TVec apply(TVec a, TVec b, AutoTag<tBls> /*tag*/) {
    using Value = TVec::Value;
    static constexpr int imm8 = convert<4>(tBls).value().imm8();
    const f32x4 fa = reinterpret(a, type_tag<f32>);
    const f32x4 fb = reinterpret(b, type_tag<f32>);
    return reinterpret(f32x4{_mm_blend_ps(fa.r, fb.r, imm8)}, type_tag<Value>);
  }
  static constexpr std::pair<f64, f64> cost(auto /*bzs*/) {
    return {0.5, 1};
  }
};

struct BlenderBlend16x8 : public BaseExpensiveOp {
  template<AnyBlendSelectors auto tBls>
  static constexpr bool is_applicable(AutoTag<tBls> /*tag*/) {
#if GREX_X86_64_LEVEL >= 2
    return convert<2>(tBls).has_value();
#else
    return false;
#endif
  }
  template<AnyVector TVec, BlendSelectorsFor<TVec> tBls>
  static TVec apply(TVec a, TVec b, AutoTag<tBls> /*tag*/) {
    using Value = TVec::Value;
    static constexpr int imm8 = convert<2>(tBls).value().imm8();
    const i16x8 ia = reinterpret(a, type_tag<i16>);
    const i16x8 ib = reinterpret(b, type_tag<i16>);
    return reinterpret(i16x8{_mm_blend_epi16(ia.r, ib.r, imm8)}, type_tag<Value>);
  }
  static constexpr std::pair<f64, f64> cost(auto /*bzs*/) {
    return {0.5, 1};
  }
};

template<AnyBlendSelectors auto tBls>
requires((tBls.value_size * tBls.size == 16))
struct BlenderTrait<tBls> {
  using Type =
    CheapestType<tBls, BlenderConstant, BlenderBlend32x4, BlenderBlend16x8, BlenderVariable>;
};
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_BLEND_STATIC_128_HPP
