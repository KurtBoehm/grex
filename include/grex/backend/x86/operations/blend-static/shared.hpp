// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_BLEND_STATIC_SHARED_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_BLEND_STATIC_SHARED_HPP

#include <cstddef>
#include <utility>

#include "grex/backend/defs.hpp"
#include "grex/backend/x86/defs.hpp"
#include "grex/backend/x86/instruction-sets.hpp"
#include "grex/backend/x86/operations/blend-static/base.hpp"
#include "grex/backend/x86/operations/blend.hpp"
#include "grex/backend/x86/operations/set.hpp"
#include "grex/backend/x86/sizes.hpp"
#include "grex/base/defs.hpp"

namespace grex::backend {
struct BlenderConstant : public BaseExpensiveOp {
  template<AnyBlendSelectors auto tBls>
  static constexpr bool is_applicable(AutoTag<tBls> /*tag*/) {
    return tBls.constant().has_value();
  }
  template<AnyVector TVec, BlendSelectorsFor<TVec> tBls>
  static TVec apply(TVec a, TVec b, AutoTag<tBls> /*tag*/) {
    constexpr BlendSelector bl = tBls.constant().value();
    if constexpr (bl == rhs_bl) {
      return b;
    } else {
      return a;
    }
  }
  static constexpr std::pair<f64, f64> cost(auto /*bzs*/) {
    return {0, 0};
  }
};

struct BlenderVariable : public BaseExpensiveOp {
  template<AnyBlendSelectors auto tBls>
  static constexpr bool is_applicable(AutoTag<tBls> /*tag*/) {
    return true;
  }
  template<AnyVector TVec, BlendSelectorsFor<TVec> tBls>
  static TVec apply(TVec a, TVec b, AutoTag<tBls> /*tag*/) {
    using Value = TVec::Value;
    static constexpr std::size_t size = TVec::size;
    using VMask = Mask<Value, size>;

    const VMask mask = static_apply<size>(
      []<std::size_t... tIdxs>() { return set(type_tag<VMask>, (tBls[tIdxs] == rhs_bl)...); });
    return blend(mask, a, b);
  }
  static constexpr std::pair<f64, f64> cost(auto /*bzs*/) {
    return {0.5, 4};
  }
};

struct SubBlender : public BaseExpensiveOp {
  template<AnyBlendSelectors auto tBls>
  using Base = Blender<tBls.sub_extended()>;

  template<AnyBlendSelectors auto tBls>
  static constexpr bool is_applicable(AutoTag<tBls> /*tag*/) {
    return Base<tBls>::is_applicable(auto_tag<tBls.sub_extended()>);
  }
  template<AnyVector TVec, BlendSelectorsFor<TVec> tBls>
  static TVec apply(TVec a, TVec b, AutoTag<tBls> /*tag*/) {
    return TVec{Base<tBls>::apply(a.full, b.full, auto_tag<tBls.sub_extended()>)};
  }
  template<AnyBlendSelectors auto tBls>
  static constexpr std::pair<f64, f64> cost(AutoTag<tBls> /*tag*/) {
    return Base<tBls>::cost(auto_tag<tBls.sub_extended()>);
  }
};
struct SuperBlender : public BaseExpensiveOp {
  template<AnyBlendSelectors auto tBls>
  static constexpr bool is_applicable(AutoTag<tBls> /*tag*/) {
    return Blender<tBls.lower()>::is_applicable(auto_tag<tBls.lower()>) &&
           Blender<tBls.upper()>::is_applicable(auto_tag<tBls.upper()>);
  }
  template<AnyVector TVec, BlendSelectorsFor<TVec> tBls>
  static TVec apply(TVec a, TVec b, AutoTag<tBls> /*tag*/) {
    return TVec{
      .lower = Blender<tBls.lower()>::apply(a.lower, b.lower, auto_tag<tBls.lower()>),
      .upper = Blender<tBls.upper()>::apply(a.upper, b.upper, auto_tag<tBls.upper()>),
    };
  }
  template<AnyBlendSelectors auto tBls>
  static constexpr std::pair<f64, f64> cost(AutoTag<tBls> /*tag*/) {
    const auto [c0a, c1a] = Blender<tBls.lower()>::cost(auto_tag<tBls.lower()>);
    const auto [c0b, c1b] = Blender<tBls.upper()>::cost(auto_tag<tBls.upper()>);
    return {c0a + c0b, c1a + c1b};
  }
};

template<AnyBlendSelectors auto tBls>
requires((tBls.value_size * tBls.size < register_bytes.front()))
struct BlenderTrait<tBls> {
  using Type = SubBlender;
};
template<AnyBlendSelectors auto tBls>
requires((tBls.value_size * tBls.size > register_bytes.back()))
struct BlenderTrait<tBls> {
  using Type = SuperBlender;
};
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_BLEND_STATIC_SHARED_HPP
