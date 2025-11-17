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
#include "grex/backend/shared/defs.hpp"
#include "grex/backend/shared/operations/blend-static.hpp"
#include "grex/backend/x86/instruction-sets.hpp"
#include "grex/backend/x86/operations/blend.hpp"
#include "grex/backend/x86/operations/set.hpp"
#include "grex/base/defs.hpp"

namespace grex::backend {
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
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_BLEND_STATIC_SHARED_HPP
