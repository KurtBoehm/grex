// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_NEON_OPERATIONS_BLEND_STATIC_HPP
#define INCLUDE_GREX_BACKEND_NEON_OPERATIONS_BLEND_STATIC_HPP

#include <array>
#include <cstddef>
#include <utility>

#include "grex/backend/defs.hpp" // IWYU pragma: keep
#include "grex/backend/neon/operations/load.hpp"
#include "grex/backend/neon/operations/mask-convert.hpp"
#include "grex/backend/shared/defs.hpp"
#include "grex/backend/shared/operations/blend-static.hpp"
#include "grex/base.hpp"

namespace grex::backend {
// TODO Add more efficient operations, for instance:
// - Taking just one value from one of the vector
struct BlenderVariable : public BaseExpensiveOp {
  template<AnyBlendSelectors auto tBls>
  static constexpr bool is_applicable(AutoTag<tBls> /*tag*/) {
    return true;
  }
  template<AnyVector TVec, BlendSelectorsFor<TVec> tBls>
  static TVec apply(TVec a, TVec b, AutoTag<tBls> /*tag*/) {
    using Value = TVec::Value;
    static constexpr std::size_t size = TVec::size;
    using Int = SignedInt<sizeof(Value)>;
    using IVec = Vector<Int, size>;
    using VMask = Mask<Value, size>;

    static constexpr std::array<Int, size> mask_idxs =
      static_apply<size>([]<std::size_t... tIdxs>() {
        return std::array<Int, size>{((tBls[tIdxs] == rhs_bl) ? Int(-1) : Int(0))...};
      });

    const VMask mask = vector2mask(load(mask_idxs.data(), type_tag<IVec>), type_tag<Value>);
    return blend(mask, a, b);
  }
  static constexpr std::pair<f64, f64> cost(auto /*bzs*/) {
    return {1.0, 8};
  }
};

template<AnyBlendSelectors auto tBls>
requires((tBls.value_size * tBls.size == 16))
struct BlenderTrait<tBls> {
  using Type = CheapestType<tBls, BlenderConstant, BlenderVariable>;
};
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_BLEND_STATIC_HPP
