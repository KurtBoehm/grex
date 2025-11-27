// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_NEON_OPERATIONS_BLEND_ZERO_STATIC_HPP
#define INCLUDE_GREX_BACKEND_NEON_OPERATIONS_BLEND_ZERO_STATIC_HPP

#include <array>
#include <cstddef>
#include <utility>

#include "grex/backend/defs.hpp"
#include "grex/backend/neon/operations/bitwise.hpp"
#include "grex/backend/neon/operations/load.hpp"
#include "grex/backend/neon/operations/reinterpret.hpp"
#include "grex/backend/shared/defs.hpp"
#include "grex/backend/shared/operations/blend-zero-static.hpp"
#include "grex/base.hpp"

namespace grex::backend {
// TODO Add more efficient operations, for instance:
// - Inserting one zero
// - Inserting one value into zeros
// - A contiguous ranges of non-zeros/zeros
struct ZeroBlenderAnd : public BaseExpensiveOp {
  template<AnyBlendZeroSelectors auto tBzs>
  static constexpr bool is_applicable(AutoTag<tBzs> /*tag*/) {
    return true;
  }
  template<AnyVector TVec, BlendZeroSelectorsFor<TVec> tBzs>
  static TVec apply(TVec vec, AutoTag<tBzs> /*tag*/) {
    using Value = TVec::Value;
    static constexpr std::size_t size = TVec::size;
    using Int = SignedInt<sizeof(Value)>;
    using IVec = Vector<Int, size>;
    static constexpr std::array<Int, size> mask_idxs =
      static_apply<size>([]<std::size_t... tIdxs>() {
        return std::array<Int, size>{((tBzs[tIdxs] == keep_bz) ? Int(-1) : Int(0))...};
      });

    const IVec ivec = {.r = reinterpret(vec.r, type_tag<Int>)};
    const IVec mask = load(mask_idxs.data(), type_tag<IVec>);
    return {.r = reinterpret(bitwise_and(ivec, mask).r, type_tag<Value>)};
  }
  static constexpr std::pair<f64, f64> cost(auto /*bzs*/) {
    return {1.0, 8};
  }
};

template<AnyBlendZeroSelectors auto tBzs>
requires((tBzs.value_size * tBzs.size == 16))
struct ZeroBlenderTrait<tBzs> {
  using Type = CheapestType<tBzs, ZeroBlenderNoop, ZeroBlenderZero, ZeroBlenderAnd>;
};
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_BLEND_ZERO_STATIC_HPP
