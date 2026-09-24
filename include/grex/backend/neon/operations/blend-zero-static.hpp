// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_NEON_OPERATIONS_BLEND_ZERO_STATIC_HPP
#define INCLUDE_GREX_BACKEND_NEON_OPERATIONS_BLEND_ZERO_STATIC_HPP

#include <array>
#include <cstddef>

#include "grex/backend/base.hpp"
#include "grex/backend/defs.hpp" // IWYU pragma: keep
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
  template<AnyBlendZeroSelectors auto BZS>
  static constexpr bool is_applicable(AutoTag<BZS> /*tag*/) {
    return true;
  }
  template<AnyVector Vec, BlendZeroSelectorsFor<Vec> BZS>
  static Vec apply(Vec vec, AutoTag<BZS> /*tag*/) {
    using Value = Vec::Value;
    static constexpr std::size_t size = Vec::size;
    using Int = SignedInt<sizeof(Value)>;
    using IVec = NativeVector<Int, size>;
    static constexpr std::array<Int, size> mask_idxs = static_apply<size>([]<std::size_t... I> {
      return std::array<Int, size>{((BZS[I] == keep_bz) ? Int(-1) : Int(0))...};
    });

    const IVec ivec = {.r = reinterpret(vec.r, type_tag<Int>)};
    const IVec mask = load(mask_idxs.data(), type_tag<IVec>);
    return {.r = reinterpret(bitwise_and(ivec, mask).r, type_tag<Value>)};
  }
  static constexpr Cost cost(auto /*bzs*/) {
    return {.inv_throughput = 1, .latency = 8};
  }
};

template<AnyBlendZeroSelectors auto BZS>
requires((BZS.value_size * BZS.size == 16)) // NOLINT(*-redundant-parentheses)
struct ZeroBlenderTrait<BZS> {
  using Type = CheapestType<BZS, ZeroBlenderNoop, ZeroBlenderZero, ZeroBlenderAnd>;
};
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_BLEND_ZERO_STATIC_HPP
