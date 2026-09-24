// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_NEON_OPERATIONS_SHUFFLE_STATIC_HPP
#define INCLUDE_GREX_BACKEND_NEON_OPERATIONS_SHUFFLE_STATIC_HPP

#include <cstddef>

#include "grex/backend/base.hpp"
#include "grex/backend/defs.hpp" // IWYU pragma: keep
#include "grex/backend/neon/operations/extract.hpp"
#include "grex/backend/neon/operations/set.hpp"
#include "grex/backend/shared/defs.hpp"
#include "grex/backend/shared/operations/shuffle-static.hpp"
#include "grex/base.hpp"

namespace grex::backend {
struct ShufflerTbl : public BaseExpensiveOp {
  template<AnyShuffleIndices auto SI>
  static constexpr bool is_applicable(AutoTag<SI> /*tag*/) {
    return true;
  }
  template<AnyVector Vec, ShuffleIndicesFor<Vec> SI>
  static Vec apply(Vec vec, AutoTag<SI> /*tag*/) {
    using Value = Vec::Value;
    static constexpr auto shuf = convert<1>(SI).value();
    return {.r = as<Value>(vqtbl1q_u8(as<u8>(vec.r), shuf.vector(false_tag).r))};
  }
  template<AnyShuffleIndices auto SI>
  static constexpr Cost cost(AutoTag<SI> /*idxs*/) {
    return {.inv_throughput = 1, .latency = 8};
  }
};

struct ShufflerExtractSet : public BaseExpensiveOp {
  template<AnyShuffleIndices auto SI>
  static constexpr bool is_applicable(AutoTag<SI> /*tag*/) {
    return true;
  }
  template<AnyVector Vec, ShuffleIndicesFor<Vec> SI>
  static Vec apply(Vec vec, AutoTag<SI> /*tag*/) {
    using Value = Vec::Value;
    static constexpr std::size_t size = Vec::size;

    auto f = [&](std::size_t i) { return is_index(SI[i]) ? extract(vec, u8(SI[i])) : Value{}; };
    return static_apply<size>([&]<std::size_t... I> { return set(type_tag<Vec>, f(I)...); });
  }
  template<AnyShuffleIndices auto SI>
  static constexpr Cost cost(AutoTag<SI> /*idxs*/) {
    return {.inv_throughput = 1, .latency = 8};
  }
};

template<AnyShuffleIndices auto I>
requires((I.value_size * I.size == 16)) // NOLINT(*-redundant-parentheses)
struct ShufflerTrait<I> {
  using Shuffler = CheapestType<I, ShufflerBlendZero, ShufflerTbl, ShufflerExtractSet>;
};
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_SHUFFLE_STATIC_HPP
