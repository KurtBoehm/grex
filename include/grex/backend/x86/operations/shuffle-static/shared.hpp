// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_SHUFFLE_STATIC_SHARED_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_SHUFFLE_STATIC_SHARED_HPP

#include <cstddef>
#include <utility>

#include "grex/backend/base.hpp"
#include "grex/backend/shared/defs.hpp"
#include "grex/backend/shared/operations/shuffle-static.hpp"
#include "grex/backend/x86/instruction-sets.hpp"
#include "grex/backend/x86/operations/extract.hpp"
#include "grex/backend/x86/operations/set.hpp"
#include "grex/base.hpp"

namespace grex::backend {
struct ShufflerExtractSet : BaseExpensiveOp {
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
    return {.inv_throughput = f64(SI.size * 2), .latency = 1};
  }
};
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_SHUFFLE_STATIC_SHARED_HPP
