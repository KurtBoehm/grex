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
struct ShufflerExtractSet : public BaseExpensiveOp {
  template<AnyShuffleIndices auto tSh>
  static constexpr bool is_applicable(AutoTag<tSh> /*tag*/) {
    return true;
  }
  template<AnyVector TVec, ShuffleIndicesFor<TVec> tSh>
  static TVec apply(TVec vec, AutoTag<tSh> /*tag*/) {
    using Value = TVec::Value;
    static constexpr std::size_t size = TVec::size;

    auto f = [&](std::size_t i) { return is_index(tSh[i]) ? extract(vec, u8(tSh[i])) : Value{}; };
    return static_apply<size>(
      [&]<std::size_t... tIdxs>() { return set(type_tag<TVec>, f(tIdxs)...); });
  }
  template<AnyShuffleIndices auto tSh>
  static constexpr std::pair<f64, f64> cost(AutoTag<tSh> /*idxs*/) {
    return {f64(tSh.size * 2), 1};
  }
};
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_SHUFFLE_STATIC_SHARED_HPP
