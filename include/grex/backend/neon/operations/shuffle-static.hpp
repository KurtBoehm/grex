// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_NEON_OPERATIONS_SHUFFLE_STATIC_HPP
#define INCLUDE_GREX_BACKEND_NEON_OPERATIONS_SHUFFLE_STATIC_HPP

#include <cstddef>
#include <utility>

#include "grex/backend/defs.hpp"
#include "grex/backend/neon/operations/extract.hpp"
#include "grex/backend/neon/operations/set.hpp"
#include "grex/backend/shared/defs.hpp"
#include "grex/backend/shared/operations/shuffle-static.hpp"
#include "grex/base.hpp"

namespace grex::backend {
struct ShufflerTbl : public BaseExpensiveOp {
  template<AnyShuffleIndices auto tSh>
  static constexpr bool is_applicable(AutoTag<tSh> /*tag*/) {
    return true;
  }
  template<AnyVector TVec, ShuffleIndicesFor<TVec> tSh>
  static TVec apply(TVec vec, AutoTag<tSh> /*tag*/) {
    using Value = TVec::Value;
    static constexpr auto shuf = convert<1>(tSh).value();
    return {.r = as<Value>(vqtbl1q_u8(as<u8>(vec.r), shuf.vector(false_tag).r))};
  }
  template<AnyShuffleIndices auto tSh>
  static constexpr std::pair<f64, f64> cost(AutoTag<tSh> /*idxs*/) {
    return {1.0, 8};
  }
};

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
    return {1.0, 8};
  }
};

template<AnyShuffleIndices auto tIdxs>
requires((tIdxs.value_size * tIdxs.size == 16))
struct ShufflerTrait<tIdxs> {
  using Shuffler = CheapestType<tIdxs, ShufflerBlendZero, ShufflerTbl, ShufflerExtractSet>;
};
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_SHUFFLE_STATIC_HPP
