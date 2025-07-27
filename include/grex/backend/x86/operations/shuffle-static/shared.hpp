// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_SHUFFLE_STATIC_SHARED_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_SHUFFLE_STATIC_SHARED_HPP

#include <cstddef>
#include <utility>

#include "grex/backend/defs.hpp"
#include "grex/backend/x86/defs.hpp"
#include "grex/backend/x86/operations/blend-zero-static/base.hpp"
#include "grex/backend/x86/operations/extract.hpp"
#include "grex/backend/x86/operations/set.hpp"
#include "grex/backend/x86/operations/shuffle-static/base.hpp"
#include "grex/base/defs.hpp"

namespace grex::backend {
struct ShufflerExtractSet : public BaseExpensiveOp {
  template<AnyShuffleIndices auto tIdxs>
  static constexpr bool is_applicable(AutoTag<tIdxs> /*tag*/) {
    return true;
  }
  template<AnyVector TVec, TypedValueTag<ShuffleIndicesFor<TVec>> TTag>
  static TVec apply(TVec vec, TTag /*tag*/) {
    using Value = TVec::Value;
    static constexpr ShuffleIndicesFor<TVec> bzs = TTag::value;
    static constexpr std::size_t size = TVec::size;

    auto f = [&](std::size_t i) { return is_index(bzs[i]) ? extract(vec, u8(bzs[i])) : Value{}; };
    return static_apply<size>(
      [&]<std::size_t... tIdxs>() { return set(type_tag<TVec>, f(tIdxs)...); });
  }
  template<AnyShuffleIndices auto tIdxs>
  static constexpr std::pair<f64, f64> cost(AutoTag<tIdxs> /*idxs*/) {
    return {f64(tIdxs.size * 2), 1};
  }
};

struct ShufflerBlendZero : public BaseExpensiveOp {
  template<AnyShuffleIndices auto tSh>
  static constexpr bool is_applicable(AutoTag<tSh> /*tag*/) {
    return static_apply<tSh.size>([]<std::size_t... tIdxs>() {
      return (... && (!is_index(tSh[tIdxs]) || u8(tSh[tIdxs]) == tIdxs));
    });
  }
  template<AnyVector TVec, ShuffleIndicesFor<TVec> tSh>
  static TVec apply(TVec vec, AutoTag<tSh> /*tag*/) {
    static_assert(is_applicable(auto_tag<tSh>));
    return ZeroBlender<tSh.blend_zeros()>::apply(vec, auto_tag<tSh.blend_zeros()>);
  }
  template<AnyShuffleIndices auto tSh>
  static constexpr std::pair<f64, f64> cost(AutoTag<tSh> /*idxs*/) {
    static_assert(is_applicable(auto_tag<tSh>));
    return ZeroBlender<tSh.blend_zeros()>::cost(auto_tag<tSh>);
  }
};
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_SHUFFLE_STATIC_SHARED_HPP
