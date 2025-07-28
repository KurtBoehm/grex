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
#include "grex/backend/x86/instruction-sets.hpp"
#include "grex/backend/x86/operations/blend-zero-static/base.hpp"
#include "grex/backend/x86/operations/extract.hpp"
#include "grex/backend/x86/operations/set.hpp"
#include "grex/backend/x86/operations/shuffle-static/base.hpp"
#include "grex/base/defs.hpp"

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

struct SubShuffler : public BaseExpensiveOp {
  template<AnyShuffleIndices auto tSh>
  using Base = Shuffler<tSh.sub_extended()>;

  template<AnyShuffleIndices auto tSh>
  static constexpr bool is_applicable(AutoTag<tSh> /*tag*/) {
    return Base<tSh>::is_applicable(auto_tag<tSh.sub_extended()>);
  }
  template<AnyVector TVec, ShuffleIndicesFor<TVec> tSh>
  static TVec apply(TVec vec, AutoTag<tSh> /*tag*/) {
    return TVec{Base<tSh>::apply(vec.full, auto_tag<tSh.sub_extended()>)};
  }
  template<AnyShuffleIndices auto tSh>
  static constexpr std::pair<f64, f64> cost(AutoTag<tSh> /*tag*/) {
    return Base<tSh>::cost(auto_tag<tSh.sub_extended()>);
  }
};
struct SuperShuffler : public BaseExpensiveOp {
  template<AnyShuffleIndices auto tSh>
  static constexpr bool is_applicable(AutoTag<tSh> /*tag*/) {
    if constexpr (!tSh.is_half_local()) {
      return false;
    } else {
      return Shuffler<tSh.lower().value()>::is_applicable(auto_tag<tSh.lower().value()>) &&
             Shuffler<tSh.upper().value()>::is_applicable(auto_tag<tSh.upper().value()>);
    }
  }
  template<AnyVector TVec, ShuffleIndicesFor<TVec> tSh>
  static TVec apply(TVec vec, AutoTag<tSh> /*tag*/) {
    return TVec{
      .lower = Shuffler<tSh.lower().value()>::apply(vec.lower, auto_tag<tSh.lower().value()>),
      .upper = Shuffler<tSh.upper().value()>::apply(vec.upper, auto_tag<tSh.upper().value()>),
    };
  }
  template<AnyShuffleIndices auto tSh>
  static constexpr std::pair<f64, f64> cost(AutoTag<tSh> /*tag*/) {
    const auto [c0a, c1a] = Shuffler<tSh.lower().value()>::cost(auto_tag<tSh.lower().value()>);
    const auto [c0b, c1b] = Shuffler<tSh.upper().value()>::cost(auto_tag<tSh.upper().value()>);
    return {c0a + c0b, c1a + c1b};
  }
};

template<AnyShuffleIndices auto tSh>
requires((tSh.value_size * tSh.size < register_bytes.front()))
struct ShufflerTrait<tSh> {
  using Shuffler = SubShuffler;
};
template<AnyShuffleIndices auto tSh>
requires((tSh.value_size * tSh.size > register_bytes.back()))
struct ShufflerTrait<tSh> {
  using Shuffler = CheapestType<tSh, SuperShuffler, ShufflerExtractSet>;
};
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_SHUFFLE_STATIC_SHARED_HPP
