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
#include "grex/backend/x86/operations/blend-static.hpp"
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
    return true;
  }
  template<AnyVector TVec, ShuffleIndicesFor<TVec> tSh>
  static TVec apply(TVec vec, AutoTag<tSh> /*tag*/) {
    static constexpr auto lower_sh = tSh.lower();
    static constexpr auto upper_sh = tSh.upper();

    const auto lower = [&] {
      if constexpr (lower_sh.has_value()) {
        return Shuffler<lower_sh.value()>::apply(vec.lower, auto_tag<lower_sh.value()>);
      } else {
        return Blender<tSh.lower_blend()>::apply(
          Shuffler<tSh.lower(true)>::apply(vec.lower, auto_tag<tSh.lower(true)>),
          Shuffler<tSh.lower(false)>::apply(vec.upper, auto_tag<tSh.lower(false)>),
          auto_tag<tSh.lower_blend()>);
      }
    }();
    const auto upper = [&] {
      if constexpr (upper_sh.has_value()) {
        return Shuffler<upper_sh.value()>::apply(vec.upper, auto_tag<upper_sh.value()>);
      } else {
        return Blender<tSh.upper_blend()>::apply(
          Shuffler<tSh.upper(true)>::apply(vec.upper, auto_tag<tSh.upper(true)>),
          Shuffler<tSh.upper(false)>::apply(vec.lower, auto_tag<tSh.upper(false)>),
          auto_tag<tSh.upper_blend()>);
      }
    }();
    return TVec{.lower = lower, .upper = upper};
  }
  template<AnyShuffleIndices auto tSh>
  static constexpr std::pair<f64, f64> cost(AutoTag<tSh> /*tag*/) {
    constexpr auto lower_sh = tSh.lower();
    constexpr auto upper_sh = tSh.upper();

    const auto [c0a, c1a] = [&] {
      if constexpr (lower_sh.has_value()) {
        return Shuffler<lower_sh.value()>::cost(auto_tag<lower_sh.value()>);
      } else {
        const auto [c00, c01] = Shuffler<tSh.lower(true)>::cost(auto_tag<tSh.lower(true)>);
        const auto [c10, c11] = Shuffler<tSh.lower(false)>::cost(auto_tag<tSh.lower(false)>);
        const auto [c20, c21] = Blender<tSh.lower_blend()>::cost(auto_tag<tSh.lower_blend()>);
        return std::make_pair(c00 + c10 + c20, c01 + c11 + c21);
      }
    }();
    const auto [c0b, c1b] = [&] {
      if constexpr (upper_sh.has_value()) {
        return Shuffler<upper_sh.value()>::cost(auto_tag<upper_sh.value()>);
      } else {
        const auto [c00, c01] = Shuffler<tSh.upper(true)>::cost(auto_tag<tSh.upper(true)>);
        const auto [c10, c11] = Shuffler<tSh.upper(false)>::cost(auto_tag<tSh.upper(false)>);
        const auto [c20, c21] = Blender<tSh.upper_blend()>::cost(auto_tag<tSh.upper_blend()>);
        return std::make_pair(c00 + c10 + c20, c01 + c11 + c21);
      }
    }();
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
  // TODO With AVX-512, the vpermi2 family can be used to merge two permutations
  using Shuffler = SuperShuffler;
};
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_SHUFFLE_STATIC_SHARED_HPP
