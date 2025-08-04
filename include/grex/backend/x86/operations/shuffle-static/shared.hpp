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
template<AnyShuffleIndices auto tSh>
requires((tSh.value_size * tSh.size < register_bytes.front()))
struct ShufflerTrait<tSh> {
  using Shuffler = SubShuffler;
};

// A pair shuffler that just shuffles one of the vectors
struct PairShufflerSingle : public BaseExpensiveOp {
  template<AnyShuffleIndices auto tSh>
  static constexpr bool is_applicable(AutoTag<tSh> /*tag*/) {
    return tSh.indices_in_vector(0).has_value() || tSh.indices_in_vector(1).has_value();
  }
  template<AnyVector TVec, ShuffleIndicesFor<TVec> tSh>
  static TVec apply(TVec a, TVec b, AutoTag<tSh> /*tag*/) {
    static constexpr auto a_sh = tSh.indices_in_vector(0);
    static constexpr auto b_sh = tSh.indices_in_vector(1);

    if constexpr (a_sh.has_value()) {
      return Shuffler<a_sh.value()>::apply(a, auto_tag<a_sh.value()>);
    } else {
      return Shuffler<b_sh.value()>::apply(b, auto_tag<b_sh.value()>);
    }
  }
  template<AnyShuffleIndices auto tSh>
  static constexpr std::pair<f64, f64> cost(AutoTag<tSh> /*tag*/) {
    constexpr auto a_sh = tSh.indices_in_vector(0);
    constexpr auto b_sh = tSh.indices_in_vector(1);

    if constexpr (a_sh.has_value()) {
      return Shuffler<a_sh.value()>::cost(auto_tag<a_sh.value()>);
    } else {
      return Shuffler<b_sh.value()>::cost(auto_tag<b_sh.value()>);
    }
  }
};
// A pair shuffler that performs two shuffles and then blends
struct PairShufflerBlend : public BaseExpensiveOp {
  template<AnyShuffleIndices auto tSh>
  static constexpr bool is_applicable(AutoTag<tSh> /*tag*/) {
    return true;
  }
  template<AnyVector TVec, ShuffleIndicesFor<TVec> tSh>
  static TVec apply(TVec a, TVec b, AutoTag<tSh> /*tag*/) {
    static constexpr auto a_sh = tSh.indices_in_vector_fallback(0, any_sh);
    static constexpr auto b_sh = tSh.indices_in_vector_fallback(1, any_sh);

    return Blender<tSh.blend_vectors()>::apply(Shuffler<a_sh>::apply(a, auto_tag<a_sh>),
                                               Shuffler<b_sh>::apply(b, auto_tag<b_sh>),
                                               auto_tag<tSh.blend_vectors()>);
  }
  template<AnyShuffleIndices auto tSh>
  static constexpr std::pair<f64, f64> cost(AutoTag<tSh> /*tag*/) {
    constexpr auto a_sh = tSh.indices_in_vector_fallback(0, any_sh);
    constexpr auto b_sh = tSh.indices_in_vector_fallback(1, any_sh);

    const auto [c00, c01] = Shuffler<a_sh>::cost(auto_tag<a_sh>);
    const auto [c10, c11] = Shuffler<b_sh>::cost(auto_tag<b_sh>);
    const auto [c20, c21] = Blender<tSh.blend_vectors()>::cost(auto_tag<tSh.blend_vectors()>);
    return std::make_pair(c00 + c10 + c20, c01 + c11 + c21);
  }
};
template<AnyShuffleIndices auto tSh>
struct PairShufflerTrait {
  using Shuffler = CheapestType<tSh, PairShufflerSingle, PairShufflerBlend>;
};

struct SuperShuffler : public BaseExpensiveOp {
  template<AnyShuffleIndices auto tSh>
  static constexpr bool is_applicable(AutoTag<tSh> /*tag*/) {
    return true;
  }
  template<AnyVector TVec, ShuffleIndicesFor<TVec> tSh>
  static TVec apply(TVec vec, AutoTag<tSh> /*tag*/) {
    static constexpr auto lower_sh = tSh.half_raw(0);
    static constexpr auto upper_sh = tSh.half_raw(1);

    const auto lower = PairShuffler<lower_sh>::apply(vec.lower, vec.upper, auto_tag<lower_sh>);
    const auto upper = PairShuffler<upper_sh>::apply(vec.lower, vec.upper, auto_tag<upper_sh>);
    return TVec{.lower = lower, .upper = upper};
  }
  template<AnyShuffleIndices auto tSh>
  static constexpr std::pair<f64, f64> cost(AutoTag<tSh> /*tag*/) {
    constexpr auto lower_sh = tSh.half_raw(0);
    constexpr auto upper_sh = tSh.half_raw(1);
    const auto [c0a, c1a] = PairShuffler<lower_sh>::cost(auto_tag<lower_sh>);
    const auto [c0b, c1b] = PairShuffler<upper_sh>::cost(auto_tag<upper_sh>);
    return {c0a + c0b, c1a + c1b};
  }
};

template<AnyShuffleIndices auto tSh>
requires((tSh.value_size * tSh.size > register_bytes.back()))
struct ShufflerTrait<tSh> {
  using Shuffler = SuperShuffler;
};
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_SHUFFLE_STATIC_SHARED_HPP
