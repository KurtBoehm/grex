// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_BLEND_ZERO_STATIC_SHARED_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_BLEND_ZERO_STATIC_SHARED_HPP

#include <cstddef>
#include <utility>

#include "grex/backend/defs.hpp"
#include "grex/backend/x86/defs.hpp"
#include "grex/backend/x86/instruction-sets.hpp"
#include "grex/backend/x86/operations/bitwise.hpp"
#include "grex/backend/x86/operations/blend-zero-static/base.hpp"
#include "grex/backend/x86/operations/reinterpret.hpp"
#include "grex/backend/x86/operations/set.hpp"
#include "grex/base/defs.hpp"

namespace grex::backend {
struct ZeroBlenderNoop : public BaseExpensiveOp {
  template<AnyBlendZeros auto tBzs>
  static constexpr bool is_applicable(AutoTag<tBzs> /*tag*/) {
    return static_apply<tBzs.size>([&]<std::size_t... tIdxs>() {
      return (... && (tBzs[tIdxs] == keep_bz || tBzs[tIdxs] == any_bz));
    });
  }
  template<AnyVector TVec, BlendZerosFor<TVec> tBzs>
  static TVec apply(TVec vec, AutoTag<tBzs> /*tag*/) {
    static_assert(is_applicable(auto_tag<tBzs>));
    return vec;
  }
  static constexpr std::pair<f64, f64> cost(auto /*bzs*/) {
    return {0, 0};
  }
};
struct ZeroBlenderZero : public BaseExpensiveOp {
  template<AnyBlendZeros auto tBzs>
  static constexpr bool is_applicable(AutoTag<tBzs> /*tag*/) {
    return static_apply<tBzs.size>([&]<std::size_t... tIdxs>() {
      return (... && (tBzs[tIdxs] == zero_bz || tBzs[tIdxs] == any_bz));
    });
  }
  template<AnyVector TVec, BlendZerosFor<TVec> tBzs>
  static TVec apply(TVec /*vec*/, AutoTag<tBzs> /*tag*/) {
    static_assert(is_applicable(auto_tag<tBzs>));
    return zeros(type_tag<TVec>);
  }
  static constexpr std::pair<f64, f64> cost(auto /*bzs*/) {
    return {0, 1};
  }
};

struct ZeroBlenderAnd : public BaseExpensiveOp {
  template<AnyBlendZeros auto tBzs>
  static constexpr bool is_applicable(AutoTag<tBzs> /*tag*/) {
    return true;
  }
  template<AnyVector TVec, BlendZerosFor<TVec> tBzs>
  static TVec apply(TVec vec, AutoTag<tBzs> /*tag*/) {
    using Value = TVec::Value;
    static constexpr std::size_t size = TVec::size;
    using Int = SignedInt<sizeof(Value)>;
    using IVec = Vector<Int, size>;

    const IVec ivec = reinterpret(vec, type_tag<Int>);
    const IVec mask = static_apply<size>([]<std::size_t... tIdxs>() {
      return set(type_tag<IVec>, ((tBzs[tIdxs] == BlendZero::keep) ? Int(-1) : Int(0))...);
    });
    return reinterpret(bitwise_and(ivec, mask), type_tag<Value>);
  }
  static constexpr std::pair<f64, f64> cost(auto /*bzs*/) {
    return {0.5, 4};
  }
};

struct SubZeroBlender : public BaseExpensiveOp {
  template<AnyBlendZeros auto tBzs>
  using Base = ZeroBlender<tBzs.sub_extended()>;

  template<AnyBlendZeros auto tBzs>
  static constexpr bool is_applicable(AutoTag<tBzs> /*tag*/) {
    return Base<tBzs>::is_applicable(auto_tag<tBzs.sub_extended()>);
  }
  template<AnyVector TVec, BlendZerosFor<TVec> tBzs>
  static TVec apply(TVec vec, AutoTag<tBzs> /*tag*/) {
    return TVec{Base<tBzs>::apply(vec.full, auto_tag<tBzs.sub_extended()>)};
  }
  template<AnyBlendZeros auto tBzs>
  static constexpr std::pair<f64, f64> cost(AutoTag<tBzs> /*tag*/) {
    return Base<tBzs>::cost(auto_tag<tBzs.sub_extended()>);
  }
};
struct SuperZeroBlender : public BaseExpensiveOp {
  template<AnyBlendZeros auto tBzs>
  static constexpr bool is_applicable(AutoTag<tBzs> /*tag*/) {
    return ZeroBlender<tBzs.lower()>::is_applicable(auto_tag<tBzs.lower()>) &&
           ZeroBlender<tBzs.upper()>::is_applicable(auto_tag<tBzs.upper()>);
  }
  template<AnyVector TVec, BlendZerosFor<TVec> tBzs>
  static TVec apply(TVec vec, AutoTag<tBzs> /*tag*/) {
    return TVec{
      .lower = ZeroBlender<tBzs.lower()>::apply(vec.lower, auto_tag<tBzs.lower()>),
      .upper = ZeroBlender<tBzs.upper()>::apply(vec.upper, auto_tag<tBzs.upper()>),
    };
  }
  template<AnyBlendZeros auto tBzs>
  static constexpr std::pair<f64, f64> cost(AutoTag<tBzs> /*tag*/) {
    const auto [c0a, c1a] = ZeroBlender<tBzs.lower()>::cost(auto_tag<tBzs.lower()>);
    const auto [c0b, c1b] = ZeroBlender<tBzs.upper()>::cost(auto_tag<tBzs.upper()>);
    return {c0a + c0b, c1a + c1b};
  }
};

template<AnyBlendZeros auto tBzs>
requires((tBzs.value_size * tBzs.size < register_bytes.front()))
struct ZeroBlenderTrait<tBzs> {
  using Type = SubZeroBlender;
};
template<AnyBlendZeros auto tBzs>
requires((tBzs.value_size * tBzs.size > register_bytes.back()))
struct ZeroBlenderTrait<tBzs> {
  using Type = SuperZeroBlender;
};
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_BLEND_ZERO_STATIC_SHARED_HPP
