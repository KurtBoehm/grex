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
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_BLEND_ZERO_STATIC_SHARED_HPP
