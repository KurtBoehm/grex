// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_SHARED_DEFS_HPP
#define INCLUDE_GREX_BACKEND_SHARED_DEFS_HPP

#include <concepts>
#include <type_traits>

#include "grex/base.hpp"

namespace grex::backend {
template<typename... T>
struct TypeSeq {
  template<typename... Other>
  using Prepended = TypeSeq<Other..., T...>;
};

struct BaseExpensiveOp {};
template<typename T>
concept AnyExpensiveOp = std::derived_from<T, BaseExpensiveOp>;

template<auto Value, AnyExpensiveOp... Remaining>
struct ApplicableTypesTrait;
template<auto Value>
struct ApplicableTypesTrait<Value> {
  using Selected = TypeSeq<>;
};
template<auto Value, AnyExpensiveOp Head, AnyExpensiveOp... Tail>
struct ApplicableTypesTrait<Value, Head, Tail...> {
  using Selected = std::conditional_t<
    Head::is_applicable(auto_tag<Value>),
    typename ApplicableTypesTrait<Value, Tail...>::Selected::template Prepended<Head>,
    typename ApplicableTypesTrait<Value, Tail...>::Selected>;
};
template<auto Value, typename... Remaining>
using ApplicableTypes = ApplicableTypesTrait<Value, Remaining...>::Selected;

struct Cost {
  f64 inv_throughput;
  f64 latency;

  auto operator<=>(const Cost&) const = default;
};

template<auto Value, typename ZeroBlenders>
struct CheapestTypeTrait;
template<auto Value, AnyExpensiveOp Only>
struct CheapestTypeTrait<Value, TypeSeq<Only>> {
  using Cheapest = Only;
};
template<auto Value, AnyExpensiveOp Head, AnyExpensiveOp... Tail>
requires(sizeof...(Tail) > 0)
struct CheapestTypeTrait<Value, TypeSeq<Head, Tail...>> {
  using CheapestTail = CheapestTypeTrait<Value, TypeSeq<Tail...>>::Cheapest;
  using Cheapest =
    std::conditional_t<Head::cost(auto_tag<Value>) <= CheapestTail::cost(auto_tag<Value>), Head,
                       CheapestTail>;
};
template<auto Value, typename... Remaining>
using CheapestType = CheapestTypeTrait<Value, ApplicableTypes<Value, Remaining...>>::Cheapest;
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_SHARED_DEFS_HPP
