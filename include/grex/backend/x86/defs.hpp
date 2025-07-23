// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_DEFS_HPP
#define INCLUDE_GREX_BACKEND_X86_DEFS_HPP

#include <concepts>
#include <type_traits>

#include "grex/base/defs.hpp"

namespace grex::backend {
struct BaseExpensiveOp {};
template<typename T>
concept AnyExpensiveOp = std::derived_from<T, BaseExpensiveOp>;

template<auto tValue, AnyExpensiveOp... TRemaining>
struct ApplicableTypesTrait;
template<auto tValue>
struct ApplicableTypesTrait<tValue> {
  using Selected = TypeSeq<>;
};
template<auto tValue, AnyExpensiveOp THead, AnyExpensiveOp... TTail>
struct ApplicableTypesTrait<tValue, THead, TTail...> {
  using Selected = std::conditional_t<
    THead::is_applicable(tValue),
    typename ApplicableTypesTrait<tValue, TTail...>::Selected::template Prepended<THead>,
    typename ApplicableTypesTrait<tValue, TTail...>::Selected>;
};
template<auto tValue, typename... TRemaining>
using ApplicableTypes = ApplicableTypesTrait<tValue, TRemaining...>::Selected;

template<auto tValue, typename TZeroBlenders>
struct CheapestTypeTrait;
template<auto tValue, AnyExpensiveOp TOnly>
struct CheapestTypeTrait<tValue, TypeSeq<TOnly>> {
  using Cheapest = TOnly;
};
template<auto tValue, AnyExpensiveOp THead, AnyExpensiveOp... TTail>
requires(sizeof...(TTail) > 0)
struct CheapestTypeTrait<tValue, TypeSeq<THead, TTail...>> {
  using CheapestTail = CheapestTypeTrait<tValue, TypeSeq<TTail...>>::Cheapest;
  using Cheapest =
    std::conditional_t<THead::cost(tValue) <= CheapestTail::cost(tValue), THead, CheapestTail>;
};
template<auto tValue, typename... TRemaining>
using CheapestType = CheapestTypeTrait<tValue, ApplicableTypes<tValue, TRemaining...>>::Cheapest;
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_DEFS_HPP
