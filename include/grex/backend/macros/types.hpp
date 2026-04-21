// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_MACROS_TYPES_HPP
#define INCLUDE_GREX_BACKEND_MACROS_TYPES_HPP

#define GREX_NN_UNARY(TYPE, NAME) \
  template<typename THalf> \
  inline Super##TYPE<THalf> NAME(Super##TYPE<THalf> v) { \
    return {.lower = NAME(v.lower), .upper = NAME(v.upper)}; \
  } \
  template<Vectorizable T, std::size_t tPart, std::size_t tSize> \
  inline Sub##TYPE<T, tPart, tSize> NAME(Sub##TYPE<T, tPart, tSize> v) { \
    return Sub##TYPE<T, tPart, tSize>{NAME(v.full)}; \
  }

#define GREX_NN_BINARY(TYPE, NAME) \
  template<typename THalf> \
  inline Super##TYPE<THalf> NAME(Super##TYPE<THalf> a, Super##TYPE<THalf> b) { \
    return {.lower = NAME(a.lower, b.lower), .upper = NAME(a.upper, b.upper)}; \
  } \
  template<Vectorizable T, std::size_t tPart, std::size_t tSize> \
  inline Sub##TYPE<T, tPart, tSize> NAME(Sub##TYPE<T, tPart, tSize> a, \
                                         Sub##TYPE<T, tPart, tSize> b) { \
    return Sub##TYPE<T, tPart, tSize>{NAME(a.full, b.full)}; \
  }

#define GREX_NN_TERNARY(TYPE, NAME) \
  template<typename THalf> \
  inline Super##TYPE<THalf> NAME(Super##TYPE<THalf> a, Super##TYPE<THalf> b, \
                                 Super##TYPE<THalf> c) { \
    return {.lower = NAME(a.lower, b.lower, c.lower), .upper = NAME(a.upper, b.upper, c.upper)}; \
  } \
  template<Vectorizable T, std::size_t tPart, std::size_t tSize> \
  inline Sub##TYPE<T, tPart, tSize> NAME( \
    Sub##TYPE<T, tPart, tSize> a, Sub##TYPE<T, tPart, tSize> b, Sub##TYPE<T, tPart, tSize> c) { \
    return Sub##TYPE<T, tPart, tSize>{NAME(a.full, b.full, c.full)}; \
  }

#define GREX_NNVECTOR_UNARY(NAME) GREX_NN_UNARY(Vector, NAME)
#define GREX_NNVECTOR_BINARY(NAME) GREX_NN_BINARY(Vector, NAME)
#define GREX_NNVECTOR_TERNARY(NAME) GREX_NN_TERNARY(Vector, NAME)
#define GREX_NNMASK_UNARY(NAME) GREX_NN_UNARY(Mask, NAME)
#define GREX_NNMASK_BINARY(NAME) GREX_NN_BINARY(Mask, NAME)
#define GREX_NNMASK_TERNARY(NAME) GREX_NN_TERNARY(Mask, NAME)

#endif // INCLUDE_GREX_BACKEND_MACROS_TYPES_HPP
