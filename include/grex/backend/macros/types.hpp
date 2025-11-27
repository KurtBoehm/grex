// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_MACROS_TYPES_HPP
#define INCLUDE_GREX_BACKEND_MACROS_TYPES_HPP

#include "grex/backend/macros/base.hpp"
#include "grex/backend/macros/math.hpp"

#define GREX_SUPER_UNARY(TYPE, NAME) \
  template<typename THalf> \
  inline TYPE<THalf> NAME(TYPE<THalf> v) { \
    return {.lower = NAME(v.lower), .upper = NAME(v.upper)}; \
  }
#define GREX_SUPER_BINARY(TYPE, NAME) \
  template<typename THalf> \
  inline TYPE<THalf> NAME(TYPE<THalf> a, TYPE<THalf> b) { \
    return {.lower = NAME(a.lower, b.lower), .upper = NAME(a.upper, b.upper)}; \
  }
#define GREX_SUPER_TERNARY(TYPE, NAME) \
  template<typename THalf> \
  inline TYPE<THalf> NAME(TYPE<THalf> a, TYPE<THalf> b, TYPE<THalf> c) { \
    return {.lower = NAME(a.lower, b.lower, c.lower), .upper = NAME(a.upper, b.upper, c.upper)}; \
  }
#define GREX_SUPERVECTOR_UNARY(NAME) GREX_SUPER_UNARY(SuperVector, NAME)
#define GREX_SUPERVECTOR_BINARY(NAME) GREX_SUPER_BINARY(SuperVector, NAME)
#define GREX_SUPERVECTOR_TERNARY(NAME) GREX_SUPER_TERNARY(SuperVector, NAME)
#define GREX_SUPERMASK_UNARY(NAME) GREX_SUPER_UNARY(SuperMask, NAME)
#define GREX_SUPERMASK_BINARY(NAME) GREX_SUPER_BINARY(SuperMask, NAME)
#define GREX_SUPERMASK_TERNARY(NAME) GREX_SUPER_TERNARY(SuperMask, NAME)

#define GREX_SUB_UNARY(TYPE, NAME) \
  template<Vectorizable T, std::size_t tPart, std::size_t tSize> \
  inline TYPE<T, tPart, tSize> NAME(TYPE<T, tPart, tSize> v) { \
    return TYPE<T, tPart, tSize>{NAME(v.full)}; \
  }
#define GREX_SUB_BINARY(TYPE, NAME) \
  template<Vectorizable T, std::size_t tPart, std::size_t tSize> \
  inline TYPE<T, tPart, tSize> NAME(TYPE<T, tPart, tSize> a, TYPE<T, tPart, tSize> b) { \
    return TYPE<T, tPart, tSize>{NAME(a.full, b.full)}; \
  }
#define GREX_SUB_TERNARY(TYPE, NAME) \
  template<Vectorizable T, std::size_t tPart, std::size_t tSize> \
  inline TYPE<T, tPart, tSize> NAME(TYPE<T, tPart, tSize> a, TYPE<T, tPart, tSize> b, \
                                    TYPE<T, tPart, tSize> c) { \
    return TYPE<T, tPart, tSize>{NAME(a.full, b.full, c.full)}; \
  }
#define GREX_SUBVECTOR_UNARY(NAME) GREX_SUB_UNARY(SubVector, NAME)
#define GREX_SUBVECTOR_BINARY(NAME) GREX_SUB_BINARY(SubVector, NAME)
#define GREX_SUBVECTOR_TERNARY(NAME) GREX_SUB_TERNARY(SubVector, NAME)
#define GREX_SUBMASK_UNARY(NAME) GREX_SUB_UNARY(SubMask, NAME)
#define GREX_SUBMASK_BINARY(NAME) GREX_SUB_BINARY(SubMask, NAME)
#define GREX_SUBMASK_TERNARY(NAME) GREX_SUB_TERNARY(SubMask, NAME)

#endif // INCLUDE_GREX_BACKEND_MACROS_TYPES_HPP
