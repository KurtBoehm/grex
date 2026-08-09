// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_MACROS_TYPES_HPP
#define INCLUDE_GREX_BACKEND_MACROS_TYPES_HPP

#define GREX_REGKIND_f16 u
#define GREX_REGKIND_f32 f
#define GREX_REGKIND_f64 f
#define GREX_REGKIND_f(BITS) GREX_REGKIND_f##BITS
#define GREX_REGKIND_i(BITS) i
#define GREX_REGKIND_u(BITS) u
#define GREX_REGKIND_I(KIND, BITS) GREX_REGKIND_##KIND(BITS)
#define GREX_REGKIND(KIND, BITS) GREX_REGKIND_I(KIND, BITS)

#define GREX_NN_UNARY(TYPE, NAME) \
  template<typename THalf> \
  inline Super##TYPE<THalf> NAME(Super##TYPE<THalf> v) { \
    return {.lower = NAME(v.lower), .upper = NAME(v.upper)}; \
  } \
  template<Vectorizable T, std::size_t tSize> \
  inline Sub##TYPE<T, tSize> NAME(Sub##TYPE<T, tSize> v) { \
    return Sub##TYPE<T, tSize>{NAME(v.full)}; \
  }

#define GREX_NN_BINARY(TYPE, NAME) \
  template<typename THalf> \
  inline Super##TYPE<THalf> NAME(Super##TYPE<THalf> a, Super##TYPE<THalf> b) { \
    return {.lower = NAME(a.lower, b.lower), .upper = NAME(a.upper, b.upper)}; \
  } \
  template<Vectorizable T, std::size_t tSize> \
  inline Sub##TYPE<T, tSize> NAME(Sub##TYPE<T, tSize> a, Sub##TYPE<T, tSize> b) { \
    return Sub##TYPE<T, tSize>{NAME(a.full, b.full)}; \
  }

#define GREX_NNVECTOR_UNARY(NAME) GREX_NN_UNARY(Vector, NAME)
#define GREX_NNVECTOR_BINARY(NAME) GREX_NN_BINARY(Vector, NAME)
#define GREX_NNMASK_UNARY(NAME) GREX_NN_UNARY(Mask, NAME)
#define GREX_NNMASK_BINARY(NAME) GREX_NN_BINARY(Mask, NAME)

#endif // INCLUDE_GREX_BACKEND_MACROS_TYPES_HPP
