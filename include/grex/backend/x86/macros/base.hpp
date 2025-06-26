// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_MACROS_BASE_HPP
#define INCLUDE_GREX_BACKEND_X86_MACROS_BASE_HPP

#define GREX_EMPTY()
#define GREX_COMMA() ,

#define GREX_VARIADIC_SIZE_I(D0, D1, D2, D3, D4, D5, D6, D7, SIZE, ...) SIZE
// Supports sizes up to 8
#define GREX_VARIADIC_SIZE(...) \
  GREX_VARIADIC_SIZE_I(__VA_ARGS__ __VA_OPT__(, ) 8, 7, 6, 5, 4, 3, 2, 1, 0)

#define GREX_CAT1(X0) X0
#define GREX_CAT2(X0, X1) X0##X1
#define GREX_CAT3(X0, X1, X2) X0##X1##X2
#define GREX_CAT4(X0, X1, X2, X3) X0##X1##X2##X3
#define GREX_CAT5(X0, X1, X2, X3, X4) X0##X1##X2##X3##X4
// Expand the size
#define GREX_CAT_II(SIZE, ...) GREX_CAT##SIZE(__VA_ARGS__)
#define GREX_CAT_I(SIZE, ...) GREX_CAT_II(SIZE __VA_OPT__(, ) __VA_ARGS__)
#define GREX_CAT(...) GREX_CAT_I(GREX_VARIADIC_SIZE(__VA_ARGS__), __VA_ARGS__)

#endif // INCLUDE_GREX_BACKEND_X86_MACROS_BASE_HPP
