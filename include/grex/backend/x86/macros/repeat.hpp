// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_MACROS_REPEAT_HPP
#define INCLUDE_GREX_BACKEND_X86_MACROS_REPEAT_HPP

#define GREX_REPEAT_2(MACRO, ...) \
  MACRO(2, 0 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(2, 1 __VA_OPT__(, ) __VA_ARGS__)
#define GREX_REPEAT_4(MACRO, ...) \
  MACRO(4, 0 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(4, 1 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(4, 2 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(4, 3 __VA_OPT__(, ) __VA_ARGS__)
#define GREX_REPEAT_8(MACRO, ...) \
  MACRO(8, 0 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(8, 1 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(8, 2 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(8, 3 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(8, 4 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(8, 5 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(8, 6 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(8, 7 __VA_OPT__(, ) __VA_ARGS__)
#define GREX_REPEAT_16(MACRO, ...) \
  MACRO(16, 0 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(16, 1 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(16, 2 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(16, 3 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(16, 4 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(16, 5 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(16, 6 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(16, 7 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(16, 8 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(16, 9 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(16, 10 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(16, 11 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(16, 12 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(16, 13 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(16, 14 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(16, 15 __VA_OPT__(, ) __VA_ARGS__)
#define GREX_REPEAT_32(MACRO, ...) \
  MACRO(32, 0 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(32, 1 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(32, 2 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(32, 3 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(32, 4 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(32, 5 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(32, 6 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(32, 7 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(32, 8 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(32, 9 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(32, 10 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(32, 11 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(32, 12 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(32, 13 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(32, 14 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(32, 15 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(32, 16 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(32, 17 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(32, 18 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(32, 19 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(32, 20 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(32, 21 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(32, 22 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(32, 23 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(32, 24 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(32, 25 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(32, 26 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(32, 27 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(32, 28 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(32, 29 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(32, 30 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(32, 31 __VA_OPT__(, ) __VA_ARGS__)
#define GREX_REPEAT_64(MACRO, ...) \
  MACRO(64, 0 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(64, 1 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(64, 2 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(64, 3 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(64, 4 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(64, 5 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(64, 6 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(64, 7 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(64, 8 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(64, 9 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(64, 10 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(64, 11 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(64, 12 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(64, 13 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(64, 14 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(64, 15 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(64, 16 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(64, 17 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(64, 18 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(64, 19 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(64, 20 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(64, 21 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(64, 22 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(64, 23 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(64, 24 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(64, 25 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(64, 26 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(64, 27 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(64, 28 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(64, 29 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(64, 30 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(64, 31 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(64, 32 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(64, 33 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(64, 34 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(64, 35 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(64, 36 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(64, 37 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(64, 38 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(64, 39 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(64, 40 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(64, 41 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(64, 42 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(64, 43 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(64, 44 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(64, 45 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(64, 46 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(64, 47 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(64, 48 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(64, 49 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(64, 50 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(64, 51 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(64, 52 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(64, 53 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(64, 54 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(64, 55 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(64, 56 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(64, 57 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(64, 58 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(64, 59 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(64, 60 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(64, 61 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(64, 62 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(64, 63 __VA_OPT__(, ) __VA_ARGS__)
#define GREX_REPEAT(SIZE, MACRO, ...) GREX_REPEAT_##SIZE(MACRO __VA_OPT__(, ) __VA_ARGS__)

#endif // INCLUDE_GREX_BACKEND_X86_MACROS_REPEAT_HPP
