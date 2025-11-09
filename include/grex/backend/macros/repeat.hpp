// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_MACROS_REPEAT_HPP
#define INCLUDE_GREX_BACKEND_MACROS_REPEAT_HPP

#define GREX_REPEAT_2(SIZE, MACRO, ...) \
  MACRO(SIZE, 0 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 1 __VA_OPT__(, ) __VA_ARGS__)
#define GREX_REPEAT_4(SIZE, MACRO, ...) \
  GREX_REPEAT_2(SIZE, MACRO __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 2 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 3 __VA_OPT__(, ) __VA_ARGS__)
#define GREX_REPEAT_8(SIZE, MACRO, ...) \
  GREX_REPEAT_4(SIZE, MACRO __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 4 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 5 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 6 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 7 __VA_OPT__(, ) __VA_ARGS__)
#define GREX_REPEAT_16(SIZE, MACRO, ...) \
  GREX_REPEAT_8(SIZE, MACRO __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 8 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 9 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 10 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 11 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 12 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 13 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 14 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 15 __VA_OPT__(, ) __VA_ARGS__)
#define GREX_REPEAT_32(SIZE, MACRO, ...) \
  GREX_REPEAT_16(SIZE, MACRO __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 16 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 17 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 18 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 19 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 20 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 21 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 22 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 23 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 24 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 25 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 26 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 27 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 28 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 29 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 30 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 31 __VA_OPT__(, ) __VA_ARGS__)
#define GREX_REPEAT_64(SIZE, MACRO, ...) \
  GREX_REPEAT_32(SIZE, MACRO __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 32 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 33 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 34 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 35 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 36 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 37 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 38 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 39 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 40 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 41 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 42 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 43 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 44 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 45 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 46 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 47 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 48 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 49 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 50 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 51 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 52 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 53 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 54 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 55 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 56 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 57 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 58 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 59 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 60 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 61 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 62 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 63 __VA_OPT__(, ) __VA_ARGS__)
#define GREX_REPEAT(SIZE, MACRO, ...) GREX_REPEAT_##SIZE(SIZE, MACRO __VA_OPT__(, ) __VA_ARGS__)

#define GREX_RREPEAT_2(SIZE, MACRO, ...) \
  MACRO(SIZE, 1 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 0 __VA_OPT__(, ) __VA_ARGS__)
#define GREX_RREPEAT_4(SIZE, MACRO, ...) \
  MACRO(SIZE, 3 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 2 __VA_OPT__(, ) __VA_ARGS__) \
  GREX_RREPEAT_2(SIZE, MACRO __VA_OPT__(, ) __VA_ARGS__)
#define GREX_RREPEAT_8(SIZE, MACRO, ...) \
  MACRO(SIZE, 7 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 6 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 5 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 4 __VA_OPT__(, ) __VA_ARGS__) \
  GREX_RREPEAT_4(SIZE, MACRO __VA_OPT__(, ) __VA_ARGS__)
#define GREX_RREPEAT_16(SIZE, MACRO, ...) \
  MACRO(SIZE, 15 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 14 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 13 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 12 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 11 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 10 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 9 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 8 __VA_OPT__(, ) __VA_ARGS__) \
  GREX_RREPEAT_8(SIZE, MACRO __VA_OPT__(, ) __VA_ARGS__)
#define GREX_RREPEAT_32(SIZE, MACRO, ...) \
  MACRO(SIZE, 31 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 30 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 29 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 28 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 27 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 26 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 25 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 24 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 23 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 22 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 21 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 20 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 19 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 18 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 17 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 16 __VA_OPT__(, ) __VA_ARGS__) \
  GREX_RREPEAT_16(SIZE, MACRO __VA_OPT__(, ) __VA_ARGS__)
#define GREX_RREPEAT_64(SIZE, MACRO, ...) \
  MACRO(SIZE, 63 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 62 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 61 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 60 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 59 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 58 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 57 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 56 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 55 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 54 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 53 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 52 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 51 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 50 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 49 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 48 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 47 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 46 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 45 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 44 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 43 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 42 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 41 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 40 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 39 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 38 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 37 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 36 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 35 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 34 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 33 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(SIZE, 32 __VA_OPT__(, ) __VA_ARGS__) \
  GREX_RREPEAT_32(SIZE, MACRO __VA_OPT__(, ) __VA_ARGS__)
#define GREX_RREPEAT(SIZE, MACRO, ...) GREX_RREPEAT_##SIZE(SIZE, MACRO __VA_OPT__(, ) __VA_ARGS__)

#endif // INCLUDE_GREX_BACKEND_MACROS_REPEAT_HPP
