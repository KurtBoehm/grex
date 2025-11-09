// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_MACROS_FOR_EACH_HPP
#define INCLUDE_GREX_BACKEND_MACROS_FOR_EACH_HPP

#include <cstddef> // IWYU pragma: keep

#include "grex/backend/macros/math.hpp"

#define GREX_FOREACH_FP_TYPE(MACRO, REGISTERBITS, ...) \
  MACRO(f, 32, GREX_DIVIDE(REGISTERBITS, 32) __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(f, 64, GREX_DIVIDE(REGISTERBITS, 64) __VA_OPT__(, ) __VA_ARGS__)

#define GREX_FOREACH_INT_TYPE_BASE(MACRO, KIND, REGISTERBITS, ...) \
  MACRO(KIND, 64, GREX_DIVIDE(REGISTERBITS, 64) __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(KIND, 32, GREX_DIVIDE(REGISTERBITS, 32) __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(KIND, 16, GREX_DIVIDE(REGISTERBITS, 16) __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(KIND, 8, GREX_DIVIDE(REGISTERBITS, 8) __VA_OPT__(, ) __VA_ARGS__)
#define GREX_FOREACH_UINT_TYPE(MACRO, REGISTERBITS, ...) \
  GREX_FOREACH_INT_TYPE_BASE(MACRO, u, REGISTERBITS, __VA_ARGS__)
#define GREX_FOREACH_SINT_TYPE(MACRO, REGISTERBITS, ...) \
  GREX_FOREACH_INT_TYPE_BASE(MACRO, i, REGISTERBITS, __VA_ARGS__)

#define GREX_FOREACH_INT_TYPE(MACRO, REGISTERBITS, ...) \
  GREX_FOREACH_UINT_TYPE(MACRO, REGISTERBITS __VA_OPT__(, ) __VA_ARGS__) \
  GREX_FOREACH_SINT_TYPE(MACRO, REGISTERBITS __VA_OPT__(, ) __VA_ARGS__)

#define GREX_FOREACH_TYPE(MACRO, REGISTERBITS, ...) \
  GREX_FOREACH_FP_TYPE(MACRO, REGISTERBITS __VA_OPT__(, ) __VA_ARGS__) \
  GREX_FOREACH_INT_TYPE(MACRO, REGISTERBITS __VA_OPT__(, ) __VA_ARGS__)
#define GREX_FOREACH_TYPE_R(MACRO, REGISTERBITS, ...) \
  MACRO(f, 32, GREX_DIVIDE(REGISTERBITS, 32) __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(f, 64, GREX_DIVIDE(REGISTERBITS, 64) __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(i, 64, GREX_DIVIDE(REGISTERBITS, 64) __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(i, 32, GREX_DIVIDE(REGISTERBITS, 32) __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(i, 16, GREX_DIVIDE(REGISTERBITS, 16) __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(i, 8, GREX_DIVIDE(REGISTERBITS, 8) __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(u, 64, GREX_DIVIDE(REGISTERBITS, 64) __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(u, 32, GREX_DIVIDE(REGISTERBITS, 32) __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(u, 16, GREX_DIVIDE(REGISTERBITS, 16) __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(u, 8, GREX_DIVIDE(REGISTERBITS, 8) __VA_OPT__(, ) __VA_ARGS__)

#define GREX_FOREACH_SUB(MACRO, ...) \
  MACRO(f, 32, 2, 4 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(i, 32, 2, 4 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(u, 32, 2, 4 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(i, 16, 4, 8 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(u, 16, 4, 8 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(i, 16, 2, 8 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(u, 16, 2, 8 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(i, 8, 8, 16 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(u, 8, 8, 16 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(i, 8, 4, 16 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(u, 8, 4, 16 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(i, 8, 2, 16 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(u, 8, 2, 16 __VA_OPT__(, ) __VA_ARGS__)

#endif // INCLUDE_GREX_BACKEND_MACROS_FOR_EACH_HPP
