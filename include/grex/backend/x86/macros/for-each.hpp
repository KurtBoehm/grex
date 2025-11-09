// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_MACROS_FOR_EACH_HPP
#define INCLUDE_GREX_BACKEND_X86_MACROS_FOR_EACH_HPP

#include <cstddef> // IWYU pragma: keep

#include "grex/backend/x86/instruction-sets.hpp"

#if GREX_X86_64_LEVEL >= 4
#define GREX_FOREACH_X86_64_LEVEL(MACRO, ...) \
  MACRO(128, _mm __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(256, _mm256 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(512, _mm512 __VA_OPT__(, ) __VA_ARGS__)
#elif GREX_X86_64_LEVEL >= 3
#define GREX_FOREACH_X86_64_LEVEL(MACRO, ...) \
  MACRO(128, _mm __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(256, _mm256 __VA_OPT__(, ) __VA_ARGS__)
#else
#define GREX_FOREACH_X86_64_LEVEL(MACRO, ...) MACRO(128, _mm __VA_OPT__(, ) __VA_ARGS__)
#endif

#endif // INCLUDE_GREX_BACKEND_X86_MACROS_FOR_EACH_HPP
