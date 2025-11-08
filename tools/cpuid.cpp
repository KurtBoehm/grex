// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <array>
#include <cstdio>
#include <cstdlib>

#include <cpuid.h>

#include "grex/backend/x86/cpuid.hpp"

int main() {
  constexpr std::array<const char*, 4> level_names{"x86-64", "x86-64-v2", "x86-64-v3", "x86-64-v4"};

  const unsigned int level = grex::backend::runtime_x86_64_level();
  if (level == 0) {
    return EXIT_FAILURE;
  }

  bool is_first = true;
  for (std::size_t i = 0; i < level; ++i) {
    const auto* level_name = level_names[i];
    if (is_first) {
      is_first = false;
      printf("%s", level_name);
    } else {
      printf(";%s", level_name);
    }
  }
}
