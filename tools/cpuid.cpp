// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <array>
#include <cstdlib>
#include <print>
#include <ranges>
#include <span>
#include <string_view>

#include <cpuid.h>

#include "grex/backend/x86/cpuid.hpp"

namespace {
// Prints `names`, joined by ";", followed by a newline.
void print_semicolon_line(std::ranges::input_range auto&& names) {
  bool is_first = true;
  for (const auto name : names) {
    if (!is_first) {
      std::print(";");
    }
    is_first = false;
    std::print("{}", name);
  }
  std::print("\n");
}
} // namespace

int main() {
  using namespace std::string_view_literals;

  constexpr std::array level_names{"x86-64"sv, "x86-64-v2"sv, "x86-64-v3"sv, "x86-64-v4"sv};
  constexpr std::array extension_names{"avx512fp16"sv, "avx512vbmi"sv, "avx512vbmi2"sv};

  const grex::backend::CpuFeatures features = grex::backend::runtime_x86_64_level();
  if (features.level == 0) {
    return EXIT_FAILURE;
  }

  // Line 1: the microarchitecture levels supported, from "x86-64" up to the detected level.
  print_semicolon_line(std::span{level_names.begin(), features.level});

  // Line 2: optional extensions detected, which are not implied by any microarchitecture level.
  auto names = std::views::zip(std::array<bool, 3>{features.avx512fp16, features.avx512vbmi,
                                                   features.avx512vbmi2},
                               extension_names) |
               std::views::filter([](auto tup) { return std::get<0>(tup); }) |
               std::views::transform([](auto tup) { return std::get<1>(tup); });
  print_semicolon_line(names);
}
