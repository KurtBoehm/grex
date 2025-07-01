// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <cstddef>

#include <fmt/base.h>

#include "grex/grex.hpp"

#include "defs.hpp"

namespace test = grex::test;

template<grex::Vectorizable T, std::size_t tSize>
void run(grex::TypeTag<T> /*tag*/, grex::IndexTag<tSize> /*tag*/) {
  using VC = test::VectorChecker<T, tSize>;
  using MC = test::MaskChecker<T, tSize>;

  grex::static_apply<tSize>([&]<std::size_t... tIdxs>() {
    VC checker{T(T(tSize) - 2 * T(tIdxs) + 1)...};
    {
      fmt::print("horizontal_add({})\n", checker.vec);
      T v{};
      for (std::size_t i = 0; i < tSize; ++i) {
        v += checker.ref[i];
      }
      test::check("horizontal_add", grex::horizontal_add(checker.vec), v);
    }
    {
      fmt::print("horizontal_min({})\n", checker.vec);
      T v = checker.ref[0];
      for (std::size_t i = 1; i < tSize; ++i) {
        v = std::min(v, checker.ref[i]);
      }
      test::check("horizontal_min", grex::horizontal_min(checker.vec), v);
    }
    {
      fmt::print("horizontal_max({})\n", checker.vec);
      T v = checker.ref[0];
      for (std::size_t i = 1; i < tSize; ++i) {
        v = std::max(v, checker.ref[i]);
      }
      test::check("horizontal_max", grex::horizontal_max(checker.vec), v);
    }
  });
  grex::static_apply<tSize>([&]<std::size_t... tIdxs>() {
    {
      MC checker{((tIdxs % 5) % 2 == 1)...};
      fmt::print("horizontal_and({})\n", checker.mask);
      bool b = true;
      for (std::size_t i = 0; i < tSize; ++i) {
        b = b && checker.mask[i];
      }
      test::check("horizontal_and", grex::horizontal_and(checker.mask), b);
    }
    {
      MC checker{(tIdxs < tSize)...};
      fmt::print("horizontal_and({})\n", checker.mask);
      bool b = true;
      for (std::size_t i = 0; i < tSize; ++i) {
        b = b && checker.mask[i];
      }
      test::check("horizontal_and", grex::horizontal_and(checker.mask), b);
    }
  });
}

int main() {
  test::run_types_sizes([](auto... args) { run(args...); });
}
