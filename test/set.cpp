// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <cstddef>

#include "grex/grex.hpp"

#include "defs.hpp"

namespace test = grex::test;

template<grex::Vectorizable T, std::size_t tSize>
void run(grex::TypeTag<T> /*tag*/, grex::IndexTag<tSize> /*tag*/) {
  // vector
  {
    test::VectorChecker<T, tSize> checker{};
    checker.check();
  }
  {
    test::VectorChecker<T, tSize> checker{T{127}};
    checker.check();
  }
  {
    grex::static_apply<tSize>([]<std::size_t... tIdxs>() {
      test::VectorChecker<T, tSize> checker{T(tSize - tIdxs)...};
      checker.check();
    });
  }
  {
    auto checker = test::VectorChecker<T, tSize>::indices();
    checker.check();
  }
  // mask
  {
    test::MaskChecker<T, tSize> checker{};
    checker.check();
  }
  {
    auto checker = test::MaskChecker<T, tSize>::ones();
    checker.check();
  }
  {
    test::MaskChecker<T, tSize> checker{false};
    checker.check();
  }
  {
    test::MaskChecker<T, tSize> checker{true};
    checker.check();
  }
  {
    grex::static_apply<tSize>([]<std::size_t... tIdxs>() {
      test::MaskChecker<T, tSize> checker{((tIdxs % 5) % 2 == 0)...};
      checker.check();
    });
  }
}

int main() {
  test::run_types_sizes([](auto... args) { run(args...); });
}
