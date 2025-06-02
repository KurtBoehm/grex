// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <cstddef>

#include "thesauros/types.hpp"

#include "grex/grex.hpp"

#include "defs.hpp"

namespace test = grex::test;

template<grex::Vectorizable T, std::size_t tSize>
void run(thes::TypeTag<T> /*tag*/, thes::IndexTag<tSize> /*tag*/) {
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
    thes::star::iota<0, tSize> | thes::star::apply([](auto... values) {
      test::VectorChecker<T, tSize> checker{T(tSize - values)...};
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
    thes::star::iota<0, tSize> | thes::star::apply([](auto... values) {
      test::MaskChecker<T, tSize> checker{((values % 5) % 2 == 0)...};
      checker.check();
    });
  }
}

int main() {
  test::run_types_sizes([](auto... args) { run(args...); });
}
