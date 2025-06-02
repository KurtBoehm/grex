// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <cstddef>

#include "thesauros/static-ranges.hpp"

#include "grex/grex.hpp"

#include "defs.hpp"

#ifdef GREX_TEST_TYPE
using Value = grex::GREX_TEST_TYPE;
#else
using Value = grex::f32;
#endif

#ifdef GREX_TEST_SIZE
inline constexpr std::size_t size = GREX_TEST_SIZE;
#else
inline constexpr std::size_t size = 4;
#endif

namespace test = grex::test;

int main() {
  // vector
  {
    test::VectorChecker<Value, size> checker{};
    checker.check();
  }
  {
    test::VectorChecker<Value, size> checker{Value{127}};
    checker.check();
  }
  {
    thes::star::iota<0, size> | thes::star::apply([](auto... values) {
      test::VectorChecker<Value, size> checker{Value(size - values)...};
      checker.check();
    });
  }
  {
    auto checker = test::VectorChecker<Value, size>::indices();
    checker.check();
  }
  // mask
  {
    test::MaskChecker<Value, size> checker{};
    checker.check();
  }
  {
    auto checker = test::MaskChecker<Value, size>::ones();
    checker.check();
  }
  {
    test::MaskChecker<Value, size> checker{false};
    checker.check();
  }
  {
    test::MaskChecker<Value, size> checker{true};
    checker.check();
  }
  {
    thes::star::iota<0, size> | thes::star::apply([](auto... values) {
      test::MaskChecker<Value, size> checker{((values % 5) % 2 == 0)...};
      checker.check();
    });
  }
}
