// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <array>
#include <cstddef>

#include <fmt/base.h>

#include "grex/grex.hpp"

#include "defs.hpp"

namespace test = grex::test;

template<grex::Vectorizable T, std::size_t tSize>
void run(grex::TypeTag<T> /*tag*/, grex::IndexTag<tSize> /*tag*/) {
  using VC = test::VectorChecker<T, tSize>;
  using MC = test::MaskChecker<T, tSize>;
  using Mask = grex::Mask<T, tSize>;

  // vector
  {
    VC checker{};
    checker.check();
  }
  {
    VC checker{T{127}};
    checker.check();
  }
  {
    auto checker = VC::indices();
    checker.check();
  }
  grex::static_apply<tSize>([]<std::size_t... tIdxs>() {
    VC checker{T(tSize - tIdxs)...};
    checker.check();
  });
  grex::static_apply<tSize>([]<std::size_t... tIdxs>() {
    const VC base{T(T(tSize) - 2 * T(tIdxs) + 1)...};
    for (std::size_t i = 0; i < tSize; ++i) {
      const auto val = T((T(i % 7) - T(2)) * T(3));
      VC v{base.vec.insert(i, val), std::array{((tIdxs == i) ? val : base.ref[tIdxs])...}};
      v.check();
    }
    for (std::size_t i = 0; i <= tSize; ++i) {
      VC v{base.vec.cutoff(i), std::array{((tIdxs < i) ? base.ref[tIdxs] : T(0))...}};
      v.check();
    }
  });

  // mask
  {
    MC checker{};
    checker.check();
  }
  {
    auto checker = MC::ones();
    checker.check();
  }
  {
    MC checker{false};
    checker.check();
  }
  {
    MC checker{true};
    checker.check();
  }
  grex::static_apply<tSize>([]<std::size_t... tIdxs>() {
    MC checker{((tIdxs % 5) % 2 == 0)...};
    checker.check();
  });
  grex::static_apply<tSize>([]<std::size_t... tIdxs>() {
    const MC base{(tIdxs % 3 != 1)...};
    for (std::size_t i = 0; i < tSize; ++i) {
      const bool val = i % 5 == 0;
      MC v{base.mask.insert(i, val), std::array{((tIdxs == i) ? val : base.ref[tIdxs])...}};
      v.check();
    }
  });
  grex::static_apply<tSize>([]<std::size_t... tIdxs>() {
    for (std::size_t i = 0; i <= tSize; ++i) {
      MC v{Mask::cutoff_mask(i), std::array{(tIdxs < i)...}};
      v.check();
    }
  });
}

int main() {
  test::run_types_sizes([](auto... args) { run(args...); });
}
