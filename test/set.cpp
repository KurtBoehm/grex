// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <array>
#include <cstddef>

#include "grex/grex.hpp"

#include "defs.hpp"

namespace test = grex::test;

template<grex::Vectorizable T, std::size_t tSize>
void run(grex::TypeTag<T> /*tag*/, grex::IndexTag<tSize> /*tag*/) {
  using VC = test::VectorChecker<T, tSize>;
  using MC = test::MaskChecker<T, tSize>;
  using Mask = grex::Mask<T, tSize>;

  grex::static_apply<tSize>([]<std::size_t... tIdxs>() {
    // vector
    {
      VC checker{};
      checker.check("vector zeros");
    }
    {
      VC checker{T{127}};
      checker.check("vector broadcast");
    }
    {
      VC checker{grex::Vector<T, tSize>::indices(), std::array{T(tIdxs)...}};
      checker.check("vector indices");
    }
    {
      VC checker{T(tSize - tIdxs)...};
      checker.check("vector set");
    }
    {
      const VC base{T(T(tSize) - 2 * T(tIdxs) + 1)...};
      for (std::size_t i = 0; i < tSize; ++i) {
        const auto val = T((T(i % 7) - T(2)) * T(3));
        VC v{base.vec.insert(i, val), std::array{((tIdxs == i) ? val : base.ref[tIdxs])...}};
        v.check("vector insert");
      }
      for (std::size_t i = 0; i <= tSize; ++i) {
        VC v{base.vec.cutoff(i), std::array{((tIdxs < i) ? base.ref[tIdxs] : T(0))...}};
        v.check("vector cutoff");
      }
    }

    // mask
    {
      MC checker{};
      checker.check("mask zeros");
    }
    {
      auto f = [](std::size_t /*dummy*/) { return true; };
      test::MaskChecker checker{grex::Mask<T, tSize>::ones(), std::array{f(tIdxs)...}};
      checker.check("mask ones");
    }
    {
      MC checker{false};
      checker.check("mask broadcast false");
    }
    {
      MC checker{true};
      checker.check("mask broadcast true");
    }
    {
      MC checker{((tIdxs % 5) % 2 == 0)...};
      checker.check("mask set");
    }
    {
      const MC base{(tIdxs % 3 != 1)...};
      for (std::size_t i = 0; i < tSize; ++i) {
        const bool val = i % 5 == 0;
        MC v{base.mask.insert(i, val), std::array{((tIdxs == i) ? val : base.ref[tIdxs])...}};
        v.check("mask insert");
      }
    }
    {
      for (std::size_t i = 0; i <= tSize; ++i) {
        MC v{Mask::cutoff_mask(i), std::array{(tIdxs < i)...}};
        v.check("mask cutoff_mask");
      }
    }
  });
}

int main() {
  test::run_types_sizes([](auto... args) { run(args...); });
}
