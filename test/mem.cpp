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
  using Vec = grex::Vector<T, tSize>;

  grex::static_apply<tSize>([&]<std::size_t... tIdxs>() {
    {
      std::array buf{T(T(tSize) - 2 * T(tIdxs) + 1)...};
      VC checker{Vec::load(buf.data()), buf};
      checker.check();
    }
    {
      alignas(64) std::array buf{T(T(tSize) - 2 * T(tIdxs) + 1)...};
      VC checker{Vec::load_aligned(buf.data()), buf};
      checker.check();
    }
    {
      std::array buf{T(T(tSize) - 2 * T(tIdxs) + 1)...};
      for (std::size_t i = 0; i <= tSize; ++i) {
        VC checker{Vec::load_part(buf.data(), i), std::array{((tIdxs < i) ? buf[tIdxs] : T{})...}};
        checker.check();
      }
    }
  });

  grex::static_apply<tSize>([&]<std::size_t... tIdxs>() {
    VC checker{T(T(tSize) - 2 * T(tIdxs) + 1)...};
    {
      std::array<T, tSize> buf{};
      checker.vec.store(buf.data());
      test::check(buf, checker.ref);
    }
    {
      alignas(64) std::array<T, tSize> buf{};
      checker.vec.store_aligned(buf.data());
      test::check(buf, checker.ref);
    }
    {
      for (std::size_t i = 0; i <= tSize; ++i) {
        std::array<T, tSize> buf{};
        checker.vec.store_part(buf.data(), i);
        test::check(buf, std::array{((tIdxs < i) ? checker.ref[tIdxs] : T{})...});
      }
    }
  });
}

int main() {
  test::run_types_sizes([](auto... args) { run(args...); });
}
