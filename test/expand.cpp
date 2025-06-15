// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <array>
#include <cstddef>
#include <random>

#include <pcg_extras.hpp>

#include "grex/grex.hpp"

#include "defs.hpp"

namespace test = grex::test;
inline constexpr std::size_t repetitions = 4096;

template<grex::Vectorizable T>
void run(test::Rng& rng, grex::TypeTag<T> /*tag*/) {
  constexpr std::size_t size = grex::native_sizes<T>.front();
  using Vec = grex::Vector<T, size>;

  auto dist = test::make_distribution<T>();

  for (std::size_t i = 0; i < repetitions; ++i) {
    const T value = dist(rng);
    const Vec vector = Vec::expand_any(value);
    test::check(value, vector[0]);
  }

  grex::static_apply<size>([&]<std::size_t... tIdxs>() {
    for (std::size_t i = 0; i < repetitions; ++i) {
      const T value = dist(rng);
      const test::VectorChecker<T, size> checker{
        Vec::expand_zero(value),
        std::array{((tIdxs == 0) ? value : T{})...},
      };
      checker.check(false);
    }
  });
}

int main() {
  pcg_extras::seed_seq_from<std::random_device> seed_source{};
  test::Rng rng{seed_source};
  test::for_each_type([&](auto vtag) { run(rng, vtag); });
}
