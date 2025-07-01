// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <cstddef>
#include <random>

#include <fmt/base.h>
#include <fmt/color.h>
#include <pcg_extras.hpp>

#include "grex/grex.hpp"

#include "defs.hpp"

namespace test = grex::test;
inline constexpr std::size_t repetitions = 4096;

template<grex::Vectorizable T, std::size_t tSize>
void run(test::Rng& rng, grex::TypeTag<T> /*tag*/, grex::IndexTag<tSize> /*tag*/) {
  using VC = test::VectorChecker<T, tSize>;

  auto dist = test::make_distribution<T>();
  auto dval = [&](std::size_t /*dummy*/) { return dist(rng); };

  grex::static_apply<tSize>([&]<std::size_t... tIdxs> {
    for (std::size_t i = 0; i < repetitions; ++i) {
      VC base{dval(tIdxs)...};
      // shingle up
      {
        VC checker{
          base.vec.shingle_up(),
          {((tIdxs == 0) ? T{} : base.ref[tIdxs - 1])...},
        };
        checker.check("shingle_up zero", false);
      }
      {
        const T front = dist(rng);
        VC checker{
          base.vec.shingle_up(front),
          {((tIdxs == 0) ? front : base.ref[tIdxs - 1])...},
        };
        checker.check("shingle_up value", false);
      }
      // shingle down
      {
        VC checker{
          base.vec.shingle_down(),
          {((tIdxs + 1 == tSize) ? T{} : base.ref[tIdxs + 1])...},
        };
        checker.check("shingle_down zero", false);
      }
      {
        const T back = dist(rng);
        VC checker{
          base.vec.shingle_down(back),
          {((tIdxs + 1 == tSize) ? back : base.ref[tIdxs + 1])...},
        };
        checker.check("shingle_down value", false);
      }
    }
  });
}

int main() {
  pcg_extras::seed_seq_from<std::random_device> seed_source{};
  test::Rng rng{seed_source};
  test::run_types_sizes([&](auto vtag, auto stag) { run(rng, vtag, stag); });
}
