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
      // zero-inserting upwards shingling
      test::check("shingle_up zero scalar", grex::shingle_up(dist(rng), grex::scalar_tag), T{},
                  false);
      {
        VC checker{base.vec.shingle_up(), {((tIdxs == 0) ? T{} : base.ref[tIdxs - 1])...}};
        checker.check("shingle_up zero", false);
      }
      {
        VC checker{
          grex::shingle_up(base.vec, grex::full_tag<tSize>),
          {((tIdxs == 0) ? T{} : base.ref[tIdxs - 1])...},
        };
        checker.check("shingle_up zero tagged", false);
      }

      // value-inserting upwards shingling
      {
        const T front = dist(rng);
        test::check("shingle_up value scalar", grex::shingle_up(front, dist(rng), grex::scalar_tag),
                    front, false);
      }
      {
        const T front = dist(rng);
        VC checker{base.vec.shingle_up(front), {((tIdxs == 0) ? front : base.ref[tIdxs - 1])...}};
        checker.check("shingle_up value", false);
      }
      {
        const T front = dist(rng);
        VC checker{
          grex::shingle_up(front, base.vec, grex::full_tag<tSize>),
          {((tIdxs == 0) ? front : base.ref[tIdxs - 1])...},
        };
        checker.check("shingle_up value tagged", false);
      }

      // zero-inserting downwards shingling
      test::check("shingle_down zero scalar", grex::shingle_down(dist(rng), grex::scalar_tag), T{},
                  false);
      {
        VC checker{
          base.vec.shingle_down(),
          {((tIdxs + 1 == tSize) ? T{} : base.ref[tIdxs + 1])...},
        };
        checker.check("shingle_down zero", false);
      }
      {
        VC checker{
          grex::shingle_down(base.vec, grex::typed_full_tag<T, tSize>),
          {((tIdxs + 1 == tSize) ? T{} : base.ref[tIdxs + 1])...},
        };
        checker.check("shingle_down zero tagged", false);
      }

      // value-inserting downwards shingling
      {
        const T back = dist(rng);
        test::check("shingle_down value scalar",
                    grex::shingle_down(dist(rng), back, grex::scalar_tag), back, false);
      }
      {
        const T back = dist(rng);
        VC checker{
          base.vec.shingle_down(back),
          {((tIdxs + 1 == tSize) ? back : base.ref[tIdxs + 1])...},
        };
        checker.check("shingle_down value", false);
      }
      {
        const T back = dist(rng);
        VC checker{
          grex::shingle_down(base.vec, back, grex::typed_full_tag<T, tSize>),
          {((tIdxs + 1 == tSize) ? back : base.ref[tIdxs + 1])...},
        };
        checker.check("shingle_down value tagged", false);
      }
    }
  });
}

int main() {
  pcg_extras::seed_seq_from<std::random_device> seed_source{};
  test::Rng rng{seed_source};
  test::run_types_sizes([&](auto vtag, auto stag) { run(rng, vtag, stag); });
}
