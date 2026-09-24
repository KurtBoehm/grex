// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <cstddef>
#include <random>

#include <fmt/base.h> // IWYU pragma: keep
#include <pcg_extras.hpp>

#include "grex/grex.hpp"

#include "defs.hpp"

namespace {
namespace test = grex::test;
inline constexpr std::size_t repetitions = 4096;

#if !GREX_BACKEND_SCALAR
template<grex::Vectorizable T, std::size_t N>
void run_simd(test::Rng& rng, grex::TypeTag<T> /*tag*/, grex::IndexTag<N> /*tag*/) {
  using VC = test::VectorChecker<T, N>;

  auto dist = test::make_distribution<T>();
  auto dval = [&] { return dist(rng); };

  grex::static_apply<N>([&]<std::size_t... I> {
    for (std::size_t i = 0; i < repetitions; ++i) {
      VC base = VC::random(dval);
      // zero-inserting upwards shingling
      test::check("shingle_up zero scalar", grex::shingle_up(dist(rng), grex::scalar_tag), T{},
                  {.verbose = false});
      {
        const VC checker{base.vec.shingle_up(), {((I == 0) ? T{} : base.ref[I - 1])...}};
        checker.check("shingle_up zero", {.verbose = false});
      }
      {
        const VC checker{
          grex::shingle_up(base.vec, grex::full_tag<N>),
          {((I == 0) ? T{} : base.ref[I - 1])...},
        };
        checker.check("shingle_up zero tagged", {.verbose = false});
      }

      // value-inserting upwards shingling
      {
        const T front = dist(rng);
        test::check("shingle_up value scalar", grex::shingle_up(front, dist(rng), grex::scalar_tag),
                    front, {.verbose = false});
      }
      {
        const T front = dist(rng);
        const VC checker{base.vec.shingle_up(front), {((I == 0) ? front : base.ref[I - 1])...}};
        checker.check("shingle_up value", {.verbose = false});
      }
      {
        const T front = dist(rng);
        const VC checker{
          grex::shingle_up(front, base.vec, grex::full_tag<N>),
          {((I == 0) ? front : base.ref[I - 1])...},
        };
        checker.check("shingle_up value tagged", {.verbose = false});
      }

      // zero-inserting downwards shingling
      test::check("shingle_down zero scalar", grex::shingle_down(dist(rng), grex::scalar_tag), T{},
                  {.verbose = false});
      {
        const VC checker{
          base.vec.shingle_down(),
          {((I + 1 == N) ? T{} : base.ref[I + 1])...},
        };
        checker.check("shingle_down zero", {.verbose = false});
      }
      {
        const VC checker{
          grex::shingle_down(base.vec, grex::typed_full_tag<T, N>),
          {((I + 1 == N) ? T{} : base.ref[I + 1])...},
        };
        checker.check("shingle_down zero tagged", {.verbose = false});
      }

      // value-inserting downwards shingling
      {
        const T back = dist(rng);
        test::check("shingle_down value scalar",
                    grex::shingle_down(dist(rng), back, grex::scalar_tag), back,
                    {.verbose = false});
      }
      {
        const T back = dist(rng);
        const VC checker{
          base.vec.shingle_down(back),
          {((I + 1 == N) ? back : base.ref[I + 1])...},
        };
        checker.check("shingle_down value", {.verbose = false});
      }
      {
        const T back = dist(rng);
        const VC checker{
          grex::shingle_down(base.vec, back, grex::typed_full_tag<T, N>),
          {((I + 1 == N) ? back : base.ref[I + 1])...},
        };
        checker.check("shingle_down value tagged", {.verbose = false});
      }
    }
  });
}
#endif
template<grex::Vectorizable T>
void run_scalar(test::Rng& rng, grex::TypeTag<T> /*tag*/) {
  auto dist = test::make_distribution<T>(); // NOLINT(*-const-correctness)

  for (std::size_t i = 0; i < repetitions; ++i) {
    // zero-inserting upwards shingling
    test::check("shingle_up zero", grex::shingle_up(dist(rng), grex::scalar_tag), T{},
                {.verbose = false});

    // value-inserting upwards shingling
    {
      const T front = dist(rng);
      test::check("shingle_up value", grex::shingle_up(front, dist(rng), grex::scalar_tag), front,
                  {.verbose = false});
    }

    // zero-inserting downwards shingling
    test::check("shingle_down zero", grex::shingle_down(dist(rng), grex::scalar_tag), T{},
                {.verbose = false});

    // value-inserting downwards shingling
    {
      const T back = dist(rng);
      test::check("shingle_down value", grex::shingle_down(dist(rng), back, grex::scalar_tag), back,
                  {.verbose = false});
    }
  }
}
} // namespace

int main() {
  pcg_extras::seed_seq_from<std::random_device> seed_source{};
  test::Rng rng{seed_source};
#if !GREX_BACKEND_SCALAR
  test::run_types_sizes([&](auto vtag, auto stag) { run_simd(rng, vtag, stag); });
#endif
  test::run_types([&](auto tag) { run_scalar(rng, tag); });
}
