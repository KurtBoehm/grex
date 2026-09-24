// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <array>
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
  using Vec = grex::Vector<T, N>;

  auto dist = test::make_distribution<T>();
  auto dval = [&] { return dist(rng); };

  for (std::size_t i = 0; i < repetitions; ++i) {
    grex::static_apply<N>([&]<std::size_t... I> {
      // load scalar
      {
        std::array<T, 1> buf{dist(rng)};
        test::check("load scalar", grex::load(buf.data(), grex::scalar_tag), buf[0],
                    {.verbose = false});
      }
      // load full
      {
        std::array buf = test::random_array<T, N>(dval);
        const VC checker{Vec::load(buf.data()), buf};
        checker.check("load", {.verbose = false});
      }
      {
        std::array buf = test::random_array<T, N>(dval);
        const VC checker{grex::load(buf.data(), grex::full_tag<N>), buf};
        checker.check("load tagged", {.verbose = false});
      }

      // load full aligned
      {
        alignas(64) std::array buf = test::random_array<T, N>(dval);
        const VC checker{Vec::load_aligned(buf.data()), buf};
        checker.check("load_aligned", {.verbose = false});
      }
      // there is no tagged version of aligned loading

      // load part
      {
        std::array buf = test::random_array<T, N>(dval);
        for (std::size_t j = 0; j <= N; ++j) {
          {
            const VC checker{
              Vec::load_part(buf.data(), j),
              std::array{((I < j) ? buf[I] : T{})...},
            };
            checker.check("load_part", j, {.verbose = false});
          }
          // tagged
          {
            const VC checker{
              grex::load(buf.data(), grex::part_tag<N>(j)),
              std::array{((I < j) ? buf[I] : T{})...},
            };
            checker.check("load_part tagged", j, {.verbose = false});
          }
        }
      }
      {
        std::array buf = test::random_array<T, N>(dval);
        auto load_part_wrap = [&](grex::AnyIndexTag auto j) {
          {
            const VC checker{
              Vec::load_part(buf.data(), j.value),
              std::array{((I < j) ? buf[I] : T{})...},
            };
            checker.check("load_part", j.value, {.verbose = false});
          }
          {
            const VC checker{
              Vec::load_part(buf.data(), j),
              std::array{((I < j) ? buf[I] : T{})...},
            };
            checker.check("load_part", j.value, {.verbose = false});
          }
          // tagged
          {
            const VC checker{
              grex::load(buf.data(), grex::part_tag<N>(j.value)),
              std::array{((I < j) ? buf[I] : T{})...},
            };
            checker.check("load_part tagged", j.value, {.verbose = false});
          }
          {
            const VC checker{
              grex::load(buf.data(), grex::part_tag<N>(j)),
              std::array{((I < j) ? buf[I] : T{})...},
            };
            checker.check("load_part tagged", j.value, {.verbose = false});
          }
        };
        grex::static_apply<N + 1>(
          [&]<std::size_t... J> { (..., load_part_wrap(grex::index_tag<J>)); });
      }
    });
  }
}
#endif
template<grex::Vectorizable T>
void run_scalar(test::Rng& rng, grex::TypeTag<T> /*tag*/) {
  auto dist = test::make_distribution<T>(); // NOLINT(*-const-correctness)

  for (std::size_t i = 0; i < repetitions; ++i) {
    // load scalar
    {
      std::array<T, 1> buf{dist(rng)};
      test::check("load scalar", grex::load(buf.data(), grex::scalar_tag), buf[0],
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
