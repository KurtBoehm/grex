// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <array>
#include <bit>
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
  using Vec = grex::Vector<T, tSize>;
  using VC = test::VectorChecker<T, tSize>;

  auto dist = test::make_distribution<T>();
  auto dval = [&](std::size_t /*dummy*/) { return dist(rng); };

  // expand values
  for (std::size_t i = 0; i < repetitions; ++i) {
    const T value = dist(rng);
    const Vec vector = Vec::expanded_any(value);
    test::check(value, vector[0], false);
  }
  grex::static_apply<tSize>([&]<std::size_t... tIdxs>() {
    for (std::size_t i = 0; i < repetitions; ++i) {
      const T value = dist(rng);
      const test::VectorChecker<T, tSize> checker{
        Vec::expanded_zero(value),
        std::array{((tIdxs == 0) ? value : T{})...},
      };
      checker.check(false);
    }
  });

  // expand vectors
  auto expav = [&]<std::size_t tDstSize>(grex::IndexTag<tDstSize> /*tag*/) {
    using VDC = test::VectorChecker<T, tDstSize>;

    fmt::print("size: {}\n", tDstSize);
    grex::static_apply<tSize>([&]<std::size_t... tIdxs> {
      grex::static_apply<tDstSize>([&]<std::size_t... tDstIdxs> {
        for (std::size_t i = 0; i < repetitions; ++i) {
          VC checker{dval(tIdxs)...};
          {
            const auto v = checker.vec.expand_any(grex::index_tag<tDstSize>);
            test::check_msg((... && (v[tIdxs] == checker.ref[tIdxs])), v, checker.ref, false);
          }
          {
            VDC dchecker{
              checker.vec.expand_zero(grex::index_tag<tDstSize>),
              {((tDstIdxs < tSize) ? checker.ref[tDstIdxs] : T{})...},
            };
            dchecker.check(false);
          }
        }
      });
    });
  };
  grex::static_apply<std::bit_width(tSize) - 1, std::bit_width(grex::native_sizes<T>.back()) + 1>(
    [&]<std::size_t... tLogs> { (..., expav(grex::index_tag<1ULL << tLogs>)); });
}

int main() {
  pcg_extras::seed_seq_from<std::random_device> seed_source{};
  test::Rng rng{seed_source};
  test::run_types_sizes([&](auto vtag, auto stag) { run(rng, vtag, stag); });
}
