// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <fmt/base.h>

#include "grex/grex.hpp"

#if !GREX_BACKEND_SCALAR
#include <array>
#include <bit>
#include <cstddef>
#include <cstdlib>
#include <random>

#include <fmt/color.h>
#include <fmt/format.h>
#include <pcg_extras.hpp>

#include "defs.hpp"

namespace test = grex::test;
using Value = grex::GREX_VALUE_TYPE;
using Index = grex::GREX_INDEX_TYPE;
inline constexpr std::size_t repetitions = 256;
inline constexpr auto max_shift = std::bit_width(256 / sizeof(Value));

template<std::size_t tSize>
void run_simd(test::Rng& rng, grex::IndexTag<tSize> /*tag*/) {
  using VC = test::VectorChecker<Value, tSize>;

  auto dist = test::make_distribution<Value>();
  auto dval = [&](std::size_t /*dummy*/) { return dist(rng); };

  grex::static_apply<tSize>([&]<std::size_t... tI> {
    auto fix = [&]<std::size_t tIdxSize>(grex::IndexTag<tIdxSize> /*tag*/) {
      using IVC = test::VectorChecker<Index, tIdxSize>;
      using SVC = test::VectorChecker<Value, tIdxSize>;
      auto idst = std::uniform_int_distribution<Index>(0, Index(tSize - 1));
      auto ival = [&](std::size_t /*dummy*/) { return idst(rng); };

      grex::static_apply<tIdxSize>([&]<std::size_t... tJ> {
        fmt::print(fmt::fg(fmt::terminal_color::blue) | fmt::text_style(fmt::emphasis::bold),
                   "[{}x{}, {}x{}]\n", test::type_name<Value>(), tSize, test::type_name<Index>(),
                   tIdxSize);

        for (std::size_t i = 0; i < repetitions; ++i) {
          const VC table{dval(tI)...};
          const IVC idxs{ival(tJ)...};

          SVC shuffled{
            grex::shuffle(table.vec, idxs.vec),
            std::array<Value, tIdxSize>{table.ref[idxs.ref[tJ]]...},
          };
          shuffled.check(
            [&] {
              return fmt::format("shuffle<{}x{}, {}x{}>({}, {})", test::type_name<Value>(), tSize,
                                 test::type_name<Index>(), tIdxSize, idxs.vec, table.vec);
            },
            false);
        }
      });
    };
    grex::static_apply<1, 11>([&]<std::size_t... tJ>() { (..., fix(grex::index_tag<1ZU << tJ>)); });
  });
}

int main() {
  pcg_extras::seed_seq_from<std::random_device> seed_source{};
  test::Rng rng{seed_source};
  test::for_each_size<Value, max_shift>([&](auto /*vtag*/, auto stag) { run_simd(rng, stag); });
}
#endif
