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
#include <concepts>
#include <cstddef>
#include <cstdlib>
#include <random>

#include <fmt/color.h>
#include <fmt/format.h>
#include <pcg_extras.hpp>

#include "defs.hpp"

namespace {
namespace test = grex::test;
using Value = grex::GREX_VALUE_TYPE;
using Index = grex::GREX_INDEX_TYPE;
inline constexpr std::size_t repetitions = 256;
inline constexpr auto max_shift = std::bit_width(256 / sizeof(Value));

/**
 * Test shuffling with runtime indices, including `LookupTable` (which may or may not be backed by
 * `shuffle`) if the index type matches that of the lookup table. `N` refers to the lane count of
 * the table; the function tests multiple index lane counts.
 */
template<std::size_t N>
void run_simd(test::Rng& rng, grex::IndexTag<N> /*tag*/) {
  using VC = test::VectorChecker<Value, N>;
  // Determine the index type of the lookup table, ensure that it is an unsigned vectorizable type,
  // and perform the tests if it is the same as the current index type.
  using Table = grex::LookupTable<Value, N>;
  using TableIndex = Table::Index;
  static_assert(grex::UnsignedVectorizable<TableIndex>,
                "The table index has to be both vectorizable and unsigned.");

  auto dist = test::make_distribution<Value>(); // NOLINT(*-const-correctness)
  // The lambda to be executed at each index lane count `IdxN`.
  const auto fix = [&]<std::size_t IdxN>(grex::IndexTag<IdxN> /*tag*/) {
    using IVC = test::VectorChecker<Index, IdxN>;
    using SVC = test::VectorChecker<Value, IdxN>;
    auto idst = std::uniform_int_distribution<Index>(0, Index{N - 1});

    fmt::print(fmt::fg(fmt::terminal_color::blue) | fmt::emphasis::bold, "[{}×{}, {}×{}]\n",
               test::type_name<Value>(), N, test::type_name<Index>(), IdxN);

    for (std::size_t i = 0; i < repetitions; ++i) {
      const VC table = VC::random([&] { return dist(rng); });
      const IVC idxs = IVC::random([&] { return idst(rng); });

      std::array<Value, IdxN> ref{};
      for (std::size_t j = 0; j < IdxN; ++j) {
        ref[j] = table.ref[idxs.ref[j]];
      }

      const SVC shuffled{grex::shuffle(table.vec, idxs.vec), ref};
      shuffled.check(
        [&] {
          return fmt::format("shuffle<{}×{}, {}×{}>({}, {})", test::type_name<Value>(), N,
                             test::type_name<Index>(), IdxN, idxs.vec, table.vec);
        },
        false);

      if constexpr (std::same_as<Index, TableIndex>) {
        const Table ltable{table.ref};
        const SVC looked_up{ltable.lookup(idxs.vec, grex::full_tag<IdxN>), ref};
        looked_up.check(
          [&] {
            return fmt::format("LookupTable<{}, {}>{{{}}}.lookup({}×{}{{{}}})",
                               test::type_name<Value>(), N, table.vec, test::type_name<Index>(),
                               IdxN, idxs.vec);
          },
          false);
      }
    }
  };
  grex::static_apply<1, max_shift + 2>(
    [&]<std::size_t... J> { (..., fix(grex::index_tag<1UZ << J>)); });
}
} // namespace

int main() {
  pcg_extras::seed_seq_from<std::random_device> seed_source{};
  test::Rng rng{seed_source};
  test::for_each_size<Value, max_shift>([&](auto /*vtag*/, auto stag) { run_simd(rng, stag); });
}
#endif
