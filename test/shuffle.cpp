// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <fmt/base.h>

#include "grex/grex.hpp"

#if !GREX_BACKEND_SCALAR
#include <array>
#include <cstddef>
#include <cstdlib>
#include <random>

#include <fmt/color.h>
#include <pcg_extras.hpp>

#include "defs.hpp"
#include "rng.hpp"

namespace test = grex::test;
using Value = grex::GREX_TEST_TYPE;
inline constexpr std::size_t repetitions = 256;

/**
 * Announces the shuffle that is about to be checked.
 *
 * Kept out of line: formatting the index list and the whole vector would otherwise be inlined and
 * optimized into each of the `repetitions` unrolled copies of the test body.
 */
template<std::size_t tSize>
[[gnu::noinline]] void announce_shuffle(const std::array<grex::ShuffleIndex, tSize>& idxs,
                                        const grex::Vector<Value, tSize>& base) {
  fmt::print("grex::shuffle<{}>({}x{}{{{}}});\n", fmt::join(idxs, ", "), test::type_name<Value>(),
             tSize, fmt::join(base, ", "));
}

/** Reports a failed `shuffle` and terminates, kept out of line as in `announce_shuffle`. */
template<std::size_t tSize, typename TShuffled>
[[gnu::cold, gnu::noinline]] void fail_shuffle(const std::array<grex::ShuffleIndex, tSize>& idxs,
                                               const std::array<Value, tSize>& baseref,
                                               const TShuffled& shuf) {
  std::array<Value, tSize> ref{};
  for (std::size_t i = 0; i < tSize; ++i) {
    const auto sh = idxs[i];
    ref[i] = grex::is_index(sh) ? baseref[grex::u8(sh)] : Value{};
  }
  fmt::print(fmt::fg(fmt::terminal_color::red), "shuffle({}, {}) != {} vs. {}\n", idxs, baseref,
             shuf, ref);
  std::exit(EXIT_FAILURE);
}

template<std::size_t tSize>
void run_simd(test::Rng& rng, grex::IndexTag<tSize> /*tag*/) {
  using VC = test::VectorChecker<Value, tSize>;

  auto dist = test::make_distribution<Value>();
  auto dval = [&] { return dist(rng); };

  grex::static_apply<tSize>([&]<std::size_t... tIdxs> {
    VC base = VC::random(dval);

    constexpr auto idxs = grex::static_apply<repetitions>([&]<std::size_t... tReps>() {
      test::Pcg32 pcg{};
      auto r = [&](auto /*dummy*/) {
        const auto v = pcg.bounded_random(tSize + 2);
        switch (v) {
          case tSize: return grex::any_sh;
          case tSize + 1: return grex::zero_sh;
          default: return grex::ShuffleIndex{grex::u8(v)};
        }
      };
      auto arr = [&](auto /*dummy*/) { return std::array{r(tIdxs)...}; };
      return std::array<std::array<grex::ShuffleIndex, tSize>, repetitions>{arr(tReps)...};
    });

    auto fix = [&](grex::AnyIndexTag auto rep) {
      announce_shuffle(idxs[rep], base.vec);
      const auto shuf = grex::shuffle<idxs[rep][tIdxs]...>(base.vec);
      bool same = true;
      for (std::size_t i = 0; i < tSize; ++i) {
        const auto sh = idxs[rep][i];
        switch (sh) {
          case grex::any_sh: break;
          case grex::zero_sh: same = same && shuf[i] == 0; break;
          default: same = same && shuf[i] == base.ref[grex::u8(sh)]; break;
        }
      }
      if (!same) {
        fail_shuffle(idxs[rep], base.ref, shuf);
      }
    };
    grex::static_apply<repetitions>(
      [&]<std::size_t... tReps>() { (..., fix(grex::index_tag<tReps>)); });
  });
}

int main() {
  pcg_extras::seed_seq_from<std::random_device> seed_source{};
  test::Rng rng{seed_source};
  test::for_each_size<Value>([&](auto /*vtag*/, auto stag) { run_simd(rng, stag); });
}
#endif
