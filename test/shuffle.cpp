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

namespace {
namespace test = grex::test;
using Value = grex::GREX_TEST_TYPE;
inline constexpr std::size_t repetitions = 256;

/**
 * Announces the shuffle that is about to be checked.
 *
 * Kept out of line: formatting the index list and the whole vector would otherwise be inlined and
 * optimized into each of the `repetitions` unrolled copies of the test body.
 */
template<std::size_t N>
[[gnu::noinline]] void announce_shuffle(const std::array<grex::ShuffleIndex, N>& idxs,
                                        const grex::Vector<Value, N>& base) {
  fmt::print("grex::shuffle<{}>({}×{}{{{}}});\n", fmt::join(idxs, ", "), test::type_name<Value>(),
             N, fmt::join(base, ", "));
}

/** Reports a failed `shuffle` and terminates, kept out of line as in `announce_shuffle`. */
template<std::size_t N, typename Shuffled>
[[gnu::cold, gnu::noinline]] void fail_shuffle(const std::array<grex::ShuffleIndex, N>& idxs,
                                               const std::array<Value, N>& baseref,
                                               const Shuffled& shuf) {
  std::array<Value, N> ref{};
  for (std::size_t i = 0; i < N; ++i) {
    const auto sh = idxs[i];
    ref[i] = grex::is_index(sh) ? baseref[grex::u8(sh)] : Value{};
  }
  fmt::print(fmt::fg(fmt::terminal_color::red), "shuffle({}, {}) != {} vs. {}\n", idxs, baseref,
             shuf, ref);
  std::exit(EXIT_FAILURE);
}

template<std::size_t N>
void run_simd(test::Rng& rng, grex::IndexTag<N> /*tag*/) {
  using VC = test::VectorChecker<Value, N>;

  auto dist = test::make_distribution<Value>(); // NOLINT(*-const-correctness)
  const auto dval = [&] { return dist(rng); };

  grex::static_apply<N>([&]<std::size_t... I> {
    VC base = VC::random(dval);

    constexpr auto idxs = grex::static_apply<repetitions>([&]<std::size_t... Reps> {
      test::Pcg32 pcg{};
      auto r = [&](auto /*dummy*/) {
        const auto v = pcg.bounded_random(N + 2);
        switch (v) {
          case N: return grex::any_sh;
          case N + 1: return grex::zero_sh;
          default: return grex::ShuffleIndex{static_cast<grex::u8>(v)};
        }
      };
      auto arr = [&](auto /*dummy*/) { return std::array{r(I)...}; };
      return std::array<std::array<grex::ShuffleIndex, N>, repetitions>{arr(Reps)...};
    });

    auto fix = [&](grex::AnyIndexTag auto rep) {
      announce_shuffle(idxs[rep], base.vec);
      const auto shuf = grex::shuffle<idxs[rep][I]...>(base.vec);
      bool same = true;
      for (std::size_t i = 0; i < N; ++i) {
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
      [&]<std::size_t... Reps> { (..., fix(grex::index_tag<Reps>)); });
  });
}
} // namespace

int main() {
  pcg_extras::seed_seq_from<std::random_device> seed_source{};
  test::Rng rng{seed_source};
  test::for_each_size<Value>([&](auto /*vtag*/, auto stag) { run_simd(rng, stag); });
}
#endif
