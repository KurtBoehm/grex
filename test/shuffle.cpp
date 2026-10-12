// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <fmt/base.h>

#include "grex/grex.hpp"

#if !GREX_BACKEND_SCALAR
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <random>

#include <fmt/color.h>
#include <fmt/format.h>
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
[[GREX_NEVER_INLINE]] void announce_shuffle(const std::array<grex::ShuffleIndex, N>& idxs,
                                            const grex::Vector<Value, N>& base) {
  fmt::print("grex::shuffle<{}>({}×{}{{{}}});\n", fmt::join(idxs, ", "), test::type_name<Value>(),
             N, fmt::join(base, ", "));
}

/** Reports a failed `shuffle` and terminates, kept out of line as in `announce_shuffle`. */
template<std::size_t N, typename Shuffled>
[[GREX_COLD, GREX_NEVER_INLINE]] void fail_shuffle(const std::array<grex::ShuffleIndex, N>& idxs,
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

  // broadcast_lane
  grex::static_apply<N>([&]<std::size_t... I> {
    VC base = VC::random(dval);
    const auto broadcast_lane = [&](grex::AnyIndexTag auto lane) {
      std::array<Value, N> bc_array{};
      std::ranges::fill(bc_array, base.vec[lane]);

      VC{base.vec.broadcast_lane(lane), bc_array}.check("broadcast_lane");
      VC{grex::broadcast_lane<lane>(base.vec), bc_array}.check("broadcast_lane");
    };
    (..., broadcast_lane(grex::index_tag<I>));
  });

  // shift and rotate
  grex::static_apply<N + 1>([&]<std::size_t... I> {
    VC base0 = VC::random(dval);
    VC base1 = VC::random(dval);

    const auto shrot_lane = [&](grex::AnyIndexTag auto shift) {
      std::array<Value, N> shift_down_array{};
      std::array<Value, N> shift_up_array{};
      std::array<Value, N> rot_down_array{};
      std::array<Value, N> rot_up_array{};
      std::array<Value, N> crot_down_array{};
      std::array<Value, N> crot_up_array{};

      for (std::size_t i = 0; i < N; ++i) {
        shift_down_array[i] = (i + shift < N) ? base0.ref[i + shift] : 0;
        shift_up_array[i] = (i >= shift) ? base0.ref[i - shift] : 0;
        rot_down_array[i] = base0.ref[(i + shift) % N];
        rot_up_array[i] = base0.ref[(i + N - shift) % N];
        crot_down_array[i] = (i + shift < N) ? base0.ref[i + shift] : base1.ref[i + shift - N];
        crot_up_array[i] = (i < shift) ? base0.ref[N + i - shift] : base1.ref[i - shift];
      }

      VC{base0.vec.slide_down(shift), shift_down_array}.check(
        [&] { return fmt::format("{}.slide_down(index_tag<{}>)", base0.vec, shift.value); });
      VC{base0.vec.slide_up(shift), shift_up_array}.check(
        [&] { return fmt::format("{}.slide_up(index_tag<{}>)", base0.vec, shift.value); });
      VC{grex::slide_down<shift>(base0.vec), shift_down_array}.check(
        [&] { return fmt::format("slide_down<{}>({})", shift.value, base0.vec); });
      VC{grex::slide_up<shift>(base0.vec), shift_up_array}.check(
        [&] { return fmt::format("slide_up<{}>({})", shift.value, base0.vec); });

      VC{base0.vec.rotate_down(shift), rot_down_array}.check(
        [&] { return fmt::format("{}.rotate_down(index_tag<{}>)", base0.vec, shift.value); });
      VC{base0.vec.rotate_up(shift), rot_up_array}.check(
        [&] { return fmt::format("{}.rotate_up(index_tag<{}>)", base0.vec, shift.value); });
      VC{grex::rotate_down<shift>(base0.vec), rot_down_array}.check(
        [&] { return fmt::format("rotate_down<{}>({})", shift.value, base0.vec); });
      VC{grex::rotate_up<shift>(base0.vec), rot_up_array}.check(
        [&] { return fmt::format("rotate_up<{}>({})", shift.value, base0.vec); });

      VC{grex::slide_down<shift>(base0.vec, base1.vec), crot_down_array}.check(
        [&] { return fmt::format("slide_down<{}>({}, {})", shift.value, base0.vec, base1.vec); });
      VC{grex::slide_up<shift>(base0.vec, base1.vec), crot_up_array}.check(
        [&] { return fmt::format("slide_up<{}>({}, {})", shift.value, base0.vec, base1.vec); });
    };
    (..., shrot_lane(grex::index_tag<I>));
  });

  // shuffle
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

    const auto fix = [&](grex::AnyIndexTag auto rep) {
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
