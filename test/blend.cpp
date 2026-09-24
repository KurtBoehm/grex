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

#include <pcg_extras.hpp>

#include "defs.hpp"
#include "rng.hpp"

namespace test = grex::test;
using Value = grex::GREX_TEST_TYPE;
inline constexpr std::size_t repetitions = 256;

/**
 * Reports a failed `blend_zero` and terminates.
 *
 * Kept out of line and cold: the reporting code formats whole vectors, and would otherwise be
 * inlined and optimized into each of the `repetitions` unrolled copies of the test body.
 */
template<std::size_t N, typename Blended>
[[gnu::cold, gnu::noinline]] void
fail_blend_zero(const std::array<grex::BlendZeroSelector, N>& sels, const grex::Vector<Value, N>& a,
                const std::array<Value, N>& aref, const Blended& blended) {
  std::array<Value, N> ref{};
  for (std::size_t i = 0; i < N; ++i) {
    ref[i] = (sels[i] == grex::keep_bz) ? aref[i] : Value{};
  }
  fmt::print("grex::blend_zero<{}>({}×{}, {}) == {}, ref={};\n", fmt::join(sels, ", "),
             test::type_name<Value>(), N, a, blended, ref);
  std::exit(EXIT_FAILURE);
}

/** Reports a failed `blend` and terminates, see `fail_blend_zero`. */
template<std::size_t N, typename Blended>
[[gnu::cold, gnu::noinline]] void
fail_blend(const std::array<grex::BlendSelector, N>& sels, const grex::Vector<Value, N>& a,
           const std::array<Value, N>& aref, const grex::Vector<Value, N>& b,
           const std::array<Value, N>& bref, const Blended& blended) {
  std::array<Value, N> ref{};
  for (std::size_t i = 0; i < N; ++i) {
    ref[i] = (sels[i] != grex::rhs_bl) ? aref[i] : bref[i];
  }
  fmt::print("grex::blend<{}>({}×{}, {}, {}) == {}, ref={};\n", fmt::join(sels, ", "),
             test::type_name<Value>(), N, a, b, blended, ref);
  std::exit(EXIT_FAILURE);
}

template<std::size_t N>
void run_simd(test::Rng& rng, grex::IndexTag<N> /*tag*/) {
  using VC = test::VectorChecker<Value, N>;

  auto dist = test::make_distribution<Value>();
  auto dval = [&] { return dist(rng); };

  grex::static_apply<N>([&]<std::size_t... I> {
    VC vca = VC::random(dval);
    VC vcb = VC::random(dval);

    constexpr auto bzs = grex::static_apply<repetitions>([&]<std::size_t... Reps> {
      test::Pcg32 pcg{};
      auto r = [&](auto /*dummy*/) { return grex::BlendZeroSelector(pcg.bounded_random(3)); };
      auto arr = [&](auto /*dummy*/) { return std::array{r(I)...}; };
      return std::array<std::array<grex::BlendZeroSelector, N>, repetitions>{arr(Reps)...};
    });
    constexpr auto bls = grex::static_apply<repetitions>([&]<std::size_t... Reps> {
      test::Pcg32 pcg{};
      auto r = [&](auto /*dummy*/) { return grex::BlendSelector(pcg.bounded_random(3)); };
      auto arr = [&](auto /*dummy*/) { return std::array{r(I)...}; };
      return std::array<std::array<grex::BlendSelector, N>, repetitions>{arr(Reps)...};
    });

    auto fix = [&](grex::AnyIndexTag auto rep) {
      {
        const auto blended = grex::blend_zero<bzs[rep][I]...>(vca.vec);
        bool same = true;
        for (std::size_t i = 0; i < N; ++i) {
          const grex::BlendZeroSelector bz = bzs[rep][i];
          switch (bz) {
            case grex::keep_bz: same = same && blended[i] == vca.ref[i]; break;
            case grex::zero_bz: same = same && blended[i] == 0; break;
            case grex::any_bz: break;
            default: std::abort(); break;
          }
        }
        if (!same) {
          fail_blend_zero(bzs[rep], vca.vec, vca.ref, blended);
        }
      }
      {
        const auto blended = grex::blend<bls[rep][I]...>(vca.vec, vcb.vec);
        bool same = true;
        for (std::size_t i = 0; i < N; ++i) {
          const grex::BlendSelector bl = bls[rep][i];
          switch (bl) {
            case grex::lhs_bl: same = same && blended[i] == vca.ref[i]; break;
            case grex::rhs_bl: same = same && blended[i] == vcb.ref[i]; break;
            case grex::any_bl: break;
            default: std::abort(); break;
          }
        }
        if (!same) {
          fail_blend(bls[rep], vca.vec, vca.ref, vcb.vec, vcb.ref, blended);
        }
      }
    };
    grex::static_apply<repetitions>(
      [&]<std::size_t... Reps> { (..., fix(grex::index_tag<Reps>)); });
  });
}

int main() {
  pcg_extras::seed_seq_from<std::random_device> seed_source{};
  test::Rng rng{seed_source};
  test::for_each_size<Value>([&](auto /*vtag*/, auto stag) { run_simd(rng, stag); });
}
#endif
