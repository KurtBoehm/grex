// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#if !GREX_BACKEND_SCALAR
#include <array>
#include <cstddef>
#include <cstdlib>
#include <random>

#include <fmt/base.h>
#include <pcg_extras.hpp>

#include "grex/grex.hpp"

#include "defs.hpp"
#include "rng.hpp"

namespace test = grex::test;
using Value = grex::GREX_TEST_TYPE;
inline constexpr std::size_t repetitions = 256;

template<std::size_t tSize>
void run_simd(test::Rng& rng, grex::IndexTag<tSize> /*tag*/) {
  using VC = test::VectorChecker<Value, tSize>;

  auto dist = test::make_distribution<Value>();
  auto dval = [&](std::size_t /*dummy*/) { return dist(rng); };

  grex::static_apply<tSize>([&]<std::size_t... tIdxs> {
    VC vca{dval(tIdxs)...};
    VC vcb{dval(tIdxs)...};

    constexpr auto bzs = grex::static_apply<repetitions>([&]<std::size_t... tReps>() {
      test::Pcg32 pcg{};
      auto r = [&](auto /*dummy*/) { return grex::BlendZeroSelector(pcg.bounded_random(3)); };
      auto arr = [&](auto /*dummy*/) { return std::array{r(tIdxs)...}; };
      return std::array<std::array<grex::BlendZeroSelector, tSize>, repetitions>{arr(tReps)...};
    });
    constexpr auto bls = grex::static_apply<repetitions>([&]<std::size_t... tReps>() {
      test::Pcg32 pcg{};
      auto r = [&](auto /*dummy*/) { return grex::BlendSelector(pcg.bounded_random(3)); };
      auto arr = [&](auto /*dummy*/) { return std::array{r(tIdxs)...}; };
      return std::array<std::array<grex::BlendSelector, tSize>, repetitions>{arr(tReps)...};
    });

    auto fix = [&](grex::AnyIndexTag auto rep) {
      {
        const auto blended = grex::blend_zero<bzs[rep][tIdxs]...>(vca.vec);
        bool same = true;
        for (std::size_t i = 0; i < tSize; ++i) {
          const grex::BlendZeroSelector bz = bzs[rep][i];
          switch (bz) {
            case grex::keep_bz: same = same && blended[i] == vca.ref[i]; break;
            case grex::zero_bz: same = same && blended[i] == 0; break;
            case grex::any_bz: break;
            default: std::abort(); break;
          }
        }
        if (!same) {
          auto f = [&](std::size_t i) {
            return (bzs[rep][i] == grex::keep_bz) ? vca.ref[i] : Value{};
          };
          const std::array ref{f(tIdxs)...};

          fmt::print("grex::blend_zero<{}>({}) == {}, ref={};\n", fmt::join(bzs[rep], ", "),
                     test::type_name<Value>(), tSize, vca.vec, blended, ref);
          std::exit(EXIT_FAILURE);
        }
      }
      {
        const auto blended = grex::blend<bls[rep][tIdxs]...>(vca.vec, vcb.vec);
        bool same = true;
        for (std::size_t i = 0; i < tSize; ++i) {
          const grex::BlendSelector bl = bls[rep][i];
          switch (bl) {
            case grex::lhs_bl: same = same && blended[i] == vca.ref[i]; break;
            case grex::rhs_bl: same = same && blended[i] == vcb.ref[i]; break;
            case grex::any_bl: break;
            default: std::abort(); break;
          }
        }
        if (!same) {
          auto f = [&](std::size_t i) {
            return (bls[rep][i] != grex::rhs_bl) ? vca.ref[i] : vcb.ref[i];
          };
          const std::array ref{f(tIdxs)...};

          fmt::print("grex::blend<{}>({}, {}) == {}, ref={};\n", fmt::join(bls[rep], ", "),
                     test::type_name<Value>(), tSize, vca.vec, vcb.vec, blended, ref);
          std::exit(EXIT_FAILURE);
        }
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
