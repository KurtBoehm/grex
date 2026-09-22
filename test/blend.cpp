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
template<std::size_t tSize, typename TBlended>
[[gnu::cold, gnu::noinline]] void
fail_blend_zero(const std::array<grex::BlendZeroSelector, tSize>& sels,
                const grex::Vector<Value, tSize>& a, const std::array<Value, tSize>& aref,
                const TBlended& blended) {
  std::array<Value, tSize> ref{};
  for (std::size_t i = 0; i < tSize; ++i) {
    ref[i] = (sels[i] == grex::keep_bz) ? aref[i] : Value{};
  }
  fmt::print("grex::blend_zero<{}>({}×{}, {}) == {}, ref={};\n", fmt::join(sels, ", "),
             test::type_name<Value>(), tSize, a, blended, ref);
  std::exit(EXIT_FAILURE);
}

/** Reports a failed `blend` and terminates, see `fail_blend_zero`. */
template<std::size_t tSize, typename TBlended>
[[gnu::cold, gnu::noinline]] void
fail_blend(const std::array<grex::BlendSelector, tSize>& sels, const grex::Vector<Value, tSize>& a,
           const std::array<Value, tSize>& aref, const grex::Vector<Value, tSize>& b,
           const std::array<Value, tSize>& bref, const TBlended& blended) {
  std::array<Value, tSize> ref{};
  for (std::size_t i = 0; i < tSize; ++i) {
    ref[i] = (sels[i] != grex::rhs_bl) ? aref[i] : bref[i];
  }
  fmt::print("grex::blend<{}>({}×{}, {}, {}) == {}, ref={};\n", fmt::join(sels, ", "),
             test::type_name<Value>(), tSize, a, b, blended, ref);
  std::exit(EXIT_FAILURE);
}

template<std::size_t tSize>
void run_simd(test::Rng& rng, grex::IndexTag<tSize> /*tag*/) {
  using VC = test::VectorChecker<Value, tSize>;

  auto dist = test::make_distribution<Value>();
  auto dval = [&] { return dist(rng); };

  grex::static_apply<tSize>([&]<std::size_t... tIdxs> {
    VC vca = VC::random(dval);
    VC vcb = VC::random(dval);

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
          fail_blend_zero(bzs[rep], vca.vec, vca.ref, blended);
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
          fail_blend(bls[rep], vca.vec, vca.ref, vcb.vec, vcb.ref, blended);
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
