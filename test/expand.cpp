// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#if !GREX_BACKEND_SCALAR
#include <array>
#include <bit>
#include <cstddef>
#include <random>

#include <fmt/base.h>
#include <pcg_extras.hpp>

#include "grex/grex.hpp"

#include "defs.hpp"

namespace test = grex::test;
inline constexpr std::size_t repetitions = 4096;

template<grex::Vectorizable T, std::size_t tSize>
void run_simd(test::Rng& rng, grex::TypeTag<T> /*tag*/, grex::IndexTag<tSize> /*tag*/) {
  using Vec = grex::Vector<T, tSize>;
  using VC = test::VectorChecker<T, tSize>;

  auto dist = test::make_distribution<T>();
  auto dval = [&](std::size_t /*dummy*/) { return dist(rng); };

  // expand values
  for (std::size_t i = 0; i < repetitions; ++i) {
    const T value = dist(rng);
    test::check("expanded_any", value, Vec::expanded_any(value)[0], false);
    test::check("expand_any vector tagged", value, expand_any(value, grex::full_tag<tSize>)[0],
                false);
    test::check("expand_any both tagged", expand_any(value, grex::scalar_tag),
                expand_any(value, grex::full_tag<tSize>)[0], false);
  }
  grex::static_apply<tSize>([&]<std::size_t... tIdxs>() {
    for (std::size_t i = 0; i < repetitions; ++i) {
      const T value = dist(rng);

      const test::VectorChecker<T, tSize> checker{
        Vec::expanded_zero(value),
        std::array{((tIdxs == 0) ? value : T{})...},
      };
      checker.check("expanded_zero", false);

      const test::VectorChecker<T, tSize> checker_tagged_vec{
        grex::expand_zero(value, grex::full_tag<tSize>),
        std::array{((tIdxs == 0) ? value : T{})...},
      };
      checker_tagged_vec.check("expanded_zero vector tagged", false);

      const test::VectorChecker<T, tSize> checker_tagged_both{
        grex::expand_zero(value, grex::full_tag<tSize>),
        std::array{((tIdxs == 0) ? grex::expand_zero(value, grex::scalar_tag) : T{})...},
      };
      checker_tagged_both.check("expanded_zero both tagged", false);
    }
  });

  // expand vectors
  auto expav = [&]<std::size_t tDstSize>(grex::IndexTag<tDstSize> /*tag*/) {
    using VDC = test::VectorChecker<T, tDstSize>;

    fmt::print("size: {}\n", tDstSize);
    grex::static_apply<tSize>([&]<std::size_t... tIdxs> {
      grex::static_apply<tDstSize>([&]<std::size_t... tDstIdxs> {
        // Super-native expansion leads to warnings on GCC
        GREX_DIAGNOSTIC_UNINIT_PUSH()
        for (std::size_t i = 0; i < repetitions; ++i) {
          VC checker{dval(tIdxs)...};
          {
            const auto v = checker.vec.expand_any(grex::index_tag<tDstSize>);
            test::check_msg("expand_any", (... && (v[tIdxs] == checker.ref[tIdxs])), v, checker.ref,
                            false);
          }
          {
            VDC dchecker{
              checker.vec.expand_zero(grex::index_tag<tDstSize>),
              {((tDstIdxs < tSize) ? checker.ref[tDstIdxs] : T{})...},
            };
            dchecker.check("expand_zero", false);
          }
        }
        GREX_DIAGNOSTIC_UNINIT_POP()
      });
    });
  };
  grex::static_apply<std::bit_width(tSize) - 1, std::bit_width(grex::max_native_size<T>) + 1>(
    [&]<std::size_t... tLogs> { (..., expav(grex::index_tag<1ULL << tLogs>)); });
}

int main() {
  pcg_extras::seed_seq_from<std::random_device> seed_source{};
  test::Rng rng{seed_source};
  test::run_types_sizes([&](auto vtag, auto stag) { run_simd(rng, vtag, stag); });
}
#endif
