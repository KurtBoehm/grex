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
#include <random>

#include <pcg_extras.hpp>

#include "defs.hpp"

namespace {
namespace test = grex::test;
inline constexpr std::size_t repetitions = 4096;

template<grex::Vectorizable T, std::size_t N>
void run_simd(test::Rng& rng, grex::TypeTag<T> /*tag*/, grex::IndexTag<N> /*tag*/) {
  using Vec = grex::Vector<T, N>;
  using VC = test::VectorChecker<T, N>;

  auto dist = test::make_distribution<T>();
  auto dval = [&] { return dist(rng); };

  // expand values
  for (std::size_t i = 0; i < repetitions; ++i) {
    const T value = dist(rng);
    test::check("expanded_any", value, Vec::expanded_any(value)[0], {.verbose = false});
    test::check("expand_any vector tagged", value, expand_any(value, grex::full_tag<N>)[0],
                {.verbose = false});
    test::check("expand_any both tagged", expand_any(value, grex::scalar_tag),
                expand_any(value, grex::full_tag<N>)[0], {.verbose = false});
  }
  grex::static_apply<N>([&]<std::size_t... I> {
    for (std::size_t i = 0; i < repetitions; ++i) {
      const T value = dist(rng);

      const test::VectorChecker<T, N> checker{
        Vec::expanded_zero(value),
        std::array{((I == 0) ? value : T{})...},
      };
      checker.check("expanded_zero", {.verbose = false});

      const test::VectorChecker<T, N> checker_tagged_vec{
        grex::expand_zero(value, grex::full_tag<N>),
        std::array{((I == 0) ? value : T{})...},
      };
      checker_tagged_vec.check("expanded_zero vector tagged", {.verbose = false});

      const test::VectorChecker<T, N> checker_tagged_both{
        grex::expand_zero(value, grex::full_tag<N>),
        std::array{((I == 0) ? grex::expand_zero(value, grex::scalar_tag) : T{})...},
      };
      checker_tagged_both.check("expanded_zero both tagged", {.verbose = false});
    }
  });

  // expand vectors
  auto expav = [&]<std::size_t DstN>(grex::IndexTag<DstN> /*tag*/) {
    using VDC = test::VectorChecker<T, DstN>;

    fmt::print("size: {}\n", DstN);
    grex::static_apply<N>([&]<std::size_t... I> {
      grex::static_apply<DstN>([&]<std::size_t... DstI> {
        // Super-native expansion leads to warnings on GCC
        for (std::size_t i = 0; i < repetitions; ++i) {
          const VC checker = VC::random(dval);
          {
            const auto v = checker.vec.expand_any(grex::index_tag<DstN>);
            test::check_msg("expand_any", (... && (v[I] == checker.ref[I])), v, checker.ref, false);
          }
          {
            const VDC dchecker{
              checker.vec.expand_zero(grex::index_tag<DstN>),
              {((DstI < N) ? checker.ref[DstI] : T{})...},
            };
            dchecker.check("expand_zero", {.verbose = false});
          }
        }
      });
    });
  };
  grex::static_apply<std::bit_width(N) - 1, std::bit_width(grex::max_native_size<T>) + 1>(
    [&]<std::size_t... Logs> { (..., expav(grex::index_tag<1ULL << Logs>)); });
}
} // namespace

int main() {
  pcg_extras::seed_seq_from<std::random_device> seed_source{};
  test::Rng rng{seed_source};
  test::run_types_sizes([&](auto vtag, auto stag) { run_simd(rng, vtag, stag); });
}
#endif
