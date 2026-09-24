// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <array>
#include <cstddef>
#include <random>

#include <fmt/base.h> // IWYU pragma: keep
#include <pcg_extras.hpp>

#include "grex/grex.hpp"

#include "defs.hpp"

namespace {
namespace test = grex::test;
inline constexpr std::size_t repetitions = 4096;

#if !GREX_BACKEND_SCALAR
template<grex::Vectorizable T, std::size_t N>
void run_simd(test::Rng& rng, grex::TypeTag<T> /*tag*/, grex::IndexTag<N> /*tag*/) {
  using VC = test::VectorChecker<T, N>;
  using MC = test::MaskChecker<T, N>;

  auto dist = test::make_distribution<T>();
  auto dval = [&] { return dist(rng); };
  std::uniform_int_distribution<int> bdist{0, 1};
  auto bval = [&](std::size_t /*dummy*/) { return static_cast<bool>(bdist(rng)); };

  grex::static_apply<N>([&]<std::size_t... I> {
    for (std::size_t i = 0; i < repetitions; ++i) {
      {
        const VC vc = VC::random(dval);
        test::check("vector extract run-time", std::array{vc.vec[I]...}, vc.ref,
                    {.verbose = false});
        test::check("vector extract compile-time", std::array{vc.vec[grex::index_tag<I>]...},
                    vc.ref, {.verbose = false});
        test::check("vector extract tuple-like", std::array{get<I>(vc.vec)...}, vc.ref,
                    {.verbose = false});
        test::check("vector extract_single", grex::extract_single(vc.vec), vc.ref[0],
                    {.verbose = false});
      }
      {
        const MC mc{bval(I)...};
        test::check("vector extract run-time", std::array{mc.mask[I]...}, mc.ref,
                    {.verbose = false});
        test::check("vector extract compile-time", std::array{mc.mask[grex::index_tag<I>]...},
                    mc.ref, {.verbose = false});
        test::check("vector extract tuple-like", std::array{get<I>(mc.mask)...}, mc.ref,
                    {.verbose = false});
      }
    }
  });
}
#endif
template<grex::Vectorizable T>
void run_scalar(test::Rng& rng, grex::TypeTag<T> /*tag*/) {
  auto dist = test::make_distribution<T>(); // NOLINT(misc-const-correctness)

  for (std::size_t i = 0; i < repetitions; ++i) {
    const T value = dist(rng);
    test::check("scalar extract", grex::extract_single(value), value, {.verbose = false});
  }
}
} // namespace

int main() {
  pcg_extras::seed_seq_from<std::random_device> seed_source{};
  test::Rng rng{seed_source};
#if !GREX_BACKEND_SCALAR
  test::run_types_sizes([&](auto vtag, auto stag) { run_simd(rng, vtag, stag); });
#endif
  test::run_types([&](auto tag) { run_scalar(rng, tag); });
}
