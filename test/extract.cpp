// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <array>
#include <cstddef>
#include <random>

#include <fmt/base.h>
#include <pcg_extras.hpp>

#include "grex/grex.hpp"

#include "defs.hpp"

namespace test = grex::test;
inline constexpr std::size_t repetitions = 4096;

#if !GREX_BACKEND_SCALAR
template<grex::Vectorizable T, std::size_t tSize>
void run_simd(test::Rng& rng, grex::TypeTag<T> /*tag*/, grex::IndexTag<tSize> /*tag*/) {
  using VC = test::VectorChecker<T, tSize>;
  using MC = test::MaskChecker<T, tSize>;

  auto dist = test::make_distribution<T>();
  auto dval = [&](std::size_t /*dummy*/) { return dist(rng); };
  std::uniform_int_distribution<int> bdist{0, 1};
  auto bval = [&](std::size_t /*dummy*/) { return bool(bdist(rng)); };

  grex::static_apply<tSize>([&]<std::size_t... tIdxs>() {
    for (std::size_t i = 0; i < repetitions; ++i) {
      {
        const VC vc{dval(tIdxs)...};
        test::check("vector extract run-time", std::array{vc.vec[tIdxs]...}, vc.ref, false);
        test::check("vector extract compile-time", std::array{vc.vec[grex::index_tag<tIdxs>]...},
                    vc.ref, false);
        test::check("vector extract tuple-like", std::array{get<tIdxs>(vc.vec)...}, vc.ref, false);
        test::check("vector extract_single", grex::extract_single(vc.vec), vc.ref[0], false);
      }
      {
        const MC mc{bval(tIdxs)...};
        test::check("vector extract run-time", std::array{mc.mask[tIdxs]...}, mc.ref, false);
        test::check("vector extract compile-time", std::array{mc.mask[grex::index_tag<tIdxs>]...},
                    mc.ref, false);
        test::check("vector extract tuple-like", std::array{get<tIdxs>(mc.mask)...}, mc.ref, false);
      }
    }
  });
}
#endif
template<grex::Vectorizable T>
void run_scalar(test::Rng& rng, grex::TypeTag<T> /*tag*/) {
  auto dist = test::make_distribution<T>();

  for (std::size_t i = 0; i < repetitions; ++i) {
    const T value = dist(rng);
    test::check("scalar extract", grex::extract_single(value), value, false);
  }
}

int main() {
  pcg_extras::seed_seq_from<std::random_device> seed_source{};
  test::Rng rng{seed_source};
#if !GREX_BACKEND_SCALAR
  test::run_types_sizes([&](auto vtag, auto stag) { run_simd(rng, vtag, stag); });
#endif
  test::run_types([&](auto tag) { run_scalar(rng, tag); });
}
