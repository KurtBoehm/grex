// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <array>
#include <cstddef>
#include <random>

#include <pcg_extras.hpp>

#include "grex/grex.hpp"

#include "defs.hpp"

namespace test = grex::test;
inline constexpr std::size_t repetitions = 4096;

template<grex::Vectorizable T, std::size_t tSize>
void run(test::Rng& rng, grex::TypeTag<T> /*tag*/, grex::IndexTag<tSize> /*tag*/) {
  using VC = test::VectorChecker<T, tSize>;
  using MC = test::MaskChecker<T, tSize>;
  using Mask = grex::Mask<T, tSize>;

  auto dist = test::make_distribution<T>();
  auto dval = [&](std::size_t /*dummy*/) { return dist(rng); };
  std::uniform_int_distribution<int> bdist{0, 1};
  auto bval = [&](std::size_t /*dummy*/) { return bool(bdist(rng)); };

  grex::static_apply<tSize>([&]<std::size_t... tIdxs>() {
    for (std::size_t i = 0; i < repetitions; ++i) {
      // vector
      {
        VC checker{};
        checker.check("vector zeros", false);
      }
      {
        VC checker{dval(0)};
        checker.check("vector broadcast", false);
      }
      {
        VC checker{grex::Vector<T, tSize>::indices(), std::array{T(tIdxs)...}};
        checker.check("vector indices", false);
      }
      {
        VC checker{dval(tIdxs)...};
        checker.check("vector set", false);
      }
      {
        const VC base{dval(tIdxs)...};
        for (std::size_t j = 0; j < tSize; ++j) {
          const auto val = dval(j);
          VC v{base.vec.insert(j, val), std::array{((tIdxs == j) ? val : base.ref[tIdxs])...}};
          v.check("vector insert", false);
        }
        for (std::size_t j = 0; j <= tSize; ++j) {
          VC v{base.vec.cutoff(j), std::array{((tIdxs < j) ? base.ref[tIdxs] : T(0))...}};
          v.check("vector cutoff", false);
        }
      }

      // mask
      {
        MC checker{};
        checker.check("mask zeros", false);
      }
      {
        auto f = [](std::size_t /*dummy*/) { return true; };
        test::MaskChecker checker{grex::Mask<T, tSize>::ones(), std::array{f(tIdxs)...}};
        checker.check("mask ones", false);
      }
      {
        MC checker{bval(0)};
        checker.check("mask broadcast true", false);
      }
      {
        MC checker{bval(tIdxs)...};
        checker.check("mask set", false);
      }
      {
        const MC base{bval(tIdxs)...};
        for (std::size_t j = 0; j < tSize; ++j) {
          const bool val = bval(j);
          MC v{base.mask.insert(j, val), std::array{((tIdxs == j) ? val : base.ref[tIdxs])...}};
          v.check("mask insert", false);
        }
      }
      {
        for (std::size_t j = 0; j <= tSize; ++j) {
          MC v{Mask::cutoff_mask(j), std::array{(tIdxs < j)...}};
          v.check("mask cutoff_mask", false);
        }
      }
    }
  });
}

int main() {
  pcg_extras::seed_seq_from<std::random_device> seed_source{};
  test::Rng rng{seed_source};
  test::run_types_sizes([&](auto vtag, auto stag) { run(rng, vtag, stag); });
}
