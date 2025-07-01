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
  using Vec = grex::Vector<T, tSize>;

  auto dist = test::make_distribution<T>();
  auto dval = [&](std::size_t /*dummy*/) { return dist(rng); };

  for (std::size_t i = 0; i < repetitions; ++i) {
    grex::static_apply<tSize>([&]<std::size_t... tIdxs>() {
      {
        std::array buf{dval(tIdxs)...};
        VC checker{Vec::load(buf.data()), buf};
        checker.check("load", false);
      }
      {
        alignas(64) std::array buf{dval(tIdxs)...};
        VC checker{Vec::load_aligned(buf.data()), buf};
        checker.check("load_aligned", false);
      }
      {
        std::array buf{dval(tIdxs)...};
        for (std::size_t i = 0; i <= tSize; ++i) {
          VC checker{Vec::load_part(buf.data(), i),
                     std::array{((tIdxs < i) ? buf[tIdxs] : T{})...}};
          checker.check("load_part", false);
        }
      }
    });

    grex::static_apply<tSize>([&]<std::size_t... tIdxs>() {
      VC checker{dval(tIdxs)...};
      {
        std::array<T, tSize> buf{};
        checker.vec.store(buf.data());
        test::check("store", buf, checker.ref, false);
      }
      {
        alignas(64) std::array<T, tSize> buf{};
        checker.vec.store_aligned(buf.data());
        test::check("store_aligned", buf, checker.ref, false);
      }
      {
        for (std::size_t i = 0; i <= tSize; ++i) {
          std::array<T, tSize> buf{};
          checker.vec.store_part(buf.data(), i);
          test::check("store_part", buf, std::array{((tIdxs < i) ? checker.ref[tIdxs] : T{})...},
                      false);
        }
      }
    });
  }
}

int main() {
  pcg_extras::seed_seq_from<std::random_device> seed_source{};
  test::Rng rng{seed_source};
  test::run_types_sizes([&](auto vtag, auto stag) { run(rng, vtag, stag); });
}
