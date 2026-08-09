// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <array>
#include <cstddef>
#include <random>

#include <fmt/base.h>
#include <fmt/format.h>
#include <pcg_extras.hpp>

#include "grex/grex.hpp"

#include "defs.hpp"

namespace test = grex::test;
inline constexpr std::size_t repetitions = 4096;

#if !GREX_BACKEND_SCALAR
template<grex::Vectorizable T, std::size_t tSize>
void run_simd(test::Rng& rng, grex::TypeTag<T> /*tag*/, grex::IndexTag<tSize> /*tag*/) {
  using VC = test::VectorChecker<T, tSize>;
  using Vec = grex::Vector<T, tSize>;

  auto dist = test::make_distribution<T>();
  auto dval = [&] { return dist(rng); };

  for (std::size_t i = 0; i < repetitions; ++i) {
    grex::static_apply<tSize>([&]<std::size_t... tI>() {
      // load scalar
      {
        std::array<T, 1> buf{dist(rng)};
        test::check("load scalar", grex::load(buf.data(), grex::scalar_tag), buf[0], false);
      }
      // load full
      {
        std::array buf = test::random_array<T, tSize>(dval);
        VC checker{Vec::load(buf.data()), buf};
        checker.check("load", false);
      }
      {
        std::array buf = test::random_array<T, tSize>(dval);
        VC checker{grex::load(buf.data(), grex::full_tag<tSize>), buf};
        checker.check("load tagged", false);
      }

      // load full aligned
      {
        alignas(64) std::array buf = test::random_array<T, tSize>(dval);
        VC checker{Vec::load_aligned(buf.data()), buf};
        checker.check("load_aligned", false);
      }
      // there is no tagged version of aligned loading

      // load part
      {
        std::array buf = test::random_array<T, tSize>(dval);
        for (std::size_t j = 0; j <= tSize; ++j) {
          {
            VC checker{Vec::load_part(buf.data(), j), std::array{((tI < j) ? buf[tI] : T{})...}};
            checker.check("load_part", j, false);
          }
          // tagged
          {
            VC checker{grex::load(buf.data(), grex::part_tag<tSize>(j)),
                       std::array{((tI < j) ? buf[tI] : T{})...}};
            checker.check("load_part tagged", j, false);
          }
        }
      }
      {
        std::array buf = test::random_array<T, tSize>(dval);
        auto load_part_wrap = [&](grex::AnyIndexTag auto j) {
          {
            VC checker{Vec::load_part(buf.data(), j.value),
                       std::array{((tI < j) ? buf[tI] : T{})...}};
            checker.check("load_part", j.value, false);
          }
          {
            VC checker{Vec::load_part(buf.data(), j), std::array{((tI < j) ? buf[tI] : T{})...}};
            checker.check("load_part", j.value, false);
          }
          // tagged
          {
            VC checker{grex::load(buf.data(), grex::part_tag<tSize>(j.value)),
                       std::array{((tI < j) ? buf[tI] : T{})...}};
            checker.check("load_part tagged", j.value, false);
          }
          {
            VC checker{grex::load(buf.data(), grex::part_tag<tSize>(j)),
                       std::array{((tI < j) ? buf[tI] : T{})...}};
            checker.check("load_part tagged", j.value, false);
          }
        };
        grex::static_apply<tSize + 1>(
          [&]<std::size_t... tJ>() { (..., load_part_wrap(grex::index_tag<tJ>)); });
      }
    });
  }
}
#endif
template<grex::Vectorizable T>
void run_scalar(test::Rng& rng, grex::TypeTag<T> /*tag*/) {
  auto dist = test::make_distribution<T>();

  for (std::size_t i = 0; i < repetitions; ++i) {
    // load scalar
    {
      std::array<T, 1> buf{dist(rng)};
      test::check("load scalar", grex::load(buf.data(), grex::scalar_tag), buf[0], false);
    }
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
