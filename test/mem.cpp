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
  auto dval = [&](std::size_t /*dummy*/) { return dist(rng); };

  for (std::size_t i = 0; i < repetitions; ++i) {
    grex::static_apply<tSize>([&]<std::size_t... tIdxs>() {
      // load scalar
      {
        std::array<T, 1> buf{dist(rng)};
        test::check("load scalar", grex::load(buf.data(), grex::scalar_tag), buf[0], false);
      }
      // load full
      {
        std::array buf{dval(tIdxs)...};
        VC checker{Vec::load(buf.data()), buf};
        checker.check("load", false);
      }
      {
        std::array buf{dval(tIdxs)...};
        VC checker{grex::load(buf.data(), grex::full_tag<tSize>), buf};
        checker.check("load tagged", false);
      }

      // load full aligned
      {
        alignas(64) std::array buf{dval(tIdxs)...};
        VC checker{Vec::load_aligned(buf.data()), buf};
        checker.check("load_aligned", false);
      }
      // there is no tagged version of aligned loading

      // load part
      {
        std::array buf{dval(tIdxs)...};
        for (std::size_t j = 0; j <= tSize; ++j) {
          VC checker{Vec::load_part(buf.data(), j),
                     std::array{((tIdxs < j) ? buf[tIdxs] : T{})...}};
          checker.check("load_part", j, false);
          // tagged
          VC tchecker{grex::load(buf.data(), grex::part_tag<tSize>(j)),
                      std::array{((tIdxs < j) ? buf[tIdxs] : T{})...}};
          tchecker.check("load_part tagged", j, false);
        }
      }
    });

    grex::static_apply<tSize>([&]<std::size_t... tIdxs>() {
      VC checker{dval(tIdxs)...};
      // store scalar
      {
        std::array<T, 1> buf{};
        const T val = dist(rng);
        grex::store(buf.data(), val, grex::scalar_tag);
        test::check("store scalar", buf[0], val, false);
      }

      // store full
      {
        std::array<T, tSize> buf{};
        checker.vec.store(buf.data());
        test::check("store", buf, checker.ref, false);
      }
      {
        std::array<T, tSize> buf{};
        grex::store(buf.data(), checker.vec, grex::typed_full_tag<T, tSize>);
        test::check("store tagged", buf, checker.ref, false);
      }

      // store full aligned
      {
        alignas(64) std::array<T, tSize> buf{};
        checker.vec.store_aligned(buf.data());
        test::check("store_aligned", buf, checker.ref, false);
      }
      // there is no tagged version of aligned storing

      {
        for (std::size_t j = 0; j <= tSize; ++j) {
          std::array<T, tSize> buf{};
          checker.vec.store_part(buf.data(), j);
          test::check([&] { return fmt::format("store_part({}, {})", j, checker.ref); }, buf,
                      std::array{((tIdxs < j) ? checker.ref[tIdxs] : T{})...}, false);
          // tagged
          std::array<T, tSize> tbuf{};
          grex::store(tbuf.data(), checker.vec, grex::part_tag<tSize>(j));
          test::check("store_part tagged", tbuf,
                      std::array{((tIdxs < j) ? checker.ref[tIdxs] : T{})...}, false);
        }
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

    // store scalar
    {
      std::array<T, 1> buf{};
      const T val = dist(rng);
      grex::store(buf.data(), val, grex::scalar_tag);
      test::check("store scalar", buf[0], val, false);
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
