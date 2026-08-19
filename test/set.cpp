// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <cstddef>
#include <random>

#include <fmt/base.h>
#include <fmt/format.h>
#include <pcg_extras.hpp>

#include "grex/grex.hpp"

#include "defs.hpp"

#if !GREX_BACKEND_SCALAR
#include <array>
#endif

namespace test = grex::test;
inline constexpr std::size_t repetitions = 4096;

#if !GREX_BACKEND_SCALAR
template<grex::Vectorizable T, std::size_t tSize>
void run_simd(test::Rng& rng, grex::TypeTag<T> /*tag*/, grex::IndexTag<tSize> /*tag*/) {
  using VC = test::VectorChecker<T, tSize>;
  using Vec = grex::Vector<T, tSize>;
  using MC = test::MaskChecker<T, tSize>;
  using Mask = grex::Mask<T, tSize>;

  auto dist = test::make_distribution<T>();
  auto dval = [&] { return dist(rng); };
  std::uniform_int_distribution<int> bdist{0, 1};
  auto bval = [&](std::size_t /*dummy*/) { return bool(bdist(rng)); };

  grex::static_apply<tSize>([&]<std::size_t... tIdxs>() {
    for (std::size_t i = 0; i < repetitions; ++i) {
      // zeros
      test::check("scalar zeros", grex::zeros<T>(grex::scalar_tag), T{}, {.verbose = false});
      VC{}.check("vector zeros", {.verbose = false});
      test::check("vector zeros tagged", grex::zeros<T>(grex::full_tag<tSize>), Vec{},
                  {.verbose = false});
      // broadcast
      {
        const T value = dist(rng);
        test::check("scalar broadcast", grex::broadcast(value, grex::scalar_tag), value,
                    {.verbose = false});
        VC{value}.check("vector broadcast", {.verbose = false});
        test::check("vector broadcast tagged", grex::broadcast(value, grex::full_tag<tSize>),
                    Vec{value}, {.verbose = false});
      }
      // zero-based indices
      test::check("scalar indices", grex::indices<T>(grex::scalar_tag), T{}, {.verbose = false});
      VC{Vec::indices(), std::array{T(tIdxs)...}}.check("vector indices", {.verbose = false});
      test::check("vector indices tagged", grex::indices<T>(grex::typed_full_tag<T, tSize>),
                  Vec::indices(), {.verbose = false});
      // value-based indices
      {
        const T base = dist(rng);
        test::check("scalar value indices", grex::indices<T>(base, grex::scalar_tag), base,
                    {.verbose = false});
        VC{Vec::indices(base), std::array{T(base + T(tIdxs))...}}.check("vector value indices",
                                                                        {.verbose = false});
        test::check("vector value indices tagged",
                    grex::indices<T>(base, grex::typed_full_tag<T, tSize>), Vec::indices(base),
                    {.verbose = false});
      }
      // set
      VC::random(dval).check("vector set", {.verbose = false});
      // insert
      {
        const VC base = VC::random(dval);
        for (std::size_t j = 0; j < tSize; ++j) {
          const auto val = dval();
          VC v{base.vec.insert(j, val), std::array{((tIdxs == j) ? val : base.ref[tIdxs])...}};
          v.check("vector insert", {.verbose = false});
        }
      }
      {
        const VC base = VC::random(dval);
        auto f = [&](grex::AnyIndexTag auto j) {
          const auto val = dval();
          VC v{base.vec.insert(j, val), std::array{((tIdxs == j) ? val : base.ref[tIdxs])...}};
          v.check(fmt::format("{}.insert(index_tag<{}>, {})", base.vec, j.value, val),
                  {.verbose = false});
        };
        (..., f(grex::index_tag<tIdxs>));
      }
      // cutoff
      {
        const VC base = VC::random(dval);
        for (std::size_t j = 0; j <= tSize; ++j) {
          VC v{base.vec.cutoff(j), std::array{((tIdxs < j) ? base.ref[tIdxs] : T(0))...}};
          v.check("vector cutoff", {.verbose = false});
        }
      }

      // mask
      MC{}.check("mask zeros", {.verbose = false});
      MC{Mask::ones(), std::array{(tIdxs < tSize)...}}.check("mask ones", {.verbose = false});
      MC{false}.check("mask broadcast false", {.verbose = false});
      MC{true}.check("mask broadcast true", {.verbose = false});
      MC{bval(tIdxs)...}.check("mask set", {.verbose = false});
      {
        const MC base{bval(tIdxs)...};
        for (std::size_t j = 0; j < tSize; ++j) {
          const bool val = bval(j);
          MC v{base.mask.insert(j, val), std::array{((tIdxs == j) ? val : base.ref[tIdxs])...}};
          v.check("mask insert", {.verbose = false});
        }
      }
      {
        for (std::size_t j = 0; j <= tSize; ++j) {
          MC v{Mask::cutoff_mask(j), std::array{(tIdxs < j)...}};
          v.check("mask cutoff_mask", {.verbose = false});
        }
      }
      {
        for (std::size_t j = 0; j < tSize; ++j) {
          MC v{Mask::single_mask(j), std::array{(tIdxs == j)...}};
          v.check("mask single_mask", {.verbose = false});
        }
      }
    }
  });
}
#endif
template<grex::Vectorizable T>
void run_scalar(test::Rng& rng, grex::TypeTag<T> /*tag*/) {
  auto dist = test::make_distribution<T>();
  std::uniform_int_distribution<int> bdist{0, 1};

  for (std::size_t i = 0; i < repetitions; ++i) {
    // zeros
    test::check("scalar zeros", grex::zeros<T>(grex::scalar_tag), T{}, {.verbose = false});
    // broadcast
    {
      const T value = dist(rng);
      test::check("scalar broadcast", grex::broadcast(value, grex::scalar_tag), value,
                  {.verbose = false});
    }
    // zero-based indices
    test::check("scalar indices", grex::indices<T>(grex::scalar_tag), T{}, {.verbose = false});
    // value-based indices
    {
      const T base = dist(rng);
      test::check("scalar value indices", grex::indices<T>(base, grex::scalar_tag), base,
                  {.verbose = false});
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
