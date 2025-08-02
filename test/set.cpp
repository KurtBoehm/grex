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
  using MC = test::MaskChecker<T, tSize>;
  using Mask = grex::Mask<T, tSize>;

  auto dist = test::make_distribution<T>();
  auto dval = [&](std::size_t /*dummy*/) { return dist(rng); };
  std::uniform_int_distribution<int> bdist{0, 1};
  auto bval = [&](std::size_t /*dummy*/) { return bool(bdist(rng)); };

  grex::static_apply<tSize>([&]<std::size_t... tIdxs>() {
    for (std::size_t i = 0; i < repetitions; ++i) {
      // SCALAR/VECTOR
      // zeros
      test::check("scalar zeros", grex::zeros<T>(grex::scalar_tag), T{}, false);
      VC{}.check("vector zeros", false);
      test::check("vector zeros tagged", grex::zeros<T>(grex::full_tag<tSize>), Vec{}, false);
      // broadcast
      {
        const T value = dist(rng);
        test::check("scalar broadcast", grex::broadcast(value, grex::scalar_tag), value, false);
        VC{value}.check("vector broadcast", false);
        test::check("vector broadcast tagged", grex::broadcast(value, grex::full_tag<tSize>),
                    Vec{value}, false);
      }
      // zero-based indices
      test::check("scalar indices", grex::indices<T>(grex::scalar_tag), T{}, false);
      VC{Vec::indices(), std::array{T(tIdxs)...}}.check("vector indices", false);
      test::check("vector indices tagged", grex::indices<T>(grex::typed_full_tag<T, tSize>),
                  Vec::indices(), false);
      // value-based indices
      {
        const T base = dist(rng);
        test::check("scalar value indices", grex::indices<T>(base, grex::scalar_tag), base, false);
        VC{Vec::indices(base), std::array{T(base + T(tIdxs))...}}.check("vector value indices",
                                                                        false);
        test::check("vector value indices tagged",
                    grex::indices<T>(base, grex::typed_full_tag<T, tSize>), Vec::indices(base),
                    false);
      }
      // set
      VC{dval(tIdxs)...}.check("vector set", false);
      // insert
      {
        const VC base{dval(tIdxs)...};
        for (std::size_t j = 0; j < tSize; ++j) {
          const auto val = dval(j);
          VC v{base.vec.insert(j, val), std::array{((tIdxs == j) ? val : base.ref[tIdxs])...}};
          v.check("vector insert", false);
        }
      }
      {
        const VC base{dval(tIdxs)...};
        auto f = [&](grex::AnyIndexTag auto j) {
          const auto val = dval(j);
          VC v{base.vec.insert(j, val), std::array{((tIdxs == j) ? val : base.ref[tIdxs])...}};
          v.check("vector sinsert", false);
        };
        (..., f(grex::index_tag<tIdxs>));
      }
      // cutoff
      {
        const VC base{dval(tIdxs)...};
        for (std::size_t j = 0; j <= tSize; ++j) {
          VC v{base.vec.cutoff(j), std::array{((tIdxs < j) ? base.ref[tIdxs] : T(0))...}};
          v.check("vector cutoff", false);
        }
      }

      // mask
      MC{}.check("mask zeros", false);
      MC{Mask::ones(), std::array{(tIdxs < tSize)...}}.check("mask ones", false);
      MC{false}.check("mask broadcast false", false);
      MC{true}.check("mask broadcast true", false);
      MC{bval(tIdxs)...}.check("mask set", false);
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
