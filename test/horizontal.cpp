// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <concepts>
#include <cstddef>
#include <functional>
#include <limits>
#include <random>

#include <fmt/base.h>
#include <pcg_extras.hpp>

#include "grex/grex.hpp"

#include "defs.hpp"

namespace test = grex::test;
inline constexpr std::size_t repetitions = 65536;

template<grex::Vectorizable T, std::size_t tSize>
void run(test::Rng& rng, grex::TypeTag<T> /*tag*/, grex::IndexTag<tSize> /*tag*/) {
  using VC = test::VectorChecker<T, tSize>;
  using MC = test::MaskChecker<T, tSize>;

  auto dist = test::make_distribution<T>();
  auto dval = [&](std::size_t /*dummy*/) { return dist(rng); };

  for (std::size_t i = 0; i < repetitions; ++i) {
    grex::static_apply<tSize>([&]<std::size_t... tIdxs>() {
      if constexpr (std::floating_point<T>) {
        // to avoid nasty cancellation issues, we only consider values between 0.5 and 1
        auto fdist = std::uniform_real_distribution<T>(T(0.5), T(1));
        auto fval = [&](std::size_t /*dummy*/) { return fdist(rng); };

        VC checker{fval(tIdxs)...};
        const T vsum = grex::horizontal_add(checker.vec);
        const T rsum = std::reduce(checker.ref.begin(), checker.ref.end(), T{}, std::plus{});

        // A tolerance factor is required due to the changed order of additions
        const T ftol = tSize;
        const auto [same, err] = test::are_equivalent(vsum, rsum, ftol);
        auto label = [&] {
          return fmt::format("horizontal_add({}) → {}/{}", checker.vec, err,
                             ftol * std::numeric_limits<T>::epsilon());
        };
        test::check_msg(label, same, vsum, rsum, false);
      } else {
        VC checker{dval(tIdxs)...};
        const T vsum = grex::horizontal_add(checker.vec);
        const T rsum = std::reduce(checker.ref.begin(), checker.ref.end(), T{}, std::plus{});
        test::check([&] { return fmt::format("horizontal_add({})", checker.vec); }, vsum, rsum,
                    false);
      }
      {
        VC checker{dval(tIdxs)...};
        T v = checker.ref[0];
        for (std::size_t i = 1; i < tSize; ++i) {
          v = std::min(v, checker.ref[i]);
        }
        test::check([&] { return fmt::format("horizontal_min({})", checker.vec); },
                    grex::horizontal_min(checker.vec), v, false);
      }
      {
        VC checker{dval(tIdxs)...};
        T v = checker.ref[0];
        for (std::size_t i = 1; i < tSize; ++i) {
          v = std::max(v, checker.ref[i]);
        }
        test::check([&] { return fmt::format("horizontal_max({})", checker.vec); },
                    grex::horizontal_max(checker.vec), v, false);
      }
    });
    grex::static_apply<tSize>([&]<std::size_t... tIdxs>() {
      {
        std::uniform_int_distribution<int> bdist{0, 1};
        auto bval = [&](std::size_t /*dummy*/) { return bool(bdist(rng)); };
        MC checker{bval(tIdxs)...};
        bool b = true;
        for (std::size_t i = 0; i < tSize; ++i) {
          b = b && checker.mask[i];
        }
        test::check([&] { return fmt::format("horizontal_and({})", checker.mask); },
                    grex::horizontal_and(checker.mask), b, false);
      }
    });
  }
}

int main() {
  pcg_extras::seed_seq_from<std::random_device> seed_source{};
  test::Rng rng{seed_source};
  test::run_types_sizes([&](auto vtag, auto stag) { run(rng, vtag, stag); });
}
