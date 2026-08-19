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
#include <algorithm>
#include <functional>
#include <limits>
#endif

namespace test = grex::test;
inline constexpr std::size_t repetitions = 65536;

#if !GREX_BACKEND_SCALAR
template<grex::Vectorizable T, std::size_t tSize>
void run_simd(test::Rng& rng, grex::TypeTag<T> /*tag*/, grex::IndexTag<tSize> /*tag*/) {
  using VC = test::VectorChecker<T, tSize>;
  using MC = test::MaskChecker<T, tSize>;

  auto dist = test::make_distribution<T>();
  auto dval = [&] { return dist(rng); };
  std::uniform_int_distribution<int> bdist{0, 1};
  auto bval = [&](std::size_t /*dummy*/) { return bool(bdist(rng)); };

  for (std::size_t i = 0; i < repetitions; ++i) {
    grex::static_apply<tSize>([&]<std::size_t... tIdxs>() {
      {
        auto hsum_dist = [&] {
          if constexpr (grex::Float16<T>) {
            // Binary16 is generated in binary32 and rounded, as the standard distributions do not
            // support it (see test::make_distribution); to avoid nasty cancellation issues, we only
            // consider values between 0.5 and 1, like the other floating-point types below.
            return [](test::Rng& r) {
              return grex::f32_to_f16(std::uniform_real_distribution<grex::f32>(0.5F, 1)(r));
            };
          } else if constexpr (grex::FloatVectorizable<T>) {
            // To avoid nasty cancellation issues, we only consider values between 0.5 and 1.
            return std::uniform_real_distribution<T>(T(0.5), T(1));
          } else {
            return dist;
          }
        }();
        auto hsum_val = [&](std::size_t /*dummy*/) { return hsum_dist(rng); };
        VC checker{hsum_val(tIdxs)...};
        auto cmp = [&](auto val, auto ref) {
          if constexpr (grex::FloatVectorizable<T>) {
            // A tolerance factor is required due to the changed order of additions.
            const test::Widened<T> ftol = tSize;
            const auto [same, err] = test::are_equivalent(val, ref, {.bound = T(ftol)});
            auto label = [&] {
              return fmt::format("horizontal_add({}) → {}/{}", checker.vec, err,
                                 ftol * test::widen(std::numeric_limits<T>::epsilon()));
            };
            test::check_msg(label, same, val, ref, false);
          } else {
            test::check([&] { return fmt::format("horizontal_add({})", checker.vec); }, val, ref,
                        {.verbose = false});
          }
        };

        const auto ref_full = std::reduce(checker.ref.begin(), checker.ref.end(), T{}, std::plus{});
        cmp(grex::horizontal_add(checker.vec), ref_full);
        cmp(grex::horizontal_add(checker.vec, grex::full_tag<tSize>), ref_full);
        // part
        for (std::size_t j = 0; j <= tSize; ++j) {
          cmp(grex::horizontal_add(checker.vec, grex::part_tag<tSize>(j)),
              std::reduce(checker.ref.begin(), checker.ref.begin() + j, T{}, std::plus{}));
        }
        // masked
        MC mchecker{bval(tIdxs)...};
        cmp(grex::horizontal_add(checker.vec, grex::typed_masked_tag(mchecker.mask)),
            T((... + ((mchecker.ref[tIdxs]) ? checker.ref[tIdxs] : T{}))));
      }
      {
        const VC checker = VC::random(dval);
        const auto ref = std::ranges::min(checker.ref);
        const auto label = [&] { return fmt::format("horizontal_min({})", checker.vec); };
        test::check(label, grex::horizontal_min(checker.vec), ref,
                    {.verbose = false, .cmp_zero_sign = false});
        test::check(label, grex::horizontal_min(checker.vec, grex::full_tag<tSize>), ref,
                    {.verbose = false, .cmp_zero_sign = false});
      }
      {
        const VC checker = VC::random(dval);
        const auto ref = std::ranges::max(checker.ref);
        const auto label = [&] { return fmt::format("horizontal_max({})", checker.vec); };
        test::check(label, grex::horizontal_max(checker.vec), ref,
                    {.verbose = false, .cmp_zero_sign = false});
        test::check(label, grex::horizontal_max(checker.vec, grex::full_tag<tSize>), ref,
                    {.verbose = false, .cmp_zero_sign = false});
      }
    });
    grex::static_apply<tSize>([&]<std::size_t... tIdxs>() {
      {
        const MC checker{bval(tIdxs)...};
        const bool mref = (... && checker.mask[tIdxs]);
        const auto label = [&] { return fmt::format("horizontal_and({})", checker.mask); };
        auto cmp = [&](bool val, bool ref) { test::check(label, val, ref, {.verbose = false}); };
        cmp(grex::horizontal_and(checker.mask), mref);
        cmp(grex::horizontal_and(checker.mask, grex::full_tag<tSize>), mref);
        // part
        for (std::size_t j = 0; j <= tSize; ++j) {
          cmp(grex::horizontal_and(checker.mask, grex::part_tag<tSize>(j)),
              std::reduce(checker.ref.begin(), checker.ref.begin() + j, true, std::logical_and{}));
        }
        // masked
        MC mchecker{bval(tIdxs)...};
        cmp(grex::horizontal_and(checker.mask, grex::typed_masked_tag(mchecker.mask)),
            (... && (checker.ref[tIdxs] || !mchecker.ref[tIdxs])));
      }
    });
  }
}
#endif
template<grex::Vectorizable T>
void run_scalar(test::Rng& rng, grex::TypeTag<T> /*tag*/) {
  auto dist = test::make_distribution<T>();
  std::uniform_int_distribution<int> bdist{0, 1};

  for (std::size_t i = 0; i < repetitions; ++i) {
    {
      auto hsum_dist = [&] {
        if constexpr (grex::Float16<T>) {
          // Binary16 is generated in binary32 and rounded, as the standard distributions do not
          // support it (see test::make_distribution); to avoid nasty cancellation issues, we only
          // consider values between 0.5 and 1, like the other floating-point types below.
          return [](test::Rng& r) {
            return grex::f32_to_f16(std::uniform_real_distribution<grex::f32>(0.5F, 1)(r));
          };
        } else if constexpr (grex::FloatVectorizable<T>) {
          // To avoid nasty cancellation issues, we only consider values between 0.5 and 1.
          return std::uniform_real_distribution<T>(T(0.5), T(1));
        } else {
          return dist;
        }
      }();

      const T value = hsum_dist(rng);
      test::check([&] { return fmt::format("horizontal_add({})", value); },
                  grex::horizontal_add(value, grex::scalar_tag), value, {.verbose = false});
    }
    {
      const T value = dist(rng);
      test::check([&] { return fmt::format("horizontal_min({})", value); },
                  grex::horizontal_min(value, grex::scalar_tag), value, {.verbose = false});
      test::check([&] { return fmt::format("horizontal_max({})", value); },
                  grex::horizontal_max(value, grex::scalar_tag), value, {.verbose = false});
    }
    {
      const bool value = bool(bdist(rng));
      test::check([&] { return fmt::format("horizontal_and({})", value); },
                  grex::horizontal_and(value, grex::scalar_tag), value);
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
