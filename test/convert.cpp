// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <cstddef>
#include <limits>
#include <random>
#include <type_traits>

#include <fmt/base.h>
#include <fmt/color.h>
#include <fmt/format.h>
#include <pcg_extras.hpp>
#include <pcg_random.hpp>

#include "grex/grex.hpp"

#include "defs.hpp"

#if !GREX_BACKEND_SCALAR
#include <algorithm>
#include <array>
#include <bit>
#endif

namespace {
namespace test = grex::test;
using Src = grex::GREX_TEST_TYPE;
using Rng = pcg64;
inline constexpr std::size_t repetitions = 4096;
template<typename T>
using Distribution =
  std::conditional_t<grex::FloatVectorizable<T>, std::uniform_real_distribution<T>,
                     std::uniform_int_distribution<T>>;

template<typename Src, typename Dst>
inline auto make_distribution() {
  if constexpr (grex::FloatVectorizable<Src>) {
    return [gen = test::make_distribution<Src>()](Rng& rng) mutable {
      Src f = gen(rng);
      if constexpr (grex::IntVectorizable<Dst>) {
        // The C++ standard only specifies behaviour if the floating-point value is representable by
        // the destination integer. I adopt the same approach to keep the amount of work reasonable.
        using Limits = std::numeric_limits<Dst>;
        while (test::widen(f) < test::Widened<Src>(Limits::min()) ||
               test::widen(f) > test::Widened<Src>(Limits::max())) {
          f = gen(rng);
        }
      }
      return f;
    };
  } else {
    return test::make_distribution<Src>();
  }
}

#if !GREX_BACKEND_SCALAR
void run_simd(Rng& rng) {
  const auto cvt = [&]<typename Dst>(grex::TypeTag<Dst> /*tag*/) {
    fmt::print(fmt::fg(fmt::terminal_color::blue) | fmt::emphasis::bold, "{} → {}\n",
               test::type_name<Src>(), test::type_name<Dst>());
    auto op = [&]<std::size_t N>(grex::IndexTag<N> /*tag*/) {
      fmt::print(fmt::fg(fmt::terminal_color::magenta), "{}\n", N);
      auto dist = make_distribution<Src, Dst>();
      const auto dval = [&] { return dist(rng); };
      auto bdst = std::uniform_int_distribution<int>(0, 1);
      const auto bval = [&](std::size_t /*dummy*/) { return static_cast<bool>(bdst(rng)); };

      for (std::size_t i = 0; i < repetitions; ++i) {
        grex::static_apply<N>([&]<std::size_t... I> {
          {
            const auto src = test::VectorChecker<Src, N>::random(dval);
            const test::VectorChecker<Dst, N> dst{
              src.vec.convert(grex::type_tag<Dst>),
              std::array{Dst(src.ref[I])...},
            };
            dst.check([&] { return fmt::format("vector/scalar {}", src); }, {.verbose = false});

            const grex::Vector<Dst, N> dstvec = grex::convert<Dst>(src.vec);
            test::check([&] { return fmt::format("vector/tagged vector {}", src); }, dst.vec,
                        dstvec, {.verbose = false});

            const test::VectorChecker<Dst, N> dstsca{
              src.vec.convert(grex::type_tag<Dst>),
              std::array{grex::convert<Dst>(src.ref[I])...},
            };
            dstsca.check([&] { return fmt::format("vector/tagged scalar {}", src); },
                         {.verbose = false});
          }
          {
            const auto src = test::VectorChecker<Src, N>::random(dval);
            const auto arr = src.vec.as_array();
            test::check([&] { return fmt::format("vector to array {}", src); }, arr, src.ref,
                        {.verbose = false});
          }

          {
            const test::MaskChecker<Src, N> src{bval(I)...};
            const test::MaskChecker<Dst, N> dst{src.mask.convert(grex::type_tag<Dst>), src.ref};
            dst.check([&] { return fmt::format("mask/copy {}", src); }, {.verbose = false});

            const grex::Mask<Dst, N> dstmsk = grex::convert<Dst>(src.mask);
            test::check([&] { return fmt::format("mask/tagged mask {}", src); }, dst.mask, dstmsk,
                        {.verbose = false});

            const test::MaskChecker<Dst, N> dstsca{
              src.mask.convert(grex::type_tag<Dst>),
              std::array{grex::convert<Dst>(src.ref[I])...},
            };
            dstsca.check([&] { return fmt::format("mask/tagged scalar {}", src); },
                         {.verbose = false});
          }
          {
            const test::MaskChecker<Src, N> src{bval(I)...};
            const auto arr = src.mask.as_array();
            test::check([&] { return fmt::format("mask to array {}", src); }, arr, src.ref,
                        {.verbose = false});
          }
        });
      }
    };

    constexpr std::size_t size = std::min(grex::max_native_size<Src>, grex::max_native_size<Dst>);
    grex::static_apply<1, std::bit_width(size) + 2>(
      [&]<std::size_t... Ns> { (..., op(grex::index_tag<1ULL << Ns>)); });
  };
  test::for_each_type(cvt);
}
#endif
void run_scalar(Rng& rng) {
  const auto cvt = [&]<typename Dst>(grex::TypeTag<Dst> /*tag*/) {
    fmt::print(fmt::fg(fmt::terminal_color::blue) | fmt::emphasis::bold, "{} → {}\n",
               test::type_name<Src>(), test::type_name<Dst>());
    auto dist = make_distribution<Src, Dst>();
    for (std::size_t i = 0; i < repetitions; ++i) {
      const Src src = dist(rng);
      const Dst dst_ref = Dst(src);
      const Dst dst_cvt = grex::convert<Dst>(src);
      test::check([&] { return fmt::format("{}", src); }, dst_cvt, dst_ref, {.verbose = false});
    }
  };
  test::for_each_type(cvt);
}
} // namespace

int main() {
  pcg_extras::seed_seq_from<std::random_device> seed_source{};
  Rng rng{seed_source};

#if !GREX_BACKEND_SCALAR
  run_simd(rng);
#endif
  run_scalar(rng);
}
