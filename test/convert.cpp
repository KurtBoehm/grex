// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <limits>
#include <random>
#include <type_traits>

#include <fmt/color.h>
#include <fmt/format.h>
#include <pcg_extras.hpp>
#include <pcg_random.hpp>

#include "grex/grex.hpp"

#include "defs.hpp"

namespace test = grex::test;
using Src = grex::GREX_TEST_TYPE;
using Rng = pcg64;
inline constexpr std::size_t repetitions = 4096;
template<typename T>
using Distribution =
  std::conditional_t<grex::FloatVectorizable<T>, std::uniform_real_distribution<T>,
                     std::uniform_int_distribution<T>>;

template<typename TSrc, typename TDst>
inline auto make_distribution() {
  if constexpr (grex::FloatVectorizable<TSrc>) {
    return [gen = test::make_distribution<TSrc>()](Rng& rng) mutable {
      TSrc f = gen(rng);
      if constexpr (grex::IntVectorizable<TDst>) {
        // The C++ standard only specifies behaviour if the floating-point value
        // is representable by the destination integer
        // I adopt the same approach to keep the amount of work reasonable
        using Limits = std::numeric_limits<TDst>;
        while (f < TSrc(Limits::min()) || f > TSrc(Limits::max())) {
          f = gen(rng);
        }
      }
      return f;
    };
  } else {
    return test::make_distribution<TSrc>();
  }
}

#if !GREX_BACKEND_SCALAR
void run_simd(Rng& rng) {
  auto cvt = [&]<typename TDst>(grex::TypeTag<TDst> /*tag*/) {
    fmt::print(fmt::fg(fmt::terminal_color::blue) | fmt::text_style(fmt::emphasis::bold),
               "{} → {}\n", test::type_name<Src>(), test::type_name<TDst>());
    auto op = [&]<std::size_t tSize>(grex::IndexTag<tSize> /*tag*/) {
      fmt::print(fmt::fg(fmt::terminal_color::magenta), "{}\n", tSize);
      auto dist = make_distribution<Src, TDst>();
      auto dval = [&](std::size_t /*dummy*/) { return dist(rng); };
      auto bdst = std::uniform_int_distribution<int>(0, 1);
      auto bval = [&](std::size_t /*dummy*/) { return bool(bdst(rng)); };

      for (std::size_t i = 0; i < repetitions; ++i) {
        grex::static_apply<tSize>([&]<std::size_t... tIdxs> {
          {
            test::VectorChecker<Src, tSize> src{dval(tIdxs)...};
            test::VectorChecker<TDst, tSize> dst{
              src.vec.convert(grex::type_tag<TDst>),
              std::array{TDst(src.ref[tIdxs])...},
            };
            dst.check([&] { return fmt::format("vector/scalar {}", src); }, false);

            grex::Vector<TDst, tSize> dstvec = grex::convert_unsafe<TDst>(src.vec);
            test::check([&] { return fmt::format("vector/tagged vector {}", src); }, dst.vec,
                        dstvec, false);

            test::VectorChecker<TDst, tSize> dstsca{
              src.vec.convert(grex::type_tag<TDst>),
              std::array{grex::convert_unsafe<TDst>(src.ref[tIdxs])...},
            };
            dstsca.check([&] { return fmt::format("vector/tagged scalar {}", src); }, false);
          }
          {
            test::MaskChecker<Src, tSize> src{bval(tIdxs)...};
            test::MaskChecker<TDst, tSize> dst{src.mask.convert(grex::type_tag<TDst>), src.ref};
            dst.check([&] { return fmt::format("mask/copy {}", src); }, false);

            grex::Mask<TDst, tSize> dstmsk = grex::convert<TDst>(src.mask);
            test::check([&] { return fmt::format("mask/tagged mask {}", src); }, dst.mask, dstmsk,
                        false);

            test::MaskChecker<TDst, tSize> dstsca{
              src.mask.convert(grex::type_tag<TDst>),
              std::array{grex::convert<TDst>(src.ref[tIdxs])...},
            };
            dstsca.check([&] { return fmt::format("mask/tagged scalar {}", src); }, false);
          }
        });
      }
    };

    constexpr std::size_t size = std::min(grex::max_native_size<Src>, grex::max_native_size<TDst>);
    grex::static_apply<1, std::bit_width(size) + 2>(
      [&]<std::size_t... tSizes> { (..., op(grex::index_tag<1ULL << tSizes>)); });
  };
  test::for_each_type(cvt);
}
#endif
void run_scalar(Rng& rng) {
  auto cvt = [&]<typename TDst>(grex::TypeTag<TDst> /*tag*/) {
    fmt::print(fmt::fg(fmt::terminal_color::blue) | fmt::text_style(fmt::emphasis::bold),
               "{} → {}\n", test::type_name<Src>(), test::type_name<TDst>());
    auto dist = make_distribution<Src, TDst>();
    for (std::size_t i = 0; i < repetitions; ++i) {
      const Src src = dist(rng);
      const TDst dst_ref = TDst(src);
      const TDst dst_cvt = grex::convert_unsafe<TDst>(src);
      test::check([&] { return fmt::format("{}", src); }, dst_cvt, dst_ref, false);
    }
  };
  test::for_each_type(cvt);
}

int main() {
  pcg_extras::seed_seq_from<std::random_device> seed_source{};
  Rng rng{seed_source};

#if !GREX_BACKEND_SCALAR
  run_simd(rng);
#endif
  run_scalar(rng);
}
