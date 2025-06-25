// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <bit>
#include <concepts>
#include <cstddef>
#include <limits>
#include <random>
#include <type_traits>

#include <fmt/base.h>
#include <fmt/color.h>
#include <pcg_extras.hpp>
#include <pcg_random.hpp>

#include "grex/grex.hpp"

#include "defs.hpp"

namespace test = grex::test;
using Rng = pcg64;
inline constexpr std::size_t repetitions = 4096;
template<typename T>
using Distribution = std::conditional_t<std::floating_point<T>, std::uniform_real_distribution<T>,
                                        std::uniform_int_distribution<T>>;

template<typename TSrc, typename TDst>
inline auto make_distribution() {
  using Limits = std::numeric_limits<TSrc>;
  if constexpr (std::floating_point<TSrc>) {
    return [](Rng& rng) {
      auto gen = [&] {
        const int sign = std::uniform_int_distribution<int>{0, 1}(rng) * 2 - 1;
        const TSrc base = std::uniform_real_distribution<TSrc>{TSrc(0.5), TSrc(1)}(rng);
        const int expo =
          std::uniform_int_distribution<int>{Limits::min_exponent, Limits::max_exponent}(rng);
        return TSrc(sign) * std::ldexp(base, expo);
      };
      TSrc f = gen();
      if constexpr (std::integral<TDst>) {
        // The C++ standard only specifies behaviour if the floating-point value
        // is representable by the destination integer
        // I adopt the same approach to keep the amount of work proportionate
        while (f < TSrc(std::numeric_limits<TDst>::min()) ||
               f > TSrc(std::numeric_limits<TDst>::max())) {
          f = gen();
        }
      }
      return f;
    };
  } else {
    return std::uniform_int_distribution<TSrc>{Limits::min(), Limits::max()};
  }
}

template<grex::Vectorizable TSrc>
void convert_from(Rng& rng, grex::TypeTag<TSrc> /*tag*/ = {}) {
  auto cvt = [&]<typename TDst>(grex::TypeTag<TDst> /*tag*/) {
    fmt::print(fmt::fg(fmt::terminal_color::blue) | fmt::text_style(fmt::emphasis::bold),
               "{} → {}\n", test::type_name<TSrc>(), test::type_name<TDst>());
    auto op = [&]<std::size_t tSize>(grex::IndexTag<tSize> /*tag*/) {
      fmt::print(fmt::fg(fmt::terminal_color::magenta), "{}\n", tSize);
      auto dist = make_distribution<TSrc, TDst>();
      auto dval = [&](std::size_t /*dummy*/) { return dist(rng); };
      auto bdst = std::uniform_int_distribution<grex::u8>(0, 1);
      auto bval = [&](std::size_t /*dummy*/) { return bdst(rng) != 0; };

      for (std::size_t i = 0; i < repetitions; ++i) {
        grex::static_apply<tSize>([&]<std::size_t... tIdxs> {
          test::VectorChecker<TSrc, tSize>{dval(tIdxs)...}
            .convert(grex::type_tag<TDst>)
            .check(false);
          test::MaskChecker<TSrc, tSize>{bval(tIdxs)...}.convert(grex::type_tag<TDst>).check(false);
        });
      }
    };

    constexpr std::size_t size =
      std::min(grex::native_sizes<TSrc>.back(), grex::native_sizes<TDst>.back());
    grex::static_apply<1, std::bit_width(size) + 2>(
      [&]<std::size_t... tSizes> { (..., op(grex::index_tag<1ULL << tSizes>)); });
  };
  test::for_each_type(cvt);
}

int main() {
  pcg_extras::seed_seq_from<std::random_device> seed_source{};
  Rng rng{seed_source};
  test::for_each_type([&](auto tag) { convert_from(rng, tag); });
}
