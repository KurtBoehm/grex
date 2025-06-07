// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

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
        return sign * std::ldexp(base, expo);
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

void for_each_integral(auto op) {
  op(grex::type_tag<grex::i64>);
  op(grex::type_tag<grex::i32>);
  op(grex::type_tag<grex::i16>);
  op(grex::type_tag<grex::i8>);
  op(grex::type_tag<grex::u64>);
  op(grex::type_tag<grex::u32>);
  op(grex::type_tag<grex::u16>);
  op(grex::type_tag<grex::u8>);
};

template<grex::Vectorizable TSrc>
void convert_from_base(Rng& rng, grex::TypeTag<TSrc> /*tag*/ = {}) {
  fmt::print(fmt::fg(fmt::terminal_color::magenta) | fmt::text_style(fmt::emphasis::bold), "{}\n",
             test::type_name<TSrc>());
  auto to_integral = [&]<std::integral TDst>(grex::TypeTag<TDst> /*tag*/) {
    if (!std::same_as<TSrc, TDst>) {
      fmt::print(fmt::fg(fmt::terminal_color::blue), "{}\n", test::type_name<TDst>());
      constexpr std::size_t size =
        std::min(grex::native_sizes<TSrc>.front(), grex::native_sizes<TDst>.front());
      auto dist = make_distribution<TSrc, TDst>();
      auto dval = [&](std::size_t /*dummy*/) { return dist(rng); };

      for (std::size_t i = 0; i < repetitions; ++i) {
        grex::static_apply<size>([&]<std::size_t... tIdxs> {
          test::VectorChecker<TSrc, size>{dval(tIdxs)...}
            .convert(grex::type_tag<TDst>)
            .check(false);
        });
      }
    }
  };
  for_each_integral(to_integral);
}

int main() {
  pcg_extras::seed_seq_from<std::random_device> seed_source{};
  Rng rng{seed_source};

  for_each_integral([&](auto tag) { convert_from_base(rng, tag); });
  convert_from_base<grex::f32>(rng);
  convert_from_base<grex::f64>(rng);
}
