// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <concepts>
#include <cstddef>
#include <functional>

#include <fmt/base.h>

#include "grex/grex.hpp"

#include "defs.hpp"

namespace test = grex::test;

template<grex::Vectorizable T, std::size_t tSize>
void run(grex::TypeTag<T> /*tag*/, grex::IndexTag<tSize> /*tag*/) {
  // vector
  auto v2v = [](auto op) {
    grex::static_apply<tSize>([&]<std::size_t... tIdxs>() {
      test::VectorChecker<T, tSize> a{T(T(tSize) - 2 * T(tIdxs))...};
      auto checker = test::v2v_cw(op, a);
      checker.check();
    });
  };
  auto vv2v = [](auto op) {
    grex::static_apply<tSize>([&]<std::size_t... tIdxs>() {
      test::VectorChecker<T, tSize> a{T(T(tSize) - 2 * T(tIdxs))...};
      test::VectorChecker<T, tSize> b{T(tIdxs % 5)...};
      auto checker = test::vv2v_cw(op, a, b);
      checker.check();
    });
  };
  v2v(std::negate{});
  vv2v(std::plus{});
  vv2v(std::minus{});
  vv2v(std::multiplies{});
  if constexpr (std::floating_point<T>) {
    vv2v(std::divides{});
  }
  if constexpr (std::integral<T>) {
    fmt::print("bit\n");
    v2v(std::bit_not{});
    vv2v(std::bit_and{});
    vv2v(std::bit_or{});
    vv2v(std::bit_xor{});
  }
  if constexpr (std::floating_point<T> || std::signed_integral<T>) {
    fmt::print("abs\n");
    v2v([]<typename TV>(TV a) {
      if constexpr (grex::AnyVector<TV>) {
        return grex::abs(a);
      } else {
        return std::abs(a);
      }
    });
  }
  fmt::print("min\n");
  vv2v([]<typename TV>(TV a, TV b) {
    if constexpr (grex::AnyVector<TV>) {
      return grex::min(a, b);
    } else {
      return std::min(a, b);
    }
  });
  fmt::print("max\n");
  vv2v([]<typename TV>(TV a, TV b) {
    if constexpr (grex::AnyVector<TV>) {
      return grex::max(a, b);
    } else {
      return std::max(a, b);
    }
  });
}

int main() {
  test::run_types_sizes([](auto... args) { run(args...); });
}
