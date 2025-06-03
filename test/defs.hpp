// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef TEST_DEFS_HPP
#define TEST_DEFS_HPP

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>

#include <fmt/base.h>
#include <fmt/color.h>
#include <fmt/ranges.h>

#include "grex/base/defs.hpp"
#include "grex/types.hpp"

namespace grex::test {
template<Vectorizable T, std::size_t tSize>
struct VectorChecker {
  grex::Vector<T, tSize> vec{};
  std::array<T, tSize> ref{};

  VectorChecker() = default;

  explicit VectorChecker(T value) : vec{value} {
    std::ranges::fill(ref, value);
  }
  template<typename... Ts>
  requires(sizeof...(Ts) == tSize)
  explicit VectorChecker(Ts... values) : vec{values...}, ref{values...} {}

  static VectorChecker indices() {
    return VectorChecker{
      grex::Vector<T, tSize>::indices(),
      static_apply<tSize>([]<std::size_t... tIdxs>() { return std::array{T(tIdxs)...}; }),
    };
  }

  void check() const {
    const bool same = static_apply<tSize>(
      [&]<std::size_t... tIdxs>() { return (... && (vec[tIdxs] == ref[tIdxs])); });
    if (same) {
      fmt::print(fmt::fg(fmt::terminal_color::green), "{} == {}\n", vec, ref);
    } else {
      fmt::print(fmt::fg(fmt::terminal_color::red), "{} != {}\n", vec, ref);
      std::exit(EXIT_FAILURE);
    }
  }

private:
  VectorChecker(grex::Vector<T, tSize> v, std::array<T, tSize> a) : vec{v}, ref{a} {}
};

template<Vectorizable T, std::size_t tSize>
struct MaskChecker {
  grex::Mask<T, tSize> mask{};
  std::array<bool, tSize> ref{};

  MaskChecker() = default;

  explicit MaskChecker(bool value) : mask{value} {
    std::ranges::fill(ref, value);
  }
  template<typename... Ts>
  requires(sizeof...(Ts) == tSize)
  explicit MaskChecker(Ts... values) : mask{values...}, ref{values...} {}

  static MaskChecker ones() {
    auto f = [](std::size_t /*dummy*/) { return true; };
    return MaskChecker{
      grex::Mask<T, tSize>::ones(),
      static_apply<tSize>([&]<std::size_t... tIdxs>() { return std::array{f(tIdxs)...}; }),
    };
  }

  void check() const {
    const bool same = static_apply<tSize>(
      [&]<std::size_t... tIdxs>() { return (... && (mask[tIdxs] == ref[tIdxs])); });
    if (same) {
      fmt::print(fmt::fg(fmt::terminal_color::green), "{} == {}\n", mask, ref);
    } else {
      fmt::print(fmt::fg(fmt::terminal_color::red), "{} != {}\n", mask, ref);
      std::exit(EXIT_FAILURE);
    }
  }

private:
  MaskChecker(grex::Mask<T, tSize> v, std::array<bool, tSize> a) : mask{v}, ref{a} {}
};
} // namespace grex::test

#endif // TEST_DEFS_HPP
