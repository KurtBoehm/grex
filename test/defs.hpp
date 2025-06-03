// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef TEST_DEFS_HPP
#define TEST_DEFS_HPP

#include <algorithm>
#include <array>
#include <concepts>
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
  requires(sizeof...(Ts) == tSize && (... && std::convertible_to<Ts, T>))
  explicit VectorChecker(Ts... values) : vec{values...}, ref{values...} {}
  VectorChecker(grex::Vector<T, tSize> v, std::array<T, tSize> a) : vec{v}, ref{a} {}

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
};

template<Vectorizable T, std::size_t tSize>
VectorChecker<T, tSize> v2v_cw(auto op, VectorChecker<T, tSize> a) {
  return VectorChecker<T, tSize>{
    op(a.vec),
    static_apply<tSize>([&]<std::size_t... tIdxs>() { return std::array{T(op(a.ref[tIdxs]))...}; }),
  };
}
template<Vectorizable T, std::size_t tSize>
VectorChecker<T, tSize> vv2v_cw(auto op, VectorChecker<T, tSize> a, VectorChecker<T, tSize> b) {
  return VectorChecker<T, tSize>{
    op(a.vec, b.vec),
    static_apply<tSize>(
      [&]<std::size_t... tIdxs>() { return std::array{T(op(a.ref[tIdxs], b.ref[tIdxs]))...}; }),
  };
}
template<Vectorizable T, std::size_t tSize>
VectorChecker<T, tSize> vvv2v_cw(auto op, VectorChecker<T, tSize> a, VectorChecker<T, tSize> b,
                                 VectorChecker<T, tSize> c) {
  return VectorChecker<T, tSize>{
    op(a.vec, b.vec, c.vec),
    static_apply<tSize>([&]<std::size_t... tIdxs>() {
      return std::array{T(op(a.ref[tIdxs], b.ref[tIdxs], c.ref[tIdxs]))...};
    }),
  };
}

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
  MaskChecker(grex::Mask<T, tSize> v, std::array<bool, tSize> a) : mask{v}, ref{a} {}

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
};

template<typename T>
inline void check(T a, T b) {
  if (a == b) {
    fmt::print(fmt::fg(fmt::terminal_color::green), "{} == {}\n", a, b);
  } else {
    fmt::print(fmt::fg(fmt::terminal_color::red), "{} != {}\n", a, b);
    std::exit(EXIT_FAILURE);
  }
}

template<Vectorizable T, std::size_t tSize>
MaskChecker<T, tSize> m2m_cw(auto op, MaskChecker<T, tSize> a) {
  return MaskChecker<T, tSize>{
    op(a.mask),
    static_apply<tSize>([&]<std::size_t... tIdxs>() { return std::array{op(a.ref[tIdxs])...}; }),
  };
}
template<Vectorizable T, std::size_t tSize>
MaskChecker<T, tSize> mm2m_cw(auto op, MaskChecker<T, tSize> a, MaskChecker<T, tSize> b) {
  return MaskChecker<T, tSize>{
    op(a.mask, b.mask),
    static_apply<tSize>(
      [&]<std::size_t... tIdxs>() { return std::array{op(a.ref[tIdxs], b.ref[tIdxs])...}; }),
  };
}
template<Vectorizable T, std::size_t tSize>
VectorChecker<T, tSize> masked_vv2v_cw(auto mop, auto op, MaskChecker<T, tSize> m,
                                       VectorChecker<T, tSize> a, VectorChecker<T, tSize> b) {
  return VectorChecker<T, tSize>{
    mop(m.mask, a.vec, b.vec),
    static_apply<tSize>([&]<std::size_t... tIdxs>() {
      return std::array{T(m.ref[tIdxs] ? op(a.ref[tIdxs], b.ref[tIdxs]) : a.ref[tIdxs])...};
    }),
  };
}
template<Vectorizable T, std::size_t tSize>
MaskChecker<T, tSize> vv2m_cw(auto op, VectorChecker<T, tSize> a, VectorChecker<T, tSize> b) {
  return MaskChecker<T, tSize>{
    op(a.vec, b.vec),
    static_apply<tSize>(
      [&]<std::size_t... tIdxs>() { return std::array{op(a.ref[tIdxs], b.ref[tIdxs])...}; }),
  };
}

template<typename T>
struct TypeNameTrait;
#define GREX_TYPE_TRAIT(TYPE) \
  template<> \
  struct TypeNameTrait<TYPE> { \
    static constexpr auto name = #TYPE; \
  };
GREX_TYPE_TRAIT(f32);
GREX_TYPE_TRAIT(f64);
GREX_TYPE_TRAIT(i8);
GREX_TYPE_TRAIT(i16);
GREX_TYPE_TRAIT(i32);
GREX_TYPE_TRAIT(i64);
GREX_TYPE_TRAIT(u8);
GREX_TYPE_TRAIT(u16);
GREX_TYPE_TRAIT(u32);
GREX_TYPE_TRAIT(u64);
#undef GREX_TYPE_TRAIT
template<typename T>
constexpr auto type_name() {
  return TypeNameTrait<T>::name;
}

inline void run_types_sizes(auto f) {
  auto inner = [&]<typename T, std::size_t tSize>(TypeTag<T> t, IndexTag<tSize> s) {
    fmt::print(fmt::fg(fmt::terminal_color::blue), "{}x{}\n", type_name<T>(), tSize);
    f(t, s);
  };
  auto outer = [&]<typename T>(TypeTag<T> t) {
    static_apply<1, 7>([&]<std::size_t... tIdxs>() { (..., inner(t, index_tag<1U << tIdxs>)); });
  };
  outer(type_tag<f32>);
  outer(type_tag<f64>);
  outer(type_tag<i8>);
  outer(type_tag<i16>);
  outer(type_tag<i32>);
  outer(type_tag<i64>);
  outer(type_tag<u8>);
  outer(type_tag<u16>);
  outer(type_tag<u32>);
  outer(type_tag<u64>);
}
} // namespace grex::test

#endif // TEST_DEFS_HPP
