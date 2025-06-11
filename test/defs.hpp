// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef TEST_DEFS_HPP
#define TEST_DEFS_HPP

#include <algorithm>
#include <array>
#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdlib>
#include <string_view>
#include <type_traits>

#include <fmt/base.h>
#include <fmt/color.h>
#include <fmt/ranges.h>

#include "grex/base/defs.hpp"
#include "grex/types.hpp"

namespace grex::test {
struct Empty {};

template<Vectorizable T, std::size_t tSize, typename TParent = void>
struct VectorChecker {
  static constexpr bool has_parent = !std::is_void_v<TParent>;
  using ParentStorage = std::conditional_t<has_parent, TParent, Empty>;

  grex::Vector<T, tSize> vec{};
  std::array<T, tSize> ref{};
  [[no_unique_address]] ParentStorage parent{};

  VectorChecker() = default;

  explicit VectorChecker(T value)
  requires(!has_parent)
      : vec{value} {
    std::ranges::fill(ref, value);
  }
  template<typename... Ts>
  requires(sizeof...(Ts) == tSize && (... && std::convertible_to<Ts, T>))
  explicit VectorChecker(Ts... values)
  requires(!has_parent)
      : vec{values...}, ref{values...} {}
  VectorChecker(grex::Vector<T, tSize> v, std::array<T, tSize> a)
  requires(!has_parent)
      : vec{v}, ref{a} {}
  VectorChecker(grex::Vector<T, tSize> v, std::array<T, tSize> a, ParentStorage p)
  requires(has_parent)
      : vec{v}, ref{a}, parent{p} {}

  static VectorChecker indices() {
    return VectorChecker{
      grex::Vector<T, tSize>::indices(),
      static_apply<tSize>([]<std::size_t... tIdxs>() { return std::array{T(tIdxs)...}; }),
    };
  }

  template<Vectorizable TDst>
  VectorChecker<TDst, tSize, VectorChecker> convert(TypeTag<TDst> tag = {}) const {
    return VectorChecker<TDst, tSize, VectorChecker>{
      vec.convert(tag),
      static_apply<tSize>(
        [&]<std::size_t... tIdxs>() { return std::array{TDst(std::get<tIdxs>(ref))...}; }),
      *this,
    };
  }

  void print(fmt::text_style ts) const {
    fmt::print(ts, "[{}, {}]", vec, ref);
  }

  void check(bool verbose = true) const {
    const bool same = static_apply<tSize>(
      [&]<std::size_t... tIdxs>() { return (... && (vec[tIdxs] == ref[tIdxs])); });
    if (same) {
      if (verbose) {
        if constexpr (has_parent) {
          parent.print(fmt::fg(fmt::terminal_color::green));
          fmt::print(" → ");
        }
        fmt::print(fmt::fg(fmt::terminal_color::green), "{} == {}\n", vec, ref);
      }
    } else {
      if constexpr (has_parent) {
        parent.print(fmt::fg(fmt::terminal_color::red));
        fmt::print(" → ");
      }
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

  void check(bool verbose = true) const {
    const bool same = static_apply<tSize>(
      [&]<std::size_t... tIdxs>() { return (... && (mask[tIdxs] == ref[tIdxs])); });
    if (same) {
      if (verbose) {
        fmt::print(fmt::fg(fmt::terminal_color::green), "{} == {}\n", mask, ref);
      }
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
constexpr std::string_view type_name() {
  return TypeNameTrait<T>::name;
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
void for_each_type(auto op) {
  op(grex::type_tag<grex::f64>);
  op(grex::type_tag<grex::f32>);
  for_each_integral(op);
};

inline void run_types_sizes(auto f) {
  auto inner = [&]<typename T, std::size_t tSize>(TypeTag<T> t, IndexTag<tSize> s) {
    fmt::print(fmt::fg(fmt::terminal_color::blue), "{}x{}\n", type_name<T>(), tSize);
    f(t, s);
  };
  auto outer = [&]<typename T>(TypeTag<T> t) {
    static_apply<1, std::bit_width(native_sizes<T>.back()) + 1>(
      [&]<std::size_t... tIdxs>() { (..., inner(t, index_tag<1U << tIdxs>)); });
  };
  for_each_type(outer);
}
} // namespace grex::test

#endif // TEST_DEFS_HPP
