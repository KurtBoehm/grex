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
#include <limits>
#include <random>
#include <string_view>
#include <type_traits>

#include <fmt/base.h>
#include <fmt/color.h>
#include <fmt/ranges.h>
#include <pcg_random.hpp>

#include "grex/base/defs.hpp"
#include "grex/types.hpp"

namespace grex::test {
using Rng = pcg64;

template<typename T>
inline auto make_distribution() {
  using Limits = std::numeric_limits<T>;
  if constexpr (std::floating_point<T>) {
    return [](Rng& rng) {
      const int sign = std::uniform_int_distribution<int>{0, 1}(rng) * 2 - 1;
      const T base = std::uniform_real_distribution<T>{T(0.5), T(1)}(rng);
      const int expo =
        std::uniform_int_distribution<int>{Limits::min_exponent, Limits::max_exponent}(rng);
      return T(sign) * std::ldexp(base, expo);
    };
  } else {
    return std::uniform_int_distribution<T>{Limits::min(), Limits::max()};
  }
}

struct Empty {};

template<typename T1, typename T2, typename TLabel>
inline void check_msg(const TLabel& label, bool same, T1 a, T2 b, bool verbose = true) {
  if (same) {
    if (verbose) {
      if constexpr (std::invocable<TLabel>) {
        fmt::print(fmt::fg(fmt::terminal_color::green), "{}: {} == {}\n", label(), a, b);
      } else {
        fmt::print(fmt::fg(fmt::terminal_color::green), "{}: {} == {}\n", label, a, b);
      }
    }
  } else {
    if constexpr (std::invocable<TLabel>) {
      fmt::print(fmt::fg(fmt::terminal_color::red), "{}: {} != {}\n", label(), a, b);
    } else {
      fmt::print(fmt::fg(fmt::terminal_color::red), "{}: {} != {}\n", label, a, b);
    }
    std::exit(EXIT_FAILURE);
  }
}

template<typename T>
struct EquivVal {
  bool result{};
  T err{};

  operator bool() const {
    return result;
  }
};
template<std::floating_point T>
inline EquivVal<T> are_equivalent(T val, T ref, T f = T{}) {
  if (std::isnan(val) && std::isnan(ref)) {
    return {.result = true, .err = 0};
  }
  if (val == ref) {
    return {.result = true, .err = 0};
  }
  if (f > 0) {
    const T denom = (ref != 0 && std::isfinite(ref)) ? ref : T{1};
    const T err = std::abs((val - ref) / denom);
    return {.result = err <= f * std::numeric_limits<T>::epsilon(), .err = err};
  }
  return {};
}
template<std::integral T>
inline bool are_equivalent(T val, T ref) {
  return val == ref;
}

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

  void print(fmt::text_style ts = {}) const {
    fmt::print(ts, "[{}, {}]", vec, ref);
  }

  void check(const auto& label, bool verbose = true) const {
    const bool same = static_apply<tSize>(
      [&]<std::size_t... tIdxs>() { return (... && are_equivalent(vec[tIdxs], ref[tIdxs])); });
    if (same) {
      if (verbose) {
        if constexpr (has_parent) {
          parent.print(fmt::fg(fmt::terminal_color::green));
          fmt::print(" → ");
        }
        fmt::print(fmt::fg(fmt::terminal_color::green), "{}: {} == {}\n", label, vec, ref);
      }
    } else {
      if constexpr (has_parent) {
        parent.print(fmt::fg(fmt::terminal_color::red));
        fmt::print(" → ");
      }
      fmt::print(fmt::fg(fmt::terminal_color::red), "{}: {} != {}\n", label, vec, ref);
      std::exit(EXIT_FAILURE);
    }
  }
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
  MaskChecker(grex::Mask<T, tSize> v, std::array<bool, tSize> a) : mask{v}, ref{a} {}

  void check(const auto& label, bool verbose = true) const {
    const bool same = static_apply<tSize>(
      [&]<std::size_t... tIdxs>() { return (... && (mask[tIdxs] == ref[tIdxs])); });
    check_msg(label, same, mask, ref, verbose);
  }
};

template<typename T>
inline void check(const auto& label, T a, T b, bool verbose = true) {
  check_msg(label, a == b, a, b, verbose);
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
    fmt::print(fmt::fg(fmt::terminal_color::blue), "{}×{}\n", type_name<T>(), tSize);
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
