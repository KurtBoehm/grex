// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef TEST_DEFS_HPP
#define TEST_DEFS_HPP

#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdlib>
#include <random>
#include <string_view>
#include <tuple>
#include <type_traits>

#include <fmt/base.h>
#include <fmt/color.h>
#include <fmt/ranges.h>
#include <fmt/std.h>
#include <pcg_random.hpp>

#include "grex/grex.hpp"

#if !GREX_BACKEND_SCALAR
#include <algorithm>
#include <bit>
#endif

namespace grex::test {
using Rng = pcg64;

/**
 * Widens binary16 to binary32 and leaves any other value unchanged.
 *
 * This is useful for interfacing between standard-library functionality (math functions,
 * random-number generation, etc.), which is not guaranteed to be support binary16, and binary16.
 */
template<Vectorizable T>
using Widened = std::conditional_t<Float16<T>, f32, T>;

/** Widens binary16 to binary32 and leaves any other value unchanged. See `Widened` for details. */
template<Vectorizable T>
inline Widened<T> widen(T x) {
  if constexpr (Float16<T>) {
    return f16_to_f32(x);
  } else {
    return x;
  }
}

/**
 * Makes a distribution that is appropriate for testing `T`.
 *
 * For integers, a uniform distribution between the minimum and maximum value of `T` is returned.
 * For floating-point value, the values are sampled such that each possible exponent is drawn with
 * equal probability, which is combined with a value between `0.5` and `1` drawn with a uniform
 * distribution via `ldexp`.
 */
template<typename T>
inline auto make_distribution() {
  using W = Widened<T>;
  using Trait = NumericTrait<T>;

  if constexpr (FloatVectorizable<T>) {
    return [](Rng& rng) {
      const int mode = std::uniform_int_distribution<int>{0, 2}(rng);
      switch (mode) {
        case 0: return T(-0.0);
        case 1: return T(0.0);
        default: break;
      }

      const int sign = std::uniform_int_distribution<int>{0, 1}(rng) * 2 - 1;
      const W base = std::uniform_real_distribution<W>{W(0.5), W(1)}(rng);
      const int expo =
        std::uniform_int_distribution<int>{Trait::min_exponent, Trait::max_exponent}(rng);
      return T(W(sign) * std::ldexp(base, expo));
    };
  } else {
    return std::uniform_int_distribution<T>{Trait::min(), Trait::max()};
  }
}

/**
 * Whether two values are equivalent (`result`) together with the error (if a bound was specified).
 */
template<typename T>
struct EquivVal {
  bool result{};
  T err{};

  operator bool() const { // NOLINT
    return result;
  }
};

/** Parameters for `are_equivalent`. */
template<typename T>
struct EquivParams {
  /** The multiple of `epsilon` used as the error bound (applies only to floating-point values). */
  T bound = {};
  /**
   * Whether equivalence requires that zeros have the same sign (applies only to floating-point
   * values).
   */
  bool cmp_zero_sign = false;
};

/** Checks whether `val` and `ref` are equivalent using the parameters `eq`. */
template<FloatVectorizable T>
inline EquivVal<T> are_equivalent(T val, T ref, EquivParams<T> eq = {}) {
  using W = Widened<T>;
  const W xval = widen(val);
  const W xref = widen(ref);

  if (std::isnan(xval) && std::isnan(xref)) {
    return {.result = true, .err = 0};
  }
  if (xval == xref && (!eq.cmp_zero_sign || std::signbit(xval) == std::signbit(xref))) {
    return {.result = true, .err = 0};
  }
  if (eq.bound > T{}) {
    const W denom = (xref != 0 && std::isfinite(xref)) ? xref : W{1};
    const W err = std::abs((xval - xref) / denom);
    return {.result = err <= W(eq.bound) * W(NumericTrait<T>::epsilon()), .err = T(err)};
  }
  return {};
}

/** For non-floating-point values, equivalence is equality. */
template<typename T>
requires(!FloatVectorizable<T>)
inline bool are_equivalent(T val, T ref, EquivParams<T> /*eq*/ = {}) {
  return val == ref;
}

/** Returns `label()` if `label` is invocable without arguments and `label` itself otherwise. */
template<typename T>
inline decltype(auto) resolve_label(const T& label) {
  if constexpr (std::invocable<T>) {
    return label();
  } else {
    return label;
  }
}

/** Parameters for the `check` family of operations. */
struct Check {
  /** Whether the check should always print the result. */
  bool verbose = true;
  /**
   * Whether to ensure that the sign of two zeros is the same.
   *
   * Only meaningful for floating-point values.
   */
  bool cmp_zero_sign = true;
};

/**
 * Prints an error message showing that `a` and `b` are not equivalent with `label` as the label and
 * exits with a non-zero error code.
 *
 * To avoid the compiler wasting time on optimizing this function, which should not be executed
 * during normal operation, inlining is disabled and it is marked as cols.
 */
template<typename T1, typename T2, typename Label>
[[gnu::cold, gnu::noinline]] inline void fail_msg(const Label& label, const T1& a, const T2& b) {
  fmt::print(fmt::fg(fmt::terminal_color::red), "{}: {} != {}\n", resolve_label(label), a, b);
  std::exit(EXIT_FAILURE);
}

/**
 * If `same` and `verbose` are true, prints `label`, `a`, and `b` in green; if `same` is false,
 * calls `fail_msg`, which prints the same information in red and exits with a non-zero error code.
 */
template<typename T1, typename T2, typename Label>
inline void check_msg(const Label& label, bool same, T1 a, T2 b, bool verbose = true) {
  if (same) [[likely]] {
    if (verbose) {
      fmt::print(fmt::fg(fmt::terminal_color::green), "{}: {} == {}\n", resolve_label(label), a, b);
    }
  } else {
    fail_msg(label, a, b);
  }
}

template<typename T, std::size_t N = 1>
inline constexpr bool is_complete = false;
template<typename T>
inline constexpr bool is_complete<T, sizeof(T) / sizeof(T)> = true; // NOLINT
template<typename T>
concept CompleteType = is_complete<T>;

// Checks that two non-tuple-like values, i.e. scalars, are equivalent.
template<typename T>
requires(requires(T a) {
  { a == a } -> std::same_as<bool>;
  requires !CompleteType<std::tuple_size<T>>;
})
inline void check(const auto& label, T a, T b, Check check = {}) {
  check_msg(label, are_equivalent(a, b, {.cmp_zero_sign = check.cmp_zero_sign}), a, b,
            check.verbose);
}
// Checks that two tuple-like values, which includes vectors and masks, are equivalent.
template<typename T1, typename T2>
requires(requires {
  std::tuple_size<T1>::value; // NOLINT
  std::tuple_size<T2>::value; // NOLINT
  requires std::tuple_size_v<T1> == std::tuple_size_v<T2>;
})
inline void check(const auto& label, T1 a, T2 b, Check check = {}) {
  constexpr std::size_t size = std::tuple_size_v<T1>;
  bool same = true;
  for (std::size_t i = 0; i < size; ++i) {
    if (!are_equivalent(a[i], b[i], {.cmp_zero_sign = check.cmp_zero_sign})) {
      same = false;
      break;
    }
  }
  check_msg(label, same, a, b, check.verbose);
}
// Checks that two tuple-like values, which includes vectors and masks, are equivalent up to index
// `size - 1`.
template<typename T1, typename T2>
requires(requires {
  requires CompleteType<std::tuple_size<T1>>;
  requires CompleteType<std::tuple_size<T2>>;
  requires std::tuple_size_v<T1> == std::tuple_size_v<T2>;
})
inline void check(const auto& label, T1 a, T2 b, std::size_t size, Check check = {}) {
  bool same = true;
  for (std::size_t i = 0; i < size; ++i) {
    if (!are_equivalent(a[i], b[i], {.cmp_zero_sign = check.cmp_zero_sign})) {
      same = false;
      break;
    }
  }
  check_msg(label, same, a, b, check.verbose);
}

/**
 * Generates `N` values of type `T` using `gen` and places them into an array in order.
 *
 * This function uses a loop instead of pack expansion two minimize the amount of inlining and,
 * thereby, to minimize the compile-time cost.
 */
template<typename T, std::size_t N>
inline std::array<T, N> random_array(auto&& gen) {
  std::array<T, N> values{};
  for (T& value : values) {
    value = gen();
  }
  return values;
}

#if !GREX_BACKEND_SCALAR
template<Vectorizable T, std::size_t N>
struct VectorChecker {
  Vector<T, N> vec{};
  std::array<T, N> ref{};

  /**
   * Creates a checker whose lanes are drawn from the nullary generator `gen`.
   *
   * See `random_array` for why the lanes are not filled by expanding a pack.
   */
  static VectorChecker random(auto&& gen) {
    const std::array<T, N> values = random_array<T, N>(gen);
    return {Vector<T, N>::load(values.data()), values};
  }

  VectorChecker() = default;

  explicit VectorChecker(T value) : vec{value} {
    std::ranges::fill(ref, value);
  }
  template<typename... Ts>
  requires(sizeof...(Ts) == N && (... && std::same_as<Ts, T>))
  explicit VectorChecker(Ts... values) : vec{values...}, ref{values...} {}
  VectorChecker(Vector<T, N> v, std::array<T, N> a) : vec{v}, ref{a} {}

  void check(const auto& label, Check check = {}) const {
    test::check(label, vec.as_array(), ref, check);
  }
  void check(const auto& label, std::size_t size, Check check = {}) const {
    test::check(label, vec.as_array(), ref, size, check);
  }
};
template<Vectorizable T, std::size_t N>
auto format_as(const VectorChecker<T, N>& checker) {
  return std::tie(checker.vec, checker.ref);
}

template<Vectorizable T, std::size_t N>
struct MaskChecker {
  Mask<T, N> mask{};
  std::array<bool, N> ref{};

  MaskChecker() = default;

  explicit MaskChecker(bool value) : mask{value} {
    std::ranges::fill(ref, value);
  }
  template<typename... Ts>
  requires(sizeof...(Ts) == N)
  explicit MaskChecker(Ts... values) : mask{values...}, ref{values...} {}
  MaskChecker(Mask<T, N> v, std::array<bool, N> a) : mask{v}, ref{a} {}

  void check(const auto& label, Check check = {}) const {
    test::check(label, mask.as_array(), ref, check);
  }
};
template<Vectorizable T, std::size_t N>
auto format_as(const MaskChecker<T, N>& checker) {
  return std::tie(checker.mask, checker.ref);
}
#endif

template<typename T>
struct TypeNameTrait;
#define GREX_TYPE_TRAIT(TYPE) \
  template<> \
  struct TypeNameTrait<TYPE> { \
    static constexpr auto name = #TYPE; \
  }

GREX_TYPE_TRAIT(f16);
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
  op(type_tag<i64>);
  op(type_tag<i32>);
  op(type_tag<i16>);
  op(type_tag<i8>);
  op(type_tag<u64>);
  op(type_tag<u32>);
  op(type_tag<u16>);
  op(type_tag<u8>);
}
void for_each_type(auto op) {
  op(type_tag<f64>);
  op(type_tag<f32>);
  op(type_tag<f16>);
  for_each_integral(op);
}

#if !GREX_BACKEND_SCALAR
template<Vectorizable T, std::size_t MaxShift = std::bit_width(max_native_size<T>) + 1>
inline void for_each_size(auto op) {
  static_apply<1, MaxShift>([&]<std::size_t... I> { (..., op(type_tag<T>, index_tag<1U << I>)); });
}

inline void run_types_sizes(auto f) {
  auto inner = [&]<typename T, std::size_t N>(TypeTag<T> t, IndexTag<N> s) {
    fmt::print(fmt::fg(fmt::terminal_color::blue), "{}×{}\n", type_name<T>(), N);
    f(t, s);
  };
  for_each_type([&]<typename T>(TypeTag<T> /*tag*/) { for_each_size<T>(inner); });
}
#endif
inline void run_types(auto f) {
  for_each_type([&]<typename T>(TypeTag<T> tag) {
    fmt::print(fmt::fg(fmt::terminal_color::blue), "{}\n", type_name<T>());
    f(tag);
  });
}
} // namespace grex::test

#endif // TEST_DEFS_HPP
