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
#include <limits>
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

template<typename T>
inline auto make_distribution() {
  using Limits = std::numeric_limits<T>;
  if constexpr (Float16<T>) {
    // Binary16 is generated in binary32 and rounded, as the standard distributions do not support
    // it.
    return [](Rng& rng) {
      using Trait = NumericTrait<T>;
      const int sign = std::uniform_int_distribution<int>{0, 1}(rng) * 2 - 1;
      const auto base = std::uniform_real_distribution<f32>{0.5F, 1}(rng);
      const int expo =
        std::uniform_int_distribution<int>{Trait::min_exponent, Trait::max_exponent}(rng);
      return f32_to_f16(f32(sign) * std::ldexp(base, expo));
    };
  } else if constexpr (FloatVectorizable<T>) {
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

/**
 * Promotes a binary16 value to binary32 (exact), passing every other type through unchanged.
 *
 * Intended to wrap arguments to `std::` `<cmath>` functions used as an independent reference
 * implementation: `std::numeric_limits` and the `<cmath>` functions are not guaranteed to support
 * `_Float16`, whereas binary16-to-binary32 promotion is exact and always available, and every
 * `<cmath>` function used in tests already has a `float` overload.
 */
template<Vectorizable T>
using Widened = std::conditional_t<Float16<T>, f32, T>;

/**
 * Promotes a binary16 value to binary32 (exact), passing every other type through unchanged.
 *
 * See `Widened` for details.
 */
template<Vectorizable T>
inline Widened<T> widen(T x) {
  if constexpr (Float16<T>) {
    return f16_to_f32(x);
  } else {
    return x;
  }
}

struct Empty {};

template<typename T>
struct EquivVal {
  bool result{};
  T err{};

  operator bool() const { // NOLINT
    return result;
  }
};
template<FullFloatVectorizable T>
inline EquivVal<T> are_equivalent(T val, T ref, T f = T{}) {
  if (std::isnan(val) && std::isnan(ref)) {
    return {.result = true, .err = 0};
  }
  if (val == ref) {
    return {.result = true, .err = 0};
  }
  if (f > T{}) {
    const T denom = (ref != 0 && std::isfinite(ref)) ? ref : T{1};
    const T err = std::abs((val - ref) / denom);
    return {.result = err <= f * std::numeric_limits<T>::epsilon(), .err = err};
  }
  return {};
}
/**
 * Binary16 values are compared in binary32 (which is exact), but the tolerance `f` is still
 * expressed in binary16 epsilons (`NumericTrait<f16>::epsilon()`, about 9.8e-4), not binary32 ones
 * (about 1.2e-7): delegating to the binary32 overload with the same `f` would demand roughly four
 * orders of magnitude more precision than a binary16 result can actually provide.
 */
inline EquivVal<f32> are_equivalent(f16 val, f16 ref, f32 f = 0) {
  if (f16_bits(val) == f16_bits(ref)) {
    return {.result = true, .err = 0};
  }
  const f32 fval = f16_to_f32(val);
  const f32 fref = f16_to_f32(ref);
  if (std::isnan(fval) && std::isnan(fref)) {
    return {.result = true, .err = 0};
  }
  if (f > 0) {
    const f32 denom = (fref != 0 && std::isfinite(fref)) ? fref : f32{1};
    const f32 err = std::abs((fval - fref) / denom);
    return {.result = err <= f * widen(NumericTrait<f16>::epsilon()), .err = err};
  }
  return {};
}

template<typename T>
requires(!FloatVectorizable<T>)
inline bool are_equivalent(T val, T ref) {
  return val == ref;
}

template<typename T>
inline decltype(auto) resolve_label(const T& label) {
  if constexpr (std::invocable<T>) {
    return label();
  } else {
    return label;
  }
}

/**
 * Reports a failed comparison and terminates.
 *
 * Kept out of line and cold: `label` is typically a lambda that formats whole vectors, and the
 * mismatch branch is the only one that cannot be folded away when `verbose` is a literal `false`.
 * Inlining it would instantiate and optimize that formatting code at every comparison site.
 */
template<typename T1, typename T2, typename TLabel>
[[gnu::cold, gnu::noinline]] inline void fail_msg(const TLabel& label, const T1& a, const T2& b) {
  fmt::print(fmt::fg(fmt::terminal_color::red), "{}: {} != {}\n", resolve_label(label), a, b);
  std::exit(EXIT_FAILURE);
}

template<typename T1, typename T2, typename TLabel>
inline void check_msg(const TLabel& label, bool same, T1 a, T2 b, bool verbose = true) {
  if (same) [[likely]] {
    if (verbose) {
      fmt::print(fmt::fg(fmt::terminal_color::green), "{}: {} == {}\n", resolve_label(label), a, b);
    }
  } else {
    fail_msg(label, a, b);
  }
}

template<typename T, std::size_t tSize = 1>
struct IsCompleteTrait : public std::false_type {};
template<typename T>
struct IsCompleteTrait<T, sizeof(T) / sizeof(T)> : public std::true_type {}; // NOLINT
template<typename T>
concept CompleteType = IsCompleteTrait<T>::value;

template<typename T>
requires(requires(T a) {
  { a == a } -> std::same_as<bool>;
  requires !CompleteType<std::tuple_size<T>>;
})
inline void check(const auto& label, T a, T b, bool verbose = true) {
  check_msg(label, are_equivalent(a, b), a, b, verbose);
}
template<typename T1, typename T2>
requires(requires {
  std::tuple_size<T1>::value; // NOLINT
  std::tuple_size<T2>::value; // NOLINT
  requires std::tuple_size_v<T1> == std::tuple_size_v<T2>;
})
inline void check(const auto& label, T1 a, T2 b, bool verbose = true) {
  constexpr std::size_t size = std::tuple_size_v<T1>;
  bool same = true;
  for (std::size_t i = 0; i < size; ++i) {
    if (!are_equivalent(a[i], b[i])) {
      same = false;
      break;
    }
  }
  check_msg(label, same, a, b, verbose);
}
template<typename T1, typename T2>
requires(requires {
  requires CompleteType<std::tuple_size<T1>>;
  requires CompleteType<std::tuple_size<T2>>;
  requires std::tuple_size_v<T1> == std::tuple_size_v<T2>;
})
inline void check(const auto& label, T1 a, T2 b, std::size_t size, bool verbose = true) {
  bool same = true;
  for (std::size_t i = 0; i < size; ++i) {
    if (!are_equivalent(a[i], b[i])) {
      same = false;
      break;
    }
  }
  check_msg(label, same, a, b, verbose);
}

/**
 * Returns an array whose elements are drawn, in order, from the nullary generator `gen`.
 *
 * The array is filled in a loop rather than by expanding a pack of `tSize` calls to `gen`, as
 * expanding the pack inlines the generator once per element into the caller, which dominates
 * compile time once `tSize` grows beyond a few dozen elements.
 */
template<typename T, std::size_t tSize>
inline std::array<T, tSize> random_array(auto&& gen) {
  std::array<T, tSize> values{};
  for (T& value : values) {
    value = gen();
  }
  return values;
}

#if !GREX_BACKEND_SCALAR
template<Vectorizable T, std::size_t tSize>
struct VectorChecker {
  Vector<T, tSize> vec{};
  std::array<T, tSize> ref{};

  /**
   * Creates a checker whose lanes are drawn from the nullary generator `gen`.
   *
   * See `random_array` for why the lanes are not filled by expanding a pack.
   */
  static VectorChecker random(auto&& gen) {
    const std::array<T, tSize> values = random_array<T, tSize>(gen);
    return {Vector<T, tSize>::load(values.data()), values};
  }

  VectorChecker() = default;

  explicit VectorChecker(T value) : vec{value} {
    std::ranges::fill(ref, value);
  }
  template<typename... Ts>
  requires(sizeof...(Ts) == tSize && (... && std::same_as<Ts, T>))
  explicit VectorChecker(Ts... values) : vec{values...}, ref{values...} {}
  VectorChecker(Vector<T, tSize> v, std::array<T, tSize> a) : vec{v}, ref{a} {}

  void check(const auto& label, bool verbose = true) const {
    test::check(label, vec.as_array(), ref, verbose);
  }
  void check(const auto& label, std::size_t size, bool verbose = true) const {
    test::check(label, vec.as_array(), ref, size, verbose);
  }
};
template<Vectorizable T, std::size_t tSize>
auto format_as(const VectorChecker<T, tSize>& checker) {
  return std::tie(checker.vec, checker.ref);
}

template<Vectorizable T, std::size_t tSize>
struct MaskChecker {
  Mask<T, tSize> mask{};
  std::array<bool, tSize> ref{};

  MaskChecker() = default;

  explicit MaskChecker(bool value) : mask{value} {
    std::ranges::fill(ref, value);
  }
  template<typename... Ts>
  requires(sizeof...(Ts) == tSize)
  explicit MaskChecker(Ts... values) : mask{values...}, ref{values...} {}
  MaskChecker(Mask<T, tSize> v, std::array<bool, tSize> a) : mask{v}, ref{a} {}

  void check(const auto& label, bool verbose = true) const {
    test::check(label, mask.as_array(), ref, verbose);
  }
};
template<Vectorizable T, std::size_t tSize>
auto format_as(const MaskChecker<T, tSize>& checker) {
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
template<Vectorizable T, std::size_t tMaxShift = std::bit_width(max_native_size<T>) + 1>
inline void for_each_size(auto op) {
  static_apply<1, tMaxShift>(
    [&]<std::size_t... tIdxs>() { (..., op(type_tag<T>, index_tag<1U << tIdxs>)); });
}

inline void run_types_sizes(auto f) {
  auto inner = [&]<typename T, std::size_t tSize>(TypeTag<T> t, IndexTag<tSize> s) {
    fmt::print(fmt::fg(fmt::terminal_color::blue), "{}×{}\n", type_name<T>(), tSize);
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
