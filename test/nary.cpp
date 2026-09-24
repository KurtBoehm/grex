// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <array>
#include <cstddef>
#include <cstdlib>
#include <random>
#include <string_view>

#include <fmt/color.h>
#include <fmt/format.h>
#include <pcg_extras.hpp>

#include "grex/grex.hpp"

#include "defs.hpp"

namespace {
namespace test = grex::test;
using Value = grex::GREX_TEST_TYPE;
inline constexpr std::size_t repetitions = 4096;
inline constexpr std::size_t max_args = 16;

/** Reports a failed comparison and terminates, kept out of line and cold as in `test::fail_msg`. */
template<typename Label>
[[gnu::cold, gnu::noinline]] inline void fail_with_message(const Label& label) {
  fmt::print(fmt::fg(fmt::terminal_color::red), "{}\n", test::resolve_label(label));
  std::exit(EXIT_FAILURE);
}

template<typename T1, typename T2, typename Label>
inline void check_with_message(const Label& label, bool same, T1 a, T2 b, bool verbose = true) {
  if (same) [[likely]] {
    if (verbose) {
      fmt::print(fmt::fg(fmt::terminal_color::green), "{}\n", test::resolve_label(label), a, b);
    }
  } else {
    fail_with_message(label);
  }
}

template<typename T>
void run_ops(auto op) {
  const auto make_dist = [&] {
    using W = test::Widened<T>;
    if constexpr (grex::FloatVectorizable<T>) {
      // Common range for both scalar and SIMD float tests.
      return std::uniform_real_distribution<W>{W(0.5), W(1)};
    } else {
      return test::make_distribution<W>();
    }
  };

  const auto dist = [d = make_dist()](auto& r) mutable { return T(d(r)); };

  // grex::add: v[0] + v[1] + ... + v[n - 1]
  op(
    "add", dist, [](const auto&... v) { return (... + v); },
    [](const auto&... v) { return grex::add(v...); });

  // grex::subtract: v[0] - v[1] - ... - v[n - 1]
  op(
    "subtract", dist, [](const auto& first, const auto&... rest) { return (first - ... - rest); },
    [](const auto&... v) { return grex::subtract(v...); });
}

#if !GREX_BACKEND_SCALAR
template<grex::Vectorizable T, std::size_t N>
void run_simd(test::Rng& rng, grex::TypeTag<T> /*tag*/, grex::IndexTag<N> /*tag*/) {
  run_ops<T>([&](std::string_view opname, auto dist, auto make_ref, auto make_val) {
    using VC = test::VectorChecker<T, N>;
    auto dval = [&] { return dist(rng); };

    auto per_arity = [&](grex::AnyIndexTag auto num) {
      fmt::print(fmt::fg(fmt::terminal_color::magenta), "{} {}-ary\n", opname, num.value);
      for (std::size_t i = 0; i < repetitions; ++i) {
        grex::static_apply<num.value>([&]<std::size_t... J> {
          const auto make_vc = [&](std::size_t /*dummy*/) { return VC::random(dval); };
          std::array<VC, num.value> arr{make_vc(J)...};

          const auto ref = make_ref(arr[J].vec...);
          const auto val = make_val(arr[J].vec...);

          if constexpr (grex::FloatVectorizable<T>) {
            const auto ref_arr = ref.as_array();
            const auto val_arr = val.as_array();
            for (std::size_t k = 0; k < N; ++k) {
              const auto a = ref_arr[k];
              const auto b = val_arr[k];
              check_with_message(
                [&] { return fmt::format("{}({}, {}): {} != {}", opname, val, ref, a, b); },
                test::are_equivalent(a, b, {.bound = num.value}), a, b, false);
            }
          } else {
            test::check(opname, val, ref, {.verbose = false});
          }
        });
      }
    };

    grex::static_apply<max_args - 1>(
      [&]<std::size_t... I> { (..., per_arity(grex::index_tag<I + 1>)); });
  });
}
#endif

template<grex::Vectorizable T>
void run_scalar(test::Rng& rng, grex::TypeTag<T> /*tag*/) {
  run_ops<T>([&](std::string_view opname, auto dist, auto make_ref, auto make_val) {
    auto dval = [&] { return dist(rng); };

    auto per_arity = [&](grex::AnyIndexTag auto num) {
      fmt::print(fmt::fg(fmt::terminal_color::magenta), "scalar {} {}-ary\n", opname, num.value);
      for (std::size_t i = 0; i < repetitions; ++i) {
        grex::static_apply<num.value>([&]<std::size_t... J> {
          std::array<T, num.value> arr = test::random_array<T, num.value>(dval);

          const auto val = make_val(arr[J]...);
          const auto ref = T(make_ref(arr[J]...));

          if constexpr (grex::FloatVectorizable<T>) {
            check_with_message(
              [&] { return fmt::format("{}({}, {}): {} != {}", opname, val, ref, val, ref); },
              test::are_equivalent(val, ref, {.bound = test::Widened<T>(num.value)}), val, ref,
              false);
          } else {
            test::check(opname, val, ref, {.verbose = false});
          }
        });
      }
    };

    grex::static_apply<max_args - 1>(
      [&]<std::size_t... I> { (..., per_arity(grex::index_tag<I + 1>)); });
  });
}
} // namespace

int main() {
  pcg_extras::seed_seq_from<std::random_device> seed_source{};
  test::Rng rng{seed_source};
#if !GREX_BACKEND_SCALAR
  test::for_each_size<Value>([&](auto vtag, auto stag) {
    fmt::print(fmt::fg(fmt::terminal_color::blue), "{}×{}\n", test::type_name<Value>(), stag.value);
    run_simd(rng, vtag, stag);
  });
#endif
  fmt::print(fmt::fg(fmt::terminal_color::blue), "{}\n", test::type_name<Value>());
  run_scalar(rng, grex::type_tag<Value>);
}
