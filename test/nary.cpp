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

namespace test = grex::test;
inline constexpr std::size_t repetitions = 4096;
inline constexpr std::size_t max_args = 16;

template<typename T1, typename T2, typename TLabel>
inline void check_with_message(const TLabel& label, bool same, T1 a, T2 b, bool verbose = true) {
  if (same) {
    if (verbose) {
      fmt::print(fmt::fg(fmt::terminal_color::green), "{}\n", test::resolve_label(label), a, b);
    }
  } else {
    fmt::print(fmt::fg(fmt::terminal_color::red), "{}\n", test::resolve_label(label), a, b);
    std::exit(EXIT_FAILURE);
  }
}

template<typename T>
void run_ops(auto op) {
  auto make_dist = [&]() {
    if constexpr (std::is_floating_point_v<T> || grex::FloatVectorizable<T>) {
      // common range for both scalar and SIMD float tests
      return std::uniform_real_distribution<T>(T(0.5), T(1));
    } else {
      return test::make_distribution<T>();
    }
  };

  auto dist = make_dist();

  // grex::add: v0 + v1 + ... + vn-1
  op(
    "add", dist, [](const auto&... v) { return (... + v); },
    [](const auto&... v) { return grex::add(v...); });

  // grex::subtract: v0 - v1 - ... - vn-1
  op(
    "subtract", dist, [](const auto& first, const auto&... rest) { return (first - ... - rest); },
    [](const auto&... v) { return grex::subtract(v...); });
}

#if !GREX_BACKEND_SCALAR
template<grex::Vectorizable T, std::size_t tSize>
void run_simd(test::Rng& rng, grex::TypeTag<T> /*tag*/, grex::IndexTag<tSize> /*tag*/) {
  run_ops<T>([&](std::string_view opname, auto dist, auto make_ref, auto make_val) {
    using VC = test::VectorChecker<T, tSize>;
    auto dval = [&](std::size_t /*dummy*/) { return dist(rng); };

    auto per_arity = [&](grex::AnyIndexTag auto num) {
      fmt::print(fmt::fg(fmt::terminal_color::magenta), "{} {}-ary\n", opname, num.value);
      for (std::size_t i = 0; i < repetitions; ++i) {
        grex::static_apply<tSize>([&]<std::size_t... tI>() {
          grex::static_apply<num.value>([&]<std::size_t... tJ>() {
            auto make_vc = [&](std::size_t /*dummy*/) { return VC{dval(tI)...}; };
            std::array<VC, num.value> arr{make_vc(tJ)...};

            const auto ref = make_ref(arr[tJ].vec...);
            const auto val = make_val(arr[tJ].vec...);

            if constexpr (grex::FloatVectorizable<T>) {
              const auto ref_arr = ref.as_array();
              const auto val_arr = val.as_array();
              for (std::size_t k = 0; k < tSize; ++k) {
                const auto a = ref_arr[k];
                const auto b = val_arr[k];
                check_with_message(
                  [&] { return fmt::format("{}({}, {}): {} != {}", opname, ref, val, a, b); },
                  test::are_equivalent(a, b, T(num.value)), a, b, false);
              }
            } else {
              test::check(opname, ref, val, false);
            }
          });
        });
      }
    };

    grex::static_apply<max_args - 1>(
      [&]<std::size_t... tI>() { (..., per_arity(grex::index_tag<tI + 1>)); });
  });
}
#endif

template<grex::Vectorizable T>
void run_scalar(test::Rng& rng, grex::TypeTag<T> /*tag*/) {
  run_ops<T>([&](std::string_view opname, auto dist, auto make_ref, auto make_val) {
    auto dval = [&](std::size_t /*dummy*/) { return dist(rng); };

    auto per_arity = [&](grex::AnyIndexTag auto num) {
      fmt::print(fmt::fg(fmt::terminal_color::magenta), "scalar {} {}-ary\n", opname, num.value);
      for (std::size_t i = 0; i < repetitions; ++i) {
        grex::static_apply<num.value>([&]<std::size_t... tJ>() {
          std::array<T, num.value> arr{dval(tJ)...};

          const auto ref = make_ref(arr[tJ]...);
          const auto val = make_val(arr[tJ]...);

          if constexpr (grex::FloatVectorizable<T>) {
            check_with_message(
              [&] { return fmt::format("{}({}, {}): {} != {}", opname, ref, val, ref, val); },
              test::are_equivalent(ref, val, T(num.value)), ref, val, false);
          } else {
            test::check(opname, ref, val, false);
          }
        });
      }
    };

    grex::static_apply<max_args - 1>(
      [&]<std::size_t... tI>() { (..., per_arity(grex::index_tag<tI + 1>)); });
  });
}

int main() {
  pcg_extras::seed_seq_from<std::random_device> seed_source{};
  test::Rng rng{seed_source};
#if !GREX_BACKEND_SCALAR
  test::run_types_sizes([&](auto vtag, auto stag) { run_simd(rng, vtag, stag); });
#endif
  test::run_types([&](auto tag) { run_scalar(rng, tag); });
}
