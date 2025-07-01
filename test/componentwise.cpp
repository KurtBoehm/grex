// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <array>
#include <concepts>
#include <cstddef>
#include <functional>
#include <limits>
#include <random>

#include <fmt/base.h>
#include <pcg_extras.hpp>

#include "grex/grex.hpp"

#include "defs.hpp"

namespace test = grex::test;
inline constexpr std::size_t repetitions = 4096;

template<grex::Vectorizable T, std::size_t tSize>
void run(test::Rng& rng, grex::TypeTag<T> /*tag*/, grex::IndexTag<tSize> /*tag*/) {
  using Vec = grex::Vector<T, tSize>;
  using Mask = grex::Mask<T, tSize>;
  using VC = test::VectorChecker<T, tSize>;
  using MC = test::MaskChecker<T, tSize>;

  auto dist = test::make_distribution<T>();
  auto dval = [&](std::size_t /*dummy*/) { return dist(rng); };
  std::uniform_int_distribution<int> bdist{0, 1};
  auto bval = [&](std::size_t /*dummy*/) { return bool(bdist(rng)); };

  // vector-only operations
  grex::static_apply<tSize>([&]<std::size_t... tIdxs>() {
    for (std::size_t i = 0; i < repetitions; ++i) {
      {
        auto v2v = [&](auto op) {
          VC a{dval(tIdxs)...};
          VC{op(a.vec), std::array{T(op(a.ref[tIdxs]))...}}.check(false);
        };
        auto v2vx = [&](auto op) {
          VC a{dval(tIdxs)...};
          VC checker{
            op(a.vec, grex::FullTag<tSize>{}),
            std::array{T(op(a.ref[tIdxs], grex::ScalarTag{}))...},
          };
          checker.check(false);
        };

        auto vv2v = [&](auto op) {
          VC a{dval(tIdxs)...};
          VC b{dval(tIdxs)...};
          VC{op(a.vec, b.vec), std::array{T(op(a.ref[tIdxs], b.ref[tIdxs]))...}}.check(false);
        };

        v2v(std::negate{});
        vv2v(std::plus{});
        vv2v(std::minus{});
        vv2v(std::multiplies{});
        if constexpr (std::floating_point<T>) {
          vv2v(std::divides{});
        }
        if constexpr (std::integral<T>) {
          // fmt::print("bit\n");
          v2v(std::bit_not{});
          vv2v(std::bit_and{});
          vv2v(std::bit_or{});
          vv2v(std::bit_xor{});
        }
        if constexpr (std::floating_point<T> || std::signed_integral<T>) {
          // fmt::print("abs\n");
          v2v([]<typename TV>(TV a) {
            if constexpr (grex::AnyVector<TV>) {
              return grex::abs(a);
            } else {
              return std::abs(a);
            }
          });
          v2vx([]<typename TV>(TV a, grex::AnyTag auto tag) { return grex::abs(a, tag); });
        }
        if constexpr (std::floating_point<T>) {
          // fmt::print("sqrt\n");
          v2v([]<typename TV>(TV a) {
            if constexpr (grex::AnyVector<TV>) {
              return grex::sqrt(a);
            } else {
              return std::sqrt(a);
            }
          });
          v2v([]<typename TV>(TV a) { return grex::sqrt(a); });
        }
        // fmt::print("min\n");
        vv2v([]<typename TV>(TV a, TV b) {
          if constexpr (grex::AnyVector<TV>) {
            return grex::min(a, b);
          } else {
            return std::min(a, b);
          }
        });
        // fmt::print("max\n");
        vv2v([]<typename TV>(TV a, TV b) {
          if constexpr (grex::AnyVector<TV>) {
            return grex::max(a, b);
          } else {
            return std::max(a, b);
          }
        });
        if constexpr (std::floating_point<T>) {
          auto vvv2v = [&](auto grex_op, auto fused_op, auto fb_op) {
            VC a{dval(tIdxs)...};
            VC b{dval(tIdxs)...};
            VC c{dval(tIdxs)...};

            auto op = [&] {
              if constexpr (grex::has_fma) {
                return fused_op;
              } else {
                return fb_op;
              }
            }();
            VC checker{
              grex_op(a.vec, b.vec, c.vec),
              std::array{T(op(a.ref[tIdxs], b.ref[tIdxs], c.ref[tIdxs]))...},
            };
            checker.check(false);
          };

          // fmt::print("fmadd\n");
          vvv2v([](auto a, auto b, auto c) { return grex::fmadd(a, b, c); },
                [](auto a, auto b, auto c) { return std::fma(a, b, c); },
                [](auto a, auto b, auto c) { return a * b + c; });
          // fmt::print("fmsub\n");
          vvv2v([](auto a, auto b, auto c) { return grex::fmsub(a, b, c); },
                [](auto a, auto b, auto c) { return std::fma(a, b, -c); },
                [](auto a, auto b, auto c) { return a * b - c; });
          // fmt::print("fnmadd\n");
          vvv2v([](auto a, auto b, auto c) { return grex::fnmadd(a, b, c); },
                [](auto a, auto b, auto c) { return std::fma(-a, b, c); },
                [](auto a, auto b, auto c) { return c - a * b; });
          // fmt::print("fnmsub\n");
          vvv2v([](auto a, auto b, auto c) { return grex::fnmsub(a, b, c); },
                [](auto a, auto b, auto c) { return -std::fma(a, b, c); },
                [](auto a, auto b, auto c) { return -(a * b + c); });
        }
      }

      // masked component-wise operations
      {
        // fmt::print("masked (vector, vector) → vector\n");
        auto mvv2v = [&](auto mop, auto op) {
          MC m{bval(tIdxs)...};
          VC a{dval(tIdxs)...};
          VC b{dval(tIdxs)...};
          VC checker{
            mop(m.mask, a.vec, b.vec),
            std::array{T(m.ref[tIdxs] ? op(a.ref[tIdxs], b.ref[tIdxs]) : a.ref[tIdxs])...},
          };
          checker.check(false);
        };
        mvv2v([](Mask m, Vec a, Vec b) { return grex::mask_add(m, a, b); }, std::plus{});
        mvv2v([](Mask m, Vec a, Vec b) { return grex::mask_subtract(m, a, b); }, std::minus{});
        mvv2v([](Mask m, Vec a, Vec b) { return grex::mask_multiply(m, a, b); }, std::multiplies{});
        if constexpr (std::floating_point<T>) {
          mvv2v([](Mask m, Vec a, Vec b) { return grex::mask_divide(m, a, b); }, std::divides{});
        }
        {
          // fmt::print("blend_zero\n");
          MC m{bval(tIdxs)...};
          VC a{dval(tIdxs)...};
          VC checker{
            grex::blend_zero(m.mask, a.vec),
            std::array{T(m.ref[tIdxs] ? a.ref[tIdxs] : 0)...},
          };
          checker.check(false);
        }
        {
          // fmt::print("blend\n");
          MC m{bval(tIdxs)...};
          VC a{dval(tIdxs)...};
          VC b{dval(tIdxs)...};
          VC checker{
            grex::blend(m.mask, a.vec, b.vec),
            std::array<T, tSize>{T(m.ref[tIdxs] ? b.ref[tIdxs] : a.ref[tIdxs])...},
          };
          checker.check(false);
        }
      }

      // vector-vector comparison operations
      {
        // fmt::print("(vector, vector) → mask\n");
        auto vv2m = [&](auto op) {
          VC a{dval(tIdxs)...};
          VC b{dval(tIdxs)...};
          MC checker{op(a.vec, b.vec), std::array{op(a.ref[tIdxs], b.ref[tIdxs])...}};
          checker.check(false);
        };
        vv2m(std::equal_to{});
        vv2m(std::not_equal_to{});
        vv2m(std::less{});
        vv2m(std::greater{});
        vv2m(std::greater_equal{});
        vv2m(std::less_equal{});
      }

      // vector-to-mask operations
      if constexpr (std::floating_point<T>) {
        VC a{((tIdxs % 3 == 0)
                ? std::numeric_limits<T>::quiet_NaN()
                : ((tIdxs % 3 == 2) ? std::numeric_limits<T>::infinity() : T(tIdxs)))...};
        MC checker{grex::is_finite(a.vec), std::array{std::isfinite(a.ref[tIdxs])...}};
        checker.check(false);
      }

      // mask-only operations
      {
        // fmt::print("mask-only\n");
        auto m2m = [&](auto op) {
          MC a{bval(tIdxs)...};
          MC checker{op(a.mask), std::array{op(a.ref[tIdxs])...}};
          checker.check(false);
        };
        auto mm2m = [&](auto op) {
          MC a{bval(tIdxs)...};
          MC b{bval(tIdxs)...};
          MC checker{op(a.mask, b.mask), std::array{op(a.ref[tIdxs], b.ref[tIdxs])...}};
          checker.check(false);
        };
        m2m(std::logical_not{});
        mm2m(std::logical_and{});
        mm2m(std::logical_or{});
        mm2m(std::equal_to{});
        mm2m(std::not_equal_to{});
      }
    }
  });
}

int main() {
  pcg_extras::seed_seq_from<std::random_device> seed_source{};
  test::Rng rng{seed_source};
  test::run_types_sizes([&](auto vtag, auto stag) { run(rng, vtag, stag); });
}
