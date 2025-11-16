// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <array>
#include <cstddef>
#include <functional>
#include <limits>
#include <random>

#include <pcg_extras.hpp>

#include "grex/grex.hpp"

#include "defs.hpp"

namespace test = grex::test;
inline constexpr std::size_t repetitions = 4096;

#if !GREX_BACKEND_SCALAR
template<grex::Vectorizable T, std::size_t tSize>
void run_simd(test::Rng& rng, grex::TypeTag<T> /*tag*/, grex::IndexTag<tSize> /*tag*/) {
  using VC = test::VectorChecker<T, tSize>;
  using MC = test::MaskChecker<T, tSize>;

  auto dist = test::make_distribution<T>();
  auto dval = [&](std::size_t /*dummy*/) { return dist(rng); };
  std::uniform_int_distribution<int> bdist{0, 1};
  auto bval = [&](std::size_t /*dummy*/) { return bool(bdist(rng)); };

  auto nonfin = [&](std::size_t /*dummy*/) {
    if constexpr (grex::FloatVectorizable<T>) {
      if (bdist(rng)) {
        return dist(rng);
      }
      if (bdist(rng)) {
        const T inf = std::numeric_limits<T>::infinity();
        return bdist(rng) ? inf : -inf;
      }
      return bdist(rng) ? std::numeric_limits<T>::quiet_NaN()
                        : std::numeric_limits<T>::signaling_NaN();
    }
  };

  grex::static_apply<tSize>([&]<std::size_t... tIdxs>() {
    for (std::size_t i = 0; i < repetitions; ++i) {
      // vector-only operations
      {
        auto v2vx = [&](auto label, auto vop, auto sop) {
          VC a{dval(tIdxs)...};
          VC{vop(a.vec), std::array{T(sop(a.ref[tIdxs]))...}}.check(label, false);
        };
        auto v2v = [&](auto label, auto op) { v2vx(label, op, op); };

        auto vv2vx = [&](auto label, auto vop, auto sop) {
          VC a{dval(tIdxs)...};
          VC b{dval(tIdxs)...};
          VC checker{vop(a.vec, b.vec), std::array{T(sop(a.ref[tIdxs], b.ref[tIdxs]))...}};
          checker.check(label, false);
        };
        auto vv2v = [&](auto label, auto op) { vv2vx(label, op, op); };

        // arithmetic
        v2v("negate", std::negate{});
        vv2v("plus", std::plus{});
        vv2v("minus", std::minus{});
        vv2v("multiplies", std::multiplies{});
        if constexpr (grex::FloatVectorizable<T>) {
          vv2v("divides", std::divides{});
        }

        // bit operations
        if constexpr (grex::IntVectorizable<T>) {
          v2v("bit_not", std::bit_not{});
          vv2v("bit_and", std::bit_and{});
          vv2v("bit_or", std::bit_or{});
          vv2v("bit_xor", std::bit_xor{});
        }

        // abs/sqrt
        if constexpr (grex::SignedVectorizable<T>) {
          v2vx("abs", [](auto a) { return grex::abs(a); }, [](auto a) { return std::abs(a); });
          v2v("abs", [](auto a) { return grex::abs(a); });
        }
        if constexpr (grex::FloatVectorizable<T>) {
          v2vx("sqrt", [](auto a) { return grex::sqrt(a); }, [](auto a) { return std::sqrt(a); });
          v2v("sqrt", [](auto a) { return grex::sqrt(a); });
        }

        // make_finite
        if constexpr (grex::FloatVectorizable<T>) {
          VC a{nonfin(tIdxs)...};
          VC checker{grex::make_finite(a.vec),
                     std::array{(std::isfinite(a.ref[tIdxs]) ? a.ref[tIdxs] : T{})...}};
          checker.check("make_finite", false);
          VC gchecker{grex::make_finite(a.vec), std::array{grex::make_finite(a.ref[tIdxs])...}};
          gchecker.check("make_finite", false);
        }

        // min/max
        vv2vx(
          "min", [](auto a, auto b) { return grex::min(a, b); },
          [](auto a, auto b) { return std::min(a, b); });
        vv2v("min", [](auto a, auto b) { return grex::min(a, b); });
        vv2vx(
          "max", [](auto a, auto b) { return grex::max(a, b); },
          [](auto a, auto b) { return std::max(a, b); });
        vv2v("max", [](auto a, auto b) { return grex::max(a, b); });

        // fma family
        if constexpr (grex::FloatVectorizable<T>) {
          auto vvv2v = [&](auto label, auto grex_op, auto fused_op, auto fb_op) {
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
            checker.check(label, false);

            // using the scalar operation as reference
            VC gchecker{
              grex_op(a.vec, b.vec, c.vec),
              std::array{T(grex_op(a.ref[tIdxs], b.ref[tIdxs], c.ref[tIdxs]))...},
            };
            gchecker.check(label, false);
          };

          vvv2v(
            "fmadd", [](auto a, auto b, auto c) { return grex::fmadd(a, b, c); },
            [](auto a, auto b, auto c) { return std::fma(a, b, c); },
            [](auto a, auto b, auto c) { return a * b + c; });
          vvv2v(
            "fmsub", [](auto a, auto b, auto c) { return grex::fmsub(a, b, c); },
            [](auto a, auto b, auto c) { return std::fma(a, b, -c); },
            [](auto a, auto b, auto c) { return a * b - c; });
          vvv2v(
            "fnmadd", [](auto a, auto b, auto c) { return grex::fnmadd(a, b, c); },
            [](auto a, auto b, auto c) { return std::fma(-a, b, c); },
            [](auto a, auto b, auto c) { return c - a * b; });
          vvv2v(
            "fnmsub", [](auto a, auto b, auto c) { return grex::fnmsub(a, b, c); },
            [](auto a, auto b, auto c) { return -std::fma(a, b, c); },
            [](auto a, auto b, auto c) { return -(a * b + c); });
        }
      }

      // masked component-wise operations
      {
        auto mvv2v = [&](auto label, auto gop, auto sop) {
          MC m{bval(tIdxs)...};
          VC a{dval(tIdxs)...};
          VC b{dval(tIdxs)...};
          VC checker{
            gop(m.mask, a.vec, b.vec),
            std::array{T(m.ref[tIdxs] ? sop(a.ref[tIdxs], b.ref[tIdxs]) : a.ref[tIdxs])...},
          };
          checker.check(label, false);

          // using the scalar operation as reference
          VC gchecker{
            gop(m.mask, a.vec, b.vec),
            std::array{gop(m.ref[tIdxs], a.ref[tIdxs], b.ref[tIdxs])...},
          };
          gchecker.check(label, false);
        };

        // masked arithmetic
        mvv2v(
          "mask_add", [](auto m, auto a, auto b) { return grex::mask_add(m, a, b); }, std::plus{});
        mvv2v(
          "mask_subtract", [](auto m, auto a, auto b) { return grex::mask_subtract(m, a, b); },
          std::minus{});
        mvv2v(
          "mask_multiply", [](auto m, auto a, auto b) { return grex::mask_multiply(m, a, b); },
          std::multiplies{});
        if constexpr (grex::FloatVectorizable<T>) {
          mvv2v(
            "mask_divide", [](auto m, auto a, auto b) { return grex::mask_divide(m, a, b); },
            std::divides{});
        }

        // blend_zero
        {
          MC m{bval(tIdxs)...};
          VC a{dval(tIdxs)...};
          VC checker{
            grex::blend_zero(m.mask, a.vec),
            std::array{T(m.ref[tIdxs] ? a.ref[tIdxs] : 0)...},
          };
          checker.check("blend_zero", false);
          VC gchecker{
            grex::blend_zero(m.mask, a.vec),
            std::array{grex::blend_zero(m.ref[tIdxs], a.ref[tIdxs])...},
          };
          gchecker.check("blend_zero", false);
        }
        {
          MC m{bval(tIdxs)...};
          VC a{dval(tIdxs)...};
          VC b{dval(tIdxs)...};
          VC checker{
            grex::blend(m.mask, a.vec, b.vec),
            std::array<T, tSize>{T(m.ref[tIdxs] ? b.ref[tIdxs] : a.ref[tIdxs])...},
          };
          checker.check("blend", false);
          VC gchecker{
            grex::blend(m.mask, a.vec, b.vec),
            std::array<T, tSize>{grex::blend(m.ref[tIdxs], a.ref[tIdxs], b.ref[tIdxs])...},
          };
          gchecker.check("blend", false);
        }
      }

      // vector-vector comparison operations
      {
        auto vv2m = [&](auto label, auto op) {
          VC a{dval(tIdxs)...};
          VC b{dval(tIdxs)...};
          MC checker{op(a.vec, b.vec), std::array{op(a.ref[tIdxs], b.ref[tIdxs])...}};
          checker.check(label, false);
        };
        vv2m("equal_to", std::equal_to{});
        vv2m("not_equal_to", std::not_equal_to{});
        vv2m("less", std::less{});
        vv2m("greater", std::greater{});
        vv2m("greater_equal", std::greater_equal{});
        vv2m("less_equal", std::less_equal{});
      }

      // vector-to-mask operations
      if constexpr (grex::FloatVectorizable<T>) {
        VC a{nonfin(tIdxs)...};
        MC checker{grex::is_finite(a.vec), std::array{std::isfinite(a.ref[tIdxs])...}};
        checker.check("is_finite", false);
        MC gchecker{grex::is_finite(a.vec), std::array{grex::is_finite(a.ref[tIdxs])...}};
        gchecker.check("is_finite", false);
      }

      // mask-only operations
      {
        auto m2m = [&](auto label, auto op) {
          MC a{bval(tIdxs)...};
          MC checker{op(a.mask), std::array{op(a.ref[tIdxs])...}};
          checker.check(label, false);
        };
        auto mm2m = [&](auto label, auto op) {
          MC a{bval(tIdxs)...};
          MC b{bval(tIdxs)...};
          MC checker{op(a.mask, b.mask), std::array{op(a.ref[tIdxs], b.ref[tIdxs])...}};
          checker.check(label, false);
        };
        m2m("logical_not", std::logical_not{});
        mm2m("logical_and", std::logical_and{});
        mm2m("logical_andnot", [](auto a, auto b) { return grex::andnot(a, b); });
        mm2m("logical_or", std::logical_or{});
        mm2m("equal_to", std::equal_to{});
        mm2m("not_equal_to", std::not_equal_to{});
      }
    }
  });
}
#endif
template<grex::Vectorizable T>
void run_scalar(test::Rng& rng, grex::TypeTag<T> /*tag*/) {
  auto dist = test::make_distribution<T>();
  std::uniform_int_distribution<int> bdist{0, 1};

  for (std::size_t i = 0; i < repetitions; ++i) {
    // vector-only operations
    {
      auto v2v = [&](auto label, auto vop, auto sop) {
        const T a = dist(rng);
        test::check(label, vop(a), sop(a), false);
      };
      auto vv2v = [&](auto label, auto vop, auto sop) {
        const T a = dist(rng);
        const T b = dist(rng);
        test::check(label, vop(a, b), sop(a, b), false);
      };

      // abs/sqrt
      if constexpr (grex::SignedVectorizable<T>) {
        v2v("abs", [](auto a) { return grex::abs(a); }, [](auto a) { return T(std::abs(a)); });
      }
      if constexpr (grex::FloatVectorizable<T>) {
        v2v("sqrt", [](auto a) { return grex::sqrt(a); }, [](auto a) { return std::sqrt(a); });
      }

      // min/max
      vv2v(
        "min", [](auto a, auto b) { return grex::min(a, b); },
        [](auto a, auto b) { return std::min(a, b); });
      vv2v(
        "max", [](auto a, auto b) { return grex::max(a, b); },
        [](auto a, auto b) { return std::max(a, b); });

      // fma family
      if constexpr (grex::FloatVectorizable<T>) {
        auto vvv2v = [&](auto label, auto grex_op, auto fused_op, auto fb_op) {
          const T a = dist(rng);
          const T b = dist(rng);
          const T c = dist(rng);

          auto ref_op = [&] {
            if constexpr (grex::has_fma) {
              return fused_op;
            } else {
              return fb_op;
            }
          }();
          test::check(label, grex_op(a, b, c), ref_op(a, b, c), false);
        };

        vvv2v(
          "fmadd", [](auto a, auto b, auto c) { return grex::fmadd(a, b, c); },
          [](auto a, auto b, auto c) { return std::fma(a, b, c); },
          [](auto a, auto b, auto c) { return a * b + c; });
        vvv2v(
          "fmsub", [](auto a, auto b, auto c) { return grex::fmsub(a, b, c); },
          [](auto a, auto b, auto c) { return std::fma(a, b, -c); },
          [](auto a, auto b, auto c) { return a * b - c; });
        vvv2v(
          "fnmadd", [](auto a, auto b, auto c) { return grex::fnmadd(a, b, c); },
          [](auto a, auto b, auto c) { return std::fma(-a, b, c); },
          [](auto a, auto b, auto c) { return c - a * b; });
        vvv2v(
          "fnmsub", [](auto a, auto b, auto c) { return grex::fnmsub(a, b, c); },
          [](auto a, auto b, auto c) { return -std::fma(a, b, c); },
          [](auto a, auto b, auto c) { return -(a * b + c); });
      }
    }

    // masked component-wise operations
    {
      auto bvv2v = [&](auto label, auto grex_op, auto ref_op) {
        const bool m = bool(bdist(rng));
        const T a = dist(rng);
        const T b = dist(rng);
        test::check(label, grex_op(m, a, b), m ? T(ref_op(a, b)) : a, false);
      };

      // masked arithmetic
      bvv2v(
        "mask_add", [](auto m, auto a, auto b) { return grex::mask_add(m, a, b); }, std::plus{});
      bvv2v(
        "mask_subtract", [](auto m, auto a, auto b) { return grex::mask_subtract(m, a, b); },
        std::minus{});
      bvv2v(
        "mask_multiply", [](auto m, auto a, auto b) { return grex::mask_multiply(m, a, b); },
        std::multiplies{});
      if constexpr (grex::FloatVectorizable<T>) {
        bvv2v(
          "mask_divide", [](auto m, auto a, auto b) { return grex::mask_divide(m, a, b); },
          std::divides{});
      }

      // blend_zero
      {
        const bool m = bool(bdist(rng));
        const T a = dist(rng);
        test::check("blend_zero", grex::blend_zero(m, a), m ? a : T{}, false);
      }
      {
        const bool m = bool(bdist(rng));
        const T a = dist(rng);
        const T b = dist(rng);
        test::check("blend", grex::blend(m, a, b), m ? b : a, false);
      }
    }

    // vector-to-mask operations
    if constexpr (grex::FloatVectorizable<T>) {
      const T a = bool(bdist(rng)) ? dist(rng)
                                   : (bool(bdist(rng)) ? std::numeric_limits<T>::infinity()
                                                       : std::numeric_limits<T>::quiet_NaN());
      test::check("is_finite", grex::is_finite(a), std::isfinite(a), false);
    }
  }
}

int main() {
  pcg_extras::seed_seq_from<std::random_device> seed_source{};
  test::Rng rng{seed_source};
#if !GREX_BACKEND_SCALAR
  test::run_types_sizes([&](auto vtag, auto stag) { run_simd(rng, vtag, stag); });
#endif
  test::run_types([&](auto tag) { run_scalar(rng, tag); });
}
