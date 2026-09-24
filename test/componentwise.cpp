// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <cstddef>
#include <functional>
#include <random>

#include <fmt/base.h> // IWYU pragma: keep
#include <fmt/format.h>
#include <pcg_extras.hpp>

#include "grex/grex.hpp"

#include "defs.hpp"

namespace {
namespace test = grex::test;
inline constexpr std::size_t repetitions = 4096;

namespace ref {
using namespace grex::primitives;

/** Binary64 FMA. */
f64 fma(f64 a, f64 b, f64 c) { // NOLINT(*-unused-function)
  return __builtin_fma(a, b, c);
}
/** Binary32 FMA. */
f32 fma(f32 a, f32 b, f32 c) { // NOLINT(*-unused-function)
  return __builtin_fmaf(a, b, c);
}
/** Binary16 FMA. */
f16 fma(f16 a, f16 b, f16 c) { // NOLINT(*-unused-function)
  // Computed via binary64 rather than `__builtin_fmaf16` directly: the latter requires the target
  // libm to provide `fmaf16`, which is not universally available, whereas `fma` (binary64) is; both
  // are exactly correctly rounded, so this is an equally valid reference.
  return grex::f64_to_f16(
    __builtin_fma(grex::f16_to_f64(a), grex::f16_to_f64(b), grex::f16_to_f64(c)));
}
} // namespace ref

#if !GREX_BACKEND_SCALAR
#include <array>
#include <climits>

template<grex::Vectorizable T, std::size_t N>
void run_simd(test::Rng& rng, grex::TypeTag<T> /*tag*/, grex::IndexTag<N> /*tag*/) {
  using VC = test::VectorChecker<T, N>;
  using MC = test::MaskChecker<T, N>;

  auto dist = test::make_distribution<T>();
  auto dval = [&] { return dist(rng); };
  std::uniform_int_distribution<int> bdist{0, 1};
  auto bval = [&](std::size_t /*dummy*/) { return static_cast<bool>(bdist(rng)); };

  const auto nonfin = [&](std::size_t /*dummy*/) {
    if constexpr (grex::FloatVectorizable<T>) {
      if (bdist(rng)) {
        return dist(rng);
      }
      if (bdist(rng)) {
        const T inf = grex::NumericTrait<T>::infinity();
        return bdist(rng) ? inf : -inf;
      }
      return bdist(rng) ? grex::NumericTrait<T>::quiet_NaN()
                        : grex::NumericTrait<T>::signaling_NaN();
    }
  };

  grex::static_apply<N>([&]<std::size_t... I> {
    for (std::size_t i = 0; i < repetitions; ++i) {
      // vector-only operations
      {
        const auto v2vx = [&](const auto& label, auto vop, auto sop) {
          VC a = VC::random(dval);
          VC{vop(a.vec), std::array{T(sop(a.ref[I]))...}}.check(label, {.verbose = false});
        };
        const auto v2v = [&](const auto& label, auto op) { v2vx(label, op, op); };

        const auto vv2vx = [&](auto label, auto vop, auto sop) {
          const VC a = VC::random(dval);
          const VC b = VC::random(dval);
          const VC checker{vop(a.vec, b.vec), std::array{T(sop(a.ref[I], b.ref[I]))...}};
          checker.check(fmt::format("{}({}, {})", label, a.ref, b.ref),
                        {.verbose = false, .cmp_zero_sign = false});
        };
        const auto vv2v = [&](auto label, auto op) { vv2vx(label, op, op); };

        // arithmetic
        v2v("negate", std::negate{});
        vv2v("plus", std::plus{});
        vv2v("minus", std::minus{});
        vv2v("multiplies", std::multiplies{});
        if constexpr (grex::FloatVectorizable<T>) {
          vv2v("divides", std::divides{});
        }

        if constexpr (grex::IntVectorizable<T>) {
          const auto f = [&](grex::AnyIndexTag auto offset) {
            v2v(fmt::format("shift_left<{}>", offset.value), [&](auto v) {
              return v << offset; // NOLINT(*-signed-bitwise)
            });
            v2v(fmt::format("shift_right<{}>", offset.value), [&](auto v) {
              return v >> offset; // NOLINT(*-signed-bitwise)
            });
          };
          grex::static_apply<sizeof(T) * CHAR_BIT>(
            [&]<std::size_t... J> { (..., f(grex::index_tag<J>)); });
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
          v2vx(
            "abs", [](auto a) { return grex::abs(a); },
            [](auto a) { return std::abs(test::widen(a)); });
          v2v("abs", [](auto a) { return grex::abs(a); });
        }
        if constexpr (grex::FloatVectorizable<T>) {
          v2vx(
            "sqrt", [](auto a) { return grex::sqrt(a); },
            [](auto a) { return std::sqrt(test::widen(a)); });
          v2v("sqrt", [](auto a) { return grex::sqrt(a); });
        }

        // make_finite
        if constexpr (grex::FloatVectorizable<T>) {
          const VC a{nonfin(I)...};
          const VC checker{
            grex::make_finite(a.vec),
            std::array{(std::isfinite(test::widen(a.ref[I])) ? a.ref[I] : T{})...},
          };
          checker.check("make_finite", {.verbose = false});
          const VC gchecker{grex::make_finite(a.vec), std::array{grex::make_finite(a.ref[I])...}};
          gchecker.check("make_finite", {.verbose = false});
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
          const auto vvv2v = [&](auto label, auto grex_op, auto fused_op,
                                 [[maybe_unused]] auto fb_op) {
            const VC vca = VC::random(dval);
            const VC vcb = VC::random(dval);
            const VC vcc = VC::random(dval);

            const auto op = [&](T a, T b, T c) {
              if constexpr (grex::has_fma<T>) {
                // Use the built-in FMA at each width. This implicitly assumes that there is a
                // built-in for the platform on which this is executed, which is the case for all
                // non-scalar backends.
                return fused_op(a, b, c);
              } else if constexpr (grex::has_fma<test::Widened<T>>) {
                // Matches the backend’s fallback: a single binary32 fused multiply-add, rounded
                // once back down to binary16.
                return T(fused_op(test::widen(a), test::widen(b), test::widen(c)));
              } else {
                return fb_op(a, b, c);
              }
            };
            const VC checker{
              grex_op(vca.vec, vcb.vec, vcc.vec),
              std::array{T(op(vca.ref[I], vcb.ref[I], vcc.ref[I]))...},
            };
            checker.check(label, {.verbose = false});

            // using the scalar operation as reference
            const VC gchecker{
              grex_op(vca.vec, vcb.vec, vcc.vec),
              std::array{T(grex_op(vca.ref[I], vcb.ref[I], vcc.ref[I]))...},
            };
            gchecker.check(label, {.verbose = false});
          };

          vvv2v(
            "fmadd", [](auto a, auto b, auto c) { return grex::fmadd(a, b, c); },
            [](auto a, auto b, auto c) { return ref::fma(a, b, c); },
            [](auto a, auto b, auto c) { return a * b + c; });
          vvv2v(
            "fmsub", [](auto a, auto b, auto c) { return grex::fmsub(a, b, c); },
            [](auto a, auto b, auto c) { return ref::fma(a, b, -c); },
            [](auto a, auto b, auto c) { return a * b - c; });
          vvv2v(
            "fnmadd", [](auto a, auto b, auto c) { return grex::fnmadd(a, b, c); },
            [](auto a, auto b, auto c) { return ref::fma(-a, b, c); },
            [](auto a, auto b, auto c) { return c - a * b; });
          vvv2v(
            "fnmsub", [](auto a, auto b, auto c) { return grex::fnmsub(a, b, c); },
            [](auto a, auto b, auto c) { return ref::fma(-a, b, -c); },
            [](auto a, auto b, auto c) { return -(a * b) - c; });
        }
      }

      // masked component-wise operations
      {
        const auto mvv2v = [&](auto label, auto gop, auto sop) {
          const MC m{bval(I)...};
          const VC a = VC::random(dval);
          const VC b = VC::random(dval);
          const VC checker{
            gop(m.mask, a.vec, b.vec),
            std::array{T(m.ref[I] ? sop(a.ref[I], b.ref[I]) : a.ref[I])...},
          };
          checker.check(label, {.verbose = false});

          // using the scalar operation as reference
          const VC gchecker{
            gop(m.mask, a.vec, b.vec),
            std::array{gop(m.ref[I], a.ref[I], b.ref[I])...},
          };
          gchecker.check(label, {.verbose = false});
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
          const MC m{bval(I)...};
          const VC a = VC::random(dval);
          const VC checker{
            grex::blend_zero(m.mask, a.vec),
            std::array{T(m.ref[I] ? a.ref[I] : 0)...},
          };
          checker.check("blend_zero", {.verbose = false});
          const VC gchecker{
            grex::blend_zero(m.mask, a.vec),
            std::array{grex::blend_zero(m.ref[I], a.ref[I])...},
          };
          gchecker.check("blend_zero", {.verbose = false});
        }
        {
          const MC m{bval(I)...};
          const VC a = VC::random(dval);
          const VC b = VC::random(dval);
          const VC checker{
            grex::blend(m.mask, a.vec, b.vec),
            std::array<T, N>{T(m.ref[I] ? b.ref[I] : a.ref[I])...},
          };
          checker.check("blend", {.verbose = false});
          const VC gchecker{
            grex::blend(m.mask, a.vec, b.vec),
            std::array<T, N>{grex::blend(m.ref[I], a.ref[I], b.ref[I])...},
          };
          gchecker.check("blend", {.verbose = false});
        }
      }

      // vector-vector comparison operations
      {
        const auto vv2m = [&](auto label, auto op) {
          const VC a = VC::random(dval);
          const VC b = VC::random(dval);
          const MC checker{op(a.vec, b.vec), std::array{op(a.ref[I], b.ref[I])...}};
          checker.check(label, {.verbose = false});
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
        const VC a{nonfin(I)...};
        const MC checker{
          grex::is_finite(a.vec),
          std::array{std::isfinite(test::widen(a.ref[I]))...},
        };
        checker.check("is_finite", {.verbose = false});
        const MC gchecker{grex::is_finite(a.vec), std::array{grex::is_finite(a.ref[I])...}};
        gchecker.check("is_finite", {.verbose = false});
      }

      // mask-only operations
      {
        const auto m2m = [&](auto label, auto op) {
          const MC a{bval(I)...};
          const MC checker{op(a.mask), std::array{op(a.ref[I])...}};
          checker.check(label, {.verbose = false});
        };
        const auto mm2m = [&](auto label, auto op) {
          const MC a{bval(I)...};
          const MC b{bval(I)...};
          const MC checker{op(a.mask, b.mask), std::array{op(a.ref[I], b.ref[I])...}};
          checker.check(label, {.verbose = false});
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
      const auto v2v = [&](auto label, auto vop, auto sop) {
        const T a = dist(rng);
        test::check(label, vop(a), sop(test::widen(a)), {.verbose = false});
      };
      const auto vv2v = [&](auto label, auto vop, auto sop) {
        const T a = dist(rng);
        const T b = dist(rng);
        test::check(label, vop(a, b), sop(a, b), {.verbose = false, .cmp_zero_sign = false});
      };

      // abs/sqrt
      if constexpr (grex::SignedVectorizable<T>) {
        v2v("abs", [](auto a) { return grex::abs(a); }, [](auto a) { return T(std::abs(a)); });
      }
      if constexpr (grex::FloatVectorizable<T>) {
        v2v("sqrt", [](auto a) { return grex::sqrt(a); }, [](auto a) { return T(std::sqrt(a)); });
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
        const auto vvv2v = [&](auto label, auto grex_op, auto fused_op,
                               [[maybe_unused]] auto fb_op) {
          const T a = dist(rng);
          const T b = dist(rng);
          const T c = dist(rng);

          const auto ref = [&] {
            if constexpr (grex::has_fma<T>) {
              return fused_op(a, b, c);
            } else if constexpr (grex::has_fma<test::Widened<T>>) {
              // Matches the backend’s binary16 fallback: round-trip through binary32.
              return T(fused_op(test::widen(a), test::widen(b), test::widen(c)));
            } else {
              return fb_op(a, b, c);
            }
          }();
          test::check(label, grex_op(a, b, c), ref, {.verbose = false});
        };

        vvv2v(
          "fmadd", [](auto a, auto b, auto c) { return grex::fmadd(a, b, c); },
          [](auto a, auto b, auto c) { return ref::fma(a, b, c); },
          [](auto a, auto b, auto c) { return a * b + c; });
        vvv2v(
          "fmsub", [](auto a, auto b, auto c) { return grex::fmsub(a, b, c); },
          [](auto a, auto b, auto c) { return ref::fma(a, b, -c); },
          [](auto a, auto b, auto c) { return a * b - c; });
        vvv2v(
          "fnmadd", [](auto a, auto b, auto c) { return grex::fnmadd(a, b, c); },
          [](auto a, auto b, auto c) { return ref::fma(-a, b, c); },
          [](auto a, auto b, auto c) { return c - a * b; });
        vvv2v(
          "fnmsub", [](auto a, auto b, auto c) { return grex::fnmsub(a, b, c); },
          [](auto a, auto b, auto c) { return ref::fma(-a, b, -c); },
          [](auto a, auto b, auto c) { return -(a * b) - c; });
      }
    }

    // masked component-wise operations
    {
      const auto bvv2v = [&](auto label, auto grex_op, auto ref_op) {
        const bool m = static_cast<bool>(bdist(rng));
        const T a = dist(rng);
        const T b = dist(rng);
        test::check(label, grex_op(m, a, b), m ? T(ref_op(a, b)) : a, {.verbose = false});
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
        const bool m = static_cast<bool>(bdist(rng));
        const T a = dist(rng);
        test::check("blend_zero", grex::blend_zero(m, a), m ? a : T{}, {.verbose = false});
      }
      {
        const bool m = static_cast<bool>(bdist(rng));
        const T a = dist(rng);
        const T b = dist(rng);
        test::check("blend", grex::blend(m, a, b), m ? b : a, {.verbose = false});
      }
    }

    // vector-to-mask operations
    if constexpr (grex::FloatVectorizable<T>) {
      const T a = static_cast<bool>(bdist(rng))
                    ? dist(rng)
                    : (static_cast<bool>(bdist(rng)) ? grex::NumericTrait<T>::infinity()
                                                     : grex::NumericTrait<T>::quiet_NaN());
      test::check("is_finite", grex::is_finite(a), std::isfinite(test::widen(a)),
                  {.verbose = false});
    }
  }
}
} // namespace

int main() {
  pcg_extras::seed_seq_from<std::random_device> seed_source{};
  test::Rng rng{seed_source};
#if !GREX_BACKEND_SCALAR
  test::run_types_sizes([&](auto vtag, auto stag) { run_simd(rng, vtag, stag); });
#endif
  test::run_types([&](auto tag) { run_scalar(rng, tag); });
}
