// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>

#include <fmt/base.h>
#include <fmt/color.h>
#include <fmt/format.h>
#include <pcg_extras.hpp>

#include "grex/backend/active/operations/f16.hpp"
#include "grex/grex.hpp"

#include "defs.hpp"

namespace {
namespace test = grex::test;
using namespace grex::primitives;

inline constexpr std::size_t repetitions = 1024;

// The largest finite binary16 value is 65504 and the next representable value would be 65536,
// so everything from their midpoint on rounds to infinity.
inline constexpr f64 f16_infinity_threshold = 65520;

// The reference implementations below are deliberately independent of the implementation under
// test: They decompose and compose the bit patterns using `std::ldexp`/`std::frexp` and rely on
// `std::nearbyint` for rounding to nearest with ties to even, which is the default rounding mode.

/**
 * The exact value of the binary16 with the bit pattern `bits`, which always fits into a binary64.
 */
f64 reference_value(u16 bits) {
  const f64 sign = ((bits >> 15U) != 0) ? -1 : 1;
  const auto expo = static_cast<int>((bits >> 10U) & 0x1FU); // NOLINT(*-signed-bitwise)
  const u32 mant = bits & 0x3FFU;
  if (expo == 0) {
    // Subnormal: mant · 2⁻²⁴.
    return sign * std::ldexp(static_cast<f64>(mant), -24);
  }
  if (expo == 0x1F) {
    return (mant == 0) ? sign * std::numeric_limits<f64>::infinity()
                       : std::numeric_limits<f64>::quiet_NaN();
  }
  // Normal: 1.mant · 2^(expo - 15) = (1024 + mant) · 2^(expo - 25).
  return sign * std::ldexp(static_cast<f64>(mant | 0x400U), expo - 25);
}

/** The bit pattern of the binary16 nearest to `value`, rounding ties to even. */
u16 reference_bits(f64 value) {
  const auto sign = static_cast<u16>(std::signbit(value) ? 0x8000U : 0U);
  if (std::isnan(value)) {
    return static_cast<u16>(sign | 0x7E00U);
  }
  const f64 mag = std::abs(value);
  if (mag >= f16_infinity_threshold) {
    return static_cast<u16>(sign | 0x7C00U);
  }
  if (mag < std::ldexp(1.0, -14)) {
    // Zero or subnormal, i.e. mant · 2⁻²⁴ with mant < 1024.
    const auto mant = static_cast<u32>(std::nearbyint(std::ldexp(mag, 24)));
    // Rounding up may produce the smallest normal value, whose bit pattern follows seamlessly.
    return static_cast<u16>(sign | static_cast<u16>(mant));
  }
  int expo2 = 0;
  std::frexp(mag, &expo2);
  // `frexp` returns mag = f · 2^expo2 with f ∈ [0.5, 1), so the unbiased exponent is expo2 - 1.
  int unbiased = expo2 - 1;
  auto mant = static_cast<u32>(std::nearbyint(std::ldexp(mag, 10 - unbiased)));
  if (mant == 2048) {
    // Rounding carried into the exponent.
    mant = 1024;
    ++unbiased;
  }
  if (unbiased > 15) {
    return static_cast<u16>(sign | 0x7C00U);
  }
  return static_cast<u16>(
    sign | static_cast<u16>(((static_cast<u32>(unbiased + 15) << 10U) | (mant - 1024))));
}

/** Whether two binary16 bit patterns denote the same value, treating all not-a-numbers as equal. */
bool same_f16(u16 a, u16 b) {
  const bool a_nan = (a & 0x7C00U) == 0x7C00U && (a & 0x3FFU) != 0;
  const bool b_nan = (b & 0x7C00U) == 0x7C00U && (b & 0x3FFU) != 0;
  if (a_nan || b_nan) {
    return a_nan && b_nan && ((a ^ b) & 0x8000U) == 0; // NOLINT(bugprone-signed-bitwise)
  }
  return a == b;
}

void check_f16(const auto& label, f16 value, u16 reference, bool verbose = false) {
  test::check_msg(label, same_f16(grex::f16_bits(value), reference),
                  fmt::format("{:#06x}", grex::f16_bits(value)), fmt::format("{:#06x}", reference),
                  verbose);
}

// =================================================================================================
// Scalar semantics, which are available with every back-end
// =================================================================================================

/** Every binary16 value has an exact binary32 and binary64 counterpart. */
void run_scalar_conversions() {
  for (u32 bits = 0; bits < 0x10000; ++bits) {
    const f16 value = grex::f16_from_bits(static_cast<u16>(bits));
    const f32 single = grex::f16_to_f32(value);
    const f64 dbl = grex::f16_to_f64(value);
    const f64 reference = reference_value(static_cast<u16>(bits));

    if (std::isnan(reference)) {
      test::check_msg("f16_to_f32 not-a-number", std::isnan(single), single, reference, false);
      test::check_msg("f16_to_f64 not-a-number", std::isnan(dbl), dbl, reference, false);
    } else {
      test::check_msg("f16_to_f32", static_cast<f64>(single) == reference, single, reference,
                      false);
      test::check_msg("f16_to_f64", dbl == reference, dbl, reference, false);
    }

    // Converting back is lossless.
    check_f16("round trip", grex::f32_to_f16(single), static_cast<u16>(bits));
    check_f16("round trip via f64", grex::f64_to_f16(dbl), static_cast<u16>(bits));
  }
}

/** Rounding from binary32 to binary16 is round-to-nearest, ties to even. */
void run_scalar_rounding(test::Rng& rng) {
  const auto check = [](u32 bits) {
    const f32 single = std::bit_cast<f32>(bits);
    check_f16([&] { return fmt::format("f32_to_f16({:#010x})", bits); }, grex::f32_to_f16(single),
              reference_bits(static_cast<f64>(single)));
  };

  // Every binary16 value, its immediate binary32 neighbours, and the midpoints between consecutive
  // binary16 values, which are exactly the ties.
  for (u32 bits = 0; bits < 0x10000; ++bits) {
    const u32 base =
      std::bit_cast<u32>(grex::f16_to_f32(grex::f16_from_bits(static_cast<u16>(bits))));
    for (int delta = -2; delta <= 2; ++delta) {
      check(static_cast<u32>(static_cast<int>(base) + delta));
    }
    check(base ^ 0x1000U);
    check(base | 0x1000U);
  }
  for (std::size_t i = 0; i < (1U << 20U); ++i) {
    check(static_cast<u32>(rng()));
  }
}

/**
 * Rounding from binary64 to binary16 is round-to-nearest, ties to even, in a single step: Rounding
 * to binary32 first would round a second time, which gives a different result for the binary64
 * values immediately next to a tie between two binary16 values.
 */
void run_scalar_f64_rounding(test::Rng& rng) {
  const auto check = [](f64 value) {
    check_f16([&] { return fmt::format("f64_to_f16({})", value); }, grex::f64_to_f16(value),
              reference_bits(value));
  };
  const auto check_around = [&](f64 value) {
    const auto bits = std::bit_cast<u64>(value);
    for (int delta = -2; delta <= 2; ++delta) {
      check(std::bit_cast<f64>(static_cast<u64>(static_cast<i64>(bits) + delta)));
    }
  };

  // Every finite binary16 value, the midpoints between consecutive binary16 values, which are
  // exactly the ties, and the binary64 neighbourhoods of both.
  for (u32 bits = 0; bits + 1 < 0x10000; ++bits) {
    const f64 base = reference_value(static_cast<u16>(bits));
    const f64 next = reference_value(static_cast<u16>(bits + 1));
    if (!std::isfinite(base) || !std::isfinite(next)) {
      continue;
    }
    check_around(base);
    check_around((base + next) / 2);
  }
  // The thresholds towards ±∞ and ±0, which have no binary16 value on the other side.
  check_around(f16_infinity_threshold);
  check_around(-f16_infinity_threshold);
  check_around(std::ldexp(1.0, -25));
  check_around(std::ldexp(-1.0, -25));

  for (std::size_t i = 0; i < (1U << 20U); ++i) {
    check(std::bit_cast<f64>(rng()));
  }
  // Random values across the whole binary16 range, including the subnormals and the overflow.
  for (std::size_t i = 0; i < (1U << 20U); ++i) {
    const f64 mag =
      std::ldexp(static_cast<f64>(rng() >> 11U) * 0x1p-53, static_cast<int>(rng() % 46) - 30);
    check(((rng() & 1U) != 0) ? -mag : mag);
  }
}

void run_scalar_operations(test::Rng& rng) {
  auto dist = test::make_distribution<f16>(); // NOLINT(*-const-correctness)

  for (std::size_t i = 0; i < repetitions; ++i) {
    const f16 a = dist(rng);
    const f16 b = dist(rng);
    const f64 fa = static_cast<f64>(grex::f16_to_f32(a));
    const f64 fb = static_cast<f64>(grex::f16_to_f32(b));

    check_f16("abs", grex::abs(a), reference_bits(std::abs(fa)));
    check_f16("min", grex::min(a, b), reference_bits(std::min(fa, fb)));
    check_f16("max", grex::max(a, b), reference_bits(std::max(fa, fb)));
    check_f16("sqrt", grex::sqrt(grex::abs(a)), reference_bits(std::sqrt(std::abs(fa))));
    // With genuinely fused binary16 instructions, the result is a single rounding of the exact
    // value: the product is exact in binary64 (it always fits into 22 bits), so plain binary64
    // arithmetic below computes it losslessly, and the final rounding down to binary16 has ample
    // headroom to be exact. Without them, the backend falls back to a single binary32 fused
    // multiply-add followed by one more rounding down to binary16 (see
    // grex/backend/neon/operations/fmadd-family.hpp), so the reference mirrors that instead.
    if (grex::has_fma<f16>) {
      const f64 product = fa * fb;
      check_f16("fmadd", grex::fmadd(a, b, b), reference_bits(product + fb));
      check_f16("fmsub", grex::fmsub(a, b, b), reference_bits(product - fb));
      check_f16("fnmadd", grex::fnmadd(a, b, b), reference_bits(fb - product));
      check_f16("fnmsub", grex::fnmsub(a, b, b), reference_bits(-product - fb));
    } else {
      const f32 fa32 = static_cast<f32>(fa);
      const f32 fb32 = static_cast<f32>(fb);
      check_f16("fmadd", grex::fmadd(a, b, b),
                reference_bits(static_cast<f64>(std::fma(fa32, fb32, fb32))));
      check_f16("fmsub", grex::fmsub(a, b, b),
                reference_bits(static_cast<f64>(std::fma(fa32, fb32, -fb32))));
      check_f16("fnmadd", grex::fnmadd(a, b, b),
                reference_bits(static_cast<f64>(std::fma(-fa32, fb32, fb32))));
      check_f16("fnmsub", grex::fnmsub(a, b, b),
                reference_bits(static_cast<f64>(std::fma(-fa32, fb32, -fb32))));
    }
    test::check("is_finite", grex::is_finite(a), std::isfinite(fa), {.verbose = false});
    test::check("convert to f32", grex::convert<f32>(a), static_cast<f32>(fa), {.verbose = false});
    check_f16("convert from f32", grex::convert<f16>(static_cast<f32>(fa)), grex::f16_bits(a));
  }

  // Non-finite values.
  const f16 inf = grex::f32_to_f16(std::numeric_limits<f32>::infinity());
  const f16 nan = grex::f32_to_f16(std::numeric_limits<f32>::quiet_NaN());
  test::check("is_finite(∞)", grex::is_finite(inf), false, {.verbose = false});
  test::check("is_finite(nan)", grex::is_finite(nan), false, {.verbose = false});
  check_f16("make_finite(∞)", grex::make_finite(inf), 0);
  check_f16("make_finite(nan)", grex::make_finite(nan), 0);
  check_f16("-∞", grex::f32_to_f16(-std::numeric_limits<f32>::infinity()), 0xFC00);
  check_f16("overflow", grex::f32_to_f16(1e30F), 0x7C00);
  check_f16("underflow", grex::f32_to_f16(1e-30F), 0x0000);
  check_f16("max", grex::NumericTrait<f16>::max(), 0x7BFF);
  check_f16("min", grex::NumericTrait<f16>::min(), 0x0400);
  check_f16("epsilon", grex::NumericTrait<f16>::epsilon(), 0x1400);
  static_assert(grex::NumericTrait<f16>::digits == 11);
  static_assert(grex::SafeConversion<f32, f16>);
  static_assert(grex::SafeConversion<f64, f16>);
  static_assert(!grex::SafeConversion<f16, f32>);
  static_assert(grex::SafeConversion<f16, grex::i8>);
  static_assert(grex::SafeConversion<f16, grex::u8>);
  static_assert(!grex::SafeConversion<f16, grex::u16>);
  static_assert(!grex::SafeConversion<grex::u16, f16>);
}

#if !GREX_BACKEND_SCALAR
#include <array>
#include <bit>

/**
 * The portable software conversion, which is used by back-ends without half-precision conversion
 * instructions, agrees with the scalar conversion for every binary16 value.
 */
void run_software_conversion(test::Rng& rng) {
  constexpr std::size_t size = grex::min_native_size<f16>;
  using F16Vec = grex::Vector<f16, size>;
  using SingleVec = grex::Vector<f32, size>;

  // Binary16 → binary32: Exhaustive.
  for (u32 base = 0; base < 0x10000; base += size) {
    const auto halves = grex::static_apply<size>([&]<std::size_t... I> {
      return std::array{grex::f16_from_bits(static_cast<u16>(base + I))...};
    });
    const auto in = F16Vec::load(halves.data());
    const SingleVec out{grex::backend::f16_to_f32(in.backend())};
    const auto ref = in.convert(grex::type_tag<f32>);
    test::check("f16_to_f32", out.as_array(), ref.as_array(), {.verbose = false});
  }

  // Binary32 → binary16: All binary16 values, their neighbourhoods, the ties, and random values.
  std::array<f32, size> buf{};
  std::size_t filled = 0;
  auto flush = [&] {
    const auto in = SingleVec::load(buf.data());
    const F16Vec out{grex::backend::f32_to_f16(in.backend())};
    const auto array = out.as_array();
    for (std::size_t i = 0; i < size; ++i) {
      check_f16([&] { return fmt::format("f32_to_f16({})", buf[i]); }, array[i],
                reference_bits(static_cast<f64>(buf[i])));
    }
    filled = 0;
  };
  const auto push = [&](u32 bits) {
    buf[filled++] = std::bit_cast<f32>(bits);
    if (filled == size) {
      flush();
    }
  };
  for (u32 bits = 0; bits < 0x10000; ++bits) {
    const u32 base =
      std::bit_cast<u32>(grex::f16_to_f32(grex::f16_from_bits(static_cast<u16>(bits))));
    for (int delta = -2; delta <= 2; ++delta) {
      push(static_cast<u32>(static_cast<int>(base) + delta));
    }
    push(base ^ 0x1000U);
    push(base | 0x1000U);
  }
  for (std::size_t i = 0; i < (1U << 19U); ++i) {
    push(static_cast<u32>(rng()));
  }
  while (filled != 0) {
    push(0);
  }
}
#endif
} // namespace

int main() {
  test::Rng rng{pcg_extras::seed_seq_from<std::random_device>{}};

  fmt::print(fmt::fg(fmt::terminal_color::blue), "scalar conversions\n");
  run_scalar_conversions();
  fmt::print(fmt::fg(fmt::terminal_color::blue), "scalar rounding\n");
  run_scalar_rounding(rng);
  fmt::print(fmt::fg(fmt::terminal_color::blue), "scalar binary64 rounding\n");
  run_scalar_f64_rounding(rng);
  fmt::print(fmt::fg(fmt::terminal_color::blue), "scalar operations\n");
  run_scalar_operations(rng);

#if !GREX_BACKEND_SCALAR
  fmt::print(fmt::fg(fmt::terminal_color::blue), "software conversion\n");
  run_software_conversion(rng);
#endif

  fmt::print(fmt::fg(fmt::terminal_color::green), "All binary16 tests passed!\n");
  return 0;
}
