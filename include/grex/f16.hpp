// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_F16_HPP
#define INCLUDE_GREX_F16_HPP

#include <bit>
#include <concepts>
#include <cstdint>
#include <type_traits>

// Whether the compiler provides a native IEEE 754 binary16 scalar type.
// Whether it does depends on the target architecture: It is always available on x86-64 and ARM64,
// but it is rejected outright on some architectures: Clang, for example, rejects it on PowerPC,
// MIPS, SPARC, and WebAssembly, and GCC supports a narrower set of architectures than Clang. The
// software emulation below covers those. This can be overridden by defining GREX_NATIVE_F16
// externally.
#ifndef GREX_NATIVE_F16
#if defined(__FLT16_MAX__) && (defined(__GNUC__) || defined(__clang__))
#define GREX_NATIVE_F16 true
#else
#define GREX_NATIVE_F16 false
#endif
#endif

namespace grex {
namespace f16_impl {
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using i32 = std::int32_t;
using f32 = float;
using f64 = double;

/** The IEEE 754 binary formats that are wider than binary16 and that binary16 converts to. */
template<typename T>
concept WideFloat = std::same_as<T, f32> || std::same_as<T, f64>;

/**
 * The parameters of the IEEE 754 binary format whose bits are held by the unsigned integer type
 * `TBits` and which stores `tMantissaBits` mantissa bits with the exponent bias `tExponentBias`.
 */
template<typename TBits, TBits tMantissaBits, TBits tExponentBias>
struct FormatBase {
  using Bits = TBits;

  static constexpr Bits mantissa_bits = tMantissaBits;
  static constexpr Bits exponent_bias = tExponentBias;
  /** The all-ones exponent, which denotes infinities and not-a-numbers. */
  static constexpr Bits exponent_max = 2 * tExponentBias + 1;
  static constexpr Bits mantissa_mask = (Bits{1} << tMantissaBits) - 1;
  /** The implicit leading mantissa bit of a normal value. */
  static constexpr Bits implicit_bit = Bits{1} << tMantissaBits;
  /** The bit pattern of positive infinity. */
  static constexpr Bits infinity = exponent_max << tMantissaBits;
  /** The shift between the binary16 sign bit and this format’s sign bit. */
  static constexpr Bits sign_shift = 8 * sizeof(Bits) - 16;
  /** The shift between the binary16 mantissa and the top of this format’s mantissa. */
  static constexpr Bits mantissa_shift = tMantissaBits - 10;
};

/** The parameters of the IEEE 754 binary format of `TFloat`. */
template<WideFloat TFloat>
struct Format;
template<>
struct Format<f32> : FormatBase<u32, 23, 127> {};
template<>
struct Format<f64> : FormatBase<u64, 52, 1023> {};

/** The unsigned integer type holding the bits of `TFloat`. */
template<WideFloat TFloat>
using BitsOf = Format<TFloat>::Bits;

/**
 * Converts the bits of an IEEE 754 binary16 value into the bits of the equal `TFloat` value.
 *
 * This conversion is always exact: Every binary16 value, including subnormals, infinities, and
 * quiet/signalling not-a-numbers, has an exact counterpart in every wider format.
 */
template<WideFloat TFloat>
constexpr BitsOf<TFloat> f16_bits_to_wide_bits(u16 half) {
  using Fmt = Format<TFloat>;
  using Bits = BitsOf<TFloat>;

  const Bits sign = (Bits{half} & 0x8000U) << Fmt::sign_shift;
  const Bits expo = (Bits{half} >> 10U) & 0x1FU;
  const Bits mant = Bits{half} & 0x3FFU;

  if (expo == 0) {
    if (mant == 0) {
      return sign; // ±0
    }
    // Subnormal binary16 values are normal values in every wider format: Shift the mantissa up
    // until the implicit bit is set and adapt the exponent accordingly.
    Bits shift = 0;
    Bits shifted = mant;
    for (bool go_on = true; go_on; go_on = (shifted & 0x400U) == 0) {
      ++shift;
      shifted <<= 1U;
    }
    return sign | ((Fmt::exponent_bias - 15U - shift + 1U) << Fmt::mantissa_bits) |
           ((shifted & 0x3FFU) << Fmt::mantissa_shift);
  }
  if (expo == 0x1FU) {
    // Infinity/not-a-number: The mantissa (and thus the “quiet” bit) is preserved.
    return sign | Fmt::infinity | (mant << Fmt::mantissa_shift);
  }
  return sign | ((expo - 15U + Fmt::exponent_bias) << Fmt::mantissa_bits) |
         (mant << Fmt::mantissa_shift);
}

/**
 * Converts the bits of an IEEE 754 `TFloat` value into the bits of the nearest binary16 value,
 * rounding ties to even, which is the default rounding mode.
 */
template<WideFloat TFloat>
constexpr u16 wide_bits_to_f16_bits(BitsOf<TFloat> wide) {
  using Fmt = Format<TFloat>;
  using Bits = BitsOf<TFloat>;

  const auto sign = u16((wide >> Fmt::sign_shift) & 0x8000U);
  const Bits biased = (wide >> Fmt::mantissa_bits) & Fmt::exponent_max;
  const Bits mant = wide & Fmt::mantissa_mask;

  if (biased == Fmt::exponent_max) {
    if (mant == 0) {
      return u16(sign | 0x7C00U);
    }
    // Not-a-number: Make quiet and ensure that the mantissa does not become zero, which would turn
    // the value into an infinity.
    return u16(sign | 0x7E00U | u16(mant >> Fmt::mantissa_shift));
  }

  const i32 expo = i32(biased) - i32(Fmt::exponent_bias) + 15;
  if (expo >= 0x1F) {
    // Overflow (including the case that the source is already infinite) → ±∞.
    return u16(sign | 0x7C00U);
  }
  if (expo <= 0) {
    if (expo < -10) {
      // Too small even for the smallest subnormal, i.e. below half of 2⁻²⁴ → ±0.
      return sign;
    }
    // Subnormal binary16 result: Re-introduce the implicit bit and shift it into place, i.e.
    // express the value `full · 2^(expo - 15 - mantissa_bits)` in units of the smallest binary16
    // subnormal 2⁻²⁴.
    const auto shift = Bits(i32(Fmt::mantissa_bits) - 9 - expo);
    const Bits full = mant | Fmt::implicit_bit;
    Bits out = full >> shift;
    const Bits rest = full & ((Bits{1} << shift) - 1U);
    const Bits tie = Bits{1} << (shift - 1U);
    if (rest > tie || (rest == tie && (out & 1U) != 0)) {
      ++out;
    }
    return u16(sign | u16(out));
  }

  // A carry out of the mantissa increments the exponent, which is exactly the desired behaviour and
  // also produces ±∞ if the exponent overflows.
  Bits out = (Bits(expo) << 10U) | (mant >> Fmt::mantissa_shift);
  const Bits rest = mant & ((Bits{1} << Fmt::mantissa_shift) - 1U);
  const Bits tie = Bits{1} << (Fmt::mantissa_shift - 1U);
  if (rest > tie || (rest == tie && (out & 1U) != 0)) {
    ++out;
  }
  return u16(sign | u16(out));
}

/** Converts the bits of an IEEE 754 binary16 value into the equal `TFloat` value. */
template<WideFloat TFloat>
constexpr TFloat f16_bits_to_wide(u16 half) {
  return std::bit_cast<TFloat>(f16_bits_to_wide_bits<TFloat>(half));
}
/** Converts a `TFloat` value into the bits of the nearest binary16 value, rounding ties to even. */
template<WideFloat TFloat>
constexpr u16 wide_to_f16_bits(TFloat wide) {
  return wide_bits_to_f16_bits<TFloat>(std::bit_cast<BitsOf<TFloat>>(wide));
}

/**
 * Converts an arbitrary arithmetic value into the bits of the nearest binary16 value.
 *
 * The conversion goes through binary32 for the types that binary32 represents exactly and through
 * binary64 otherwise, which avoids the double rounding a detour through binary32 would introduce.
 * The only types binary64 does not represent exactly either are the 64-bit integers and, where it
 * is wider, `long double`, for which rounding twice remains possible.
 */
template<typename T>
requires(std::is_arithmetic_v<T>)
constexpr u16 arithmetic_to_f16_bits(T value) {
  if constexpr (sizeof(T) <= 2 || std::same_as<T, f32>) {
    return wide_to_f16_bits(f32(value));
  } else {
    return wide_to_f16_bits(f64(value));
  }
}
} // namespace f16_impl

namespace primitives {
#if GREX_NATIVE_F16
/** The compiler-provided IEEE 754 binary16 (“half-precision”) floating-point type. */
using f16 = _Float16;
#else
/**
 * IEEE 754 binary16 (“half-precision”) floating-point type.
 *
 * This software emulation is used on platforms without a native `_Float16`. All operations are
 * carried out in binary32 and rounded to binary16 afterwards, which yields exactly the same results
 * as native binary16 arithmetic for addition, subtraction, multiplication, division, and square
 * roots, since binary32 provides more than twice as many mantissa bits plus two. Conversions from
 * types that binary32 does not represent exactly, above all binary64, are carried out directly,
 * without the double rounding a detour through binary32 would introduce.
 */
struct f16 {
  std::uint16_t bits;

  f16() = default;

  /** Converts an arbitrary arithmetic value into the nearest binary16 value. */
  template<typename T>
  requires(std::is_arithmetic_v<T>)
  constexpr f16(T value) // NOLINT(*-explicit-conversions)
      : bits{f16_impl::arithmetic_to_f16_bits(value)} {}

  /** Converts to binary32, which is always exact. */
  constexpr operator float() const { // NOLINT(*-explicit-conversions)
    return f16_impl::f16_bits_to_wide<float>(bits);
  }

  constexpr f16 operator+() const {
    return *this;
  }
  constexpr f16 operator-() const {
    return from_bits(std::uint16_t(bits ^ 0x8000U));
  }

  /** Creates a binary16 value from its bit pattern. */
  static constexpr f16 from_bits(std::uint16_t b) {
    f16 out{};
    out.bits = b;
    return out;
  }

#define GREX_F16_BINOP(OP) \
  friend constexpr f16 operator OP(f16 a, f16 b) { \
    return f16{float(a) OP float(b)}; \
  } \
  friend constexpr f16& operator OP## = (f16 & a, f16 b) { \
    return a = a OP b; \
  }
  GREX_F16_BINOP(+)
  GREX_F16_BINOP(-)
  GREX_F16_BINOP(*)
  GREX_F16_BINOP(/)
#undef GREX_F16_BINOP

  /* Not-a-number handling requires that the comparisons are carried out in binary32. */
  friend constexpr bool operator==(f16 a, f16 b) {
    return float(a) == float(b);
  }
  friend constexpr auto operator<=>(f16 a, f16 b) {
    return float(a) <=> float(b);
  }
};
#endif
} // namespace primitives

using primitives::f16;

/** The bit pattern of a binary16 value. */
constexpr std::uint16_t f16_bits(f16 value) {
  return std::bit_cast<std::uint16_t>(value);
}
/** The binary16 value with the given bit pattern. */
constexpr f16 f16_from_bits(std::uint16_t bits) {
  return std::bit_cast<f16>(bits);
}

/** Converts a binary16 value to binary32, which is always exact. */
constexpr float f16_to_f32(f16 value) {
#if GREX_NATIVE_F16
  return float(value);
#else
  return f16_impl::f16_bits_to_wide<float>(value.bits);
#endif
}
/** Converts a binary32 value to the nearest binary16 value, rounding ties to even. */
constexpr f16 f32_to_f16(float value) {
#if GREX_NATIVE_F16
  return f16(value);
#else
  return f16{value};
#endif
}

/** Converts a binary16 value to binary64, which is always exact. */
constexpr double f16_to_f64(f16 value) {
#if GREX_NATIVE_F16
  return double(value);
#else
  return f16_impl::f16_bits_to_wide<double>(value.bits);
#endif
}
/**
 * Converts a binary64 value to the nearest binary16 value, rounding ties to even.
 *
 * The conversion is direct rather than by way of binary32, which would round twice.
 */
constexpr f16 f64_to_f16(double value) {
#if GREX_NATIVE_F16
  return f16(value);
#else
  return f16{value};
#endif
}
} // namespace grex

#endif // INCLUDE_GREX_F16_HPP
