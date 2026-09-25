// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BASE_HPP
#define INCLUDE_GREX_BASE_HPP

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <utility>

#include "grex/f16.hpp" // IWYU pragma: export

#if defined(__GNUC__) && !defined(__clang__)
#define GREX_GCC true
#define GREX_CLANG false
#elifdef __clang__
#define GREX_GCC false
#define GREX_CLANG true
#endif

#ifdef __GNUC__
#define GREX_ALWAYS_INLINE __attribute__((always_inline))
#else
#define GREX_ALWAYS_INLINE
#endif

namespace grex {
namespace primitives {
using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using i8 = std::int8_t;
using i16 = std::int16_t;
using i32 = std::int32_t;
using i64 = std::int64_t;

using f32 = float;
static_assert(std::numeric_limits<f32>::is_iec559 && sizeof(f32) == 4);
using f64 = double;
static_assert(std::numeric_limits<f64>::is_iec559 && sizeof(f64) == 8);
// f16 is provided by f16.hpp, which is included above
static_assert(sizeof(f16) == 2);
} // namespace primitives
using namespace primitives;

template<typename T>
concept Int8 = std::same_as<T, u8> || std::same_as<T, i8>;
template<typename T>
concept Int16 = std::same_as<T, u16> || std::same_as<T, i16>;
template<typename T>
concept Int32 = std::same_as<T, u32> || std::same_as<T, i32>;
template<typename T>
concept Int64 = std::same_as<T, u64> || std::same_as<T, i64>;
/** IEEE 754 binary16: not always supported in hardware and therefore often a special case. */
template<typename T>
concept Float16 = std::same_as<T, f16>;

template<typename T>
concept UnsignedIntVectorizable =
  std::same_as<T, u8> || std::same_as<T, u16> || std::same_as<T, u32> || std::same_as<T, u64>;
template<typename T>
concept SignedIntVectorizable =
  std::same_as<T, i8> || std::same_as<T, i16> || std::same_as<T, i32> || std::same_as<T, i64>;
/** The floating-point types that are always supported. */
template<typename T>
concept FullFloatVectorizable = std::same_as<T, f32> || std::same_as<T, f64>;
template<typename T>
concept FloatVectorizable = FullFloatVectorizable<T> || Float16<T>;

template<typename T>
concept IntVectorizable = UnsignedIntVectorizable<T> || SignedIntVectorizable<T>;
template<typename T>
concept SignedVectorizable = SignedIntVectorizable<T> || FloatVectorizable<T>;
template<typename T>
concept UnsignedVectorizable = UnsignedIntVectorizable<T>;
template<typename T>
concept Vectorizable = IntVectorizable<T> || FloatVectorizable<T>;

template<typename T>
struct SignednessTrait;
#define GREX_DEF_SIGNEDNESS(U, S) \
  template<> \
  struct SignednessTrait<U> { \
    static constexpr bool is_signed = false; \
    using Unsigned = U; \
    using Signed = S; \
  }; \
  template<> \
  struct SignednessTrait<S> { \
    static constexpr bool is_signed = true; \
    using Unsigned = U; \
    using Signed = S; \
  };
GREX_DEF_SIGNEDNESS(u8, i8)
GREX_DEF_SIGNEDNESS(u16, i16)
GREX_DEF_SIGNEDNESS(u32, i32)
GREX_DEF_SIGNEDNESS(u64, i64)
#undef GREX_DEF_SIGNEDNESS
template<typename T>
using UnsignedOf = SignednessTrait<T>::Unsigned;
template<typename T>
using SignedOf = SignednessTrait<T>::Signed;
template<typename T>
static constexpr bool is_signed = SignednessTrait<T>::is_signed;

template<std::size_t Bytes>
struct SizedIntegerTrait;
#define GREX_DEF_SIZEDI(B, U, S) \
  template<> \
  struct SizedIntegerTrait<B> { \
    using Unsigned = U; \
    using Signed = S; \
  };
GREX_DEF_SIZEDI(1, u8, i8)
GREX_DEF_SIZEDI(2, u16, i16)
GREX_DEF_SIZEDI(4, u32, i32)
GREX_DEF_SIZEDI(8, u64, i64)
#undef GREX_DEF_SIZEDI
template<std::size_t Bytes>
using UnsignedInt = SizedIntegerTrait<Bytes>::Unsigned;
template<std::size_t Bytes>
using SignedInt = SizedIntegerTrait<Bytes>::Signed;
template<FloatVectorizable T>
using FloatSize = UnsignedInt<sizeof(T)>;
template<typename T, std::size_t Bytes>
using CopySignInt = std::conditional_t<is_signed<T>, SignedInt<Bytes>, UnsignedInt<Bytes>>;

template<std::size_t Bytes>
struct FloatTrait;
template<>
struct FloatTrait<2> {
  using Type = f16;
};
template<>
struct FloatTrait<4> {
  using Type = f32;
};
template<>
struct FloatTrait<8> {
  using Type = f64;
};
template<std::size_t Bytes>
using Float = FloatTrait<Bytes>::Type;

/**
 * Numeric properties of a vectorizable type.
 *
 * This mirrors the subset of `std::numeric_limits` that Grex requires, since `std::numeric_limits`
 * is not specialized for `_Float16` by every standard library and may not be specialized for it
 * by Grex, as it is not a program-defined type.
 */
template<typename T>
struct NumericTrait : std::numeric_limits<T> {};
template<>
struct NumericTrait<f16> {
  static constexpr int digits = 11;
  static constexpr int min_exponent = -13;
  static constexpr int max_exponent = 16;

  static constexpr f16 min() {
    return f16_from_bits(0x0400); // 2⁻¹⁴
  }
  static constexpr f16 max() {
    return f16_from_bits(0x7BFF); // 65504
  }
  static constexpr f16 epsilon() {
    return f16_from_bits(0x1400); // 2⁻¹⁰
  }
  static constexpr f16 infinity() {
    return f16_from_bits(0x7C00);
  }
  // Quiet (the top mantissa bit set) and signalling (clear, with some other mantissa bit set
  // instead) not-a-number values, matching the convention `wide_bits_to_f16_bits` (grex/f16.hpp)
  // itself produces when quietening a not-a-number during conversion.
  static constexpr f16 quiet_NaN() { // NOLINT(*-identifier-naming)
    return f16_from_bits(0x7E00);
  }
  static constexpr f16 signaling_NaN() { // NOLINT(*-identifier-naming)
    return f16_from_bits(0x7D00);
  }
};

template<typename T>
struct TypeTag {
  using Type = T;
};
template<typename T>
inline constexpr TypeTag<T> type_tag{};

template<typename T, T V>
struct ValueTag {
  using Value = T;
  static constexpr T value = V;
  constexpr operator T() const { // NOLINT
    return value;
  }
};
template<auto V>
using AutoTag = ValueTag<std::decay_t<decltype(V)>, V>;
template<int V>
using IntTag = AutoTag<V>;
template<std::size_t V>
using IndexTag = AutoTag<V>;
template<bool V>
using BoolTag = AutoTag<V>;

template<typename T, T V>
inline constexpr ValueTag<T, V> value_tag{};
template<auto V>
inline constexpr AutoTag<V> auto_tag{}; // NOLINT(modernize-avoid-c-style-cast)
template<int V>
inline constexpr IntTag<V> int_tag{};
template<std::size_t V>
inline constexpr IndexTag<V> index_tag{};
template<bool V>
inline constexpr BoolTag<V> bool_tag{};
inline constexpr BoolTag<true> true_tag{};
inline constexpr BoolTag<false> false_tag{};

template<bool IsSafe>
struct CastTag {
  static constexpr bool is_safe = IsSafe;
};
inline constexpr CastTag<true> safe_tag{};
inline constexpr CastTag<false> unsafe_tag{};

template<typename T, typename TRef>
concept SameAsDecayed = std::same_as<std::decay_t<T>, std::decay_t<TRef>>;
template<typename TTag>
concept AnyValueTag = requires {
  { auto_tag<TTag::value> };
};
template<typename TTag, typename TVal>
concept TypedValueTag = requires {
  { TTag::value } -> SameAsDecayed<TVal>;
};
template<typename TTag>
concept AnyIndexTag = TypedValueTag<TTag, std::size_t>;
template<typename TTag>
concept AnyIntTag = TypedValueTag<TTag, int>;
template<typename TTag>
concept AnyBoolTag = TypedValueTag<TTag, bool>;

enum struct ShuffleIndex : u8 { any = 254, zero = 255 };
inline constexpr ShuffleIndex any_sh = ShuffleIndex::any;
inline constexpr ShuffleIndex zero_sh = ShuffleIndex::zero;

constexpr bool is_index(ShuffleIndex sh) {
  return static_cast<u8>(sh) < static_cast<u8>(any_sh);
}
namespace literals {
consteval ShuffleIndex operator""_sh(unsigned long long int v) {
  if (v < 254) {
    return ShuffleIndex(v);
  }
  throw std::invalid_argument{"Value too large!"};
}
} // namespace literals

enum struct BlendZeroSelector : u8 { zero = 0, keep = 1, any = 2 };
inline constexpr BlendZeroSelector zero_bz = BlendZeroSelector::zero;
inline constexpr BlendZeroSelector keep_bz = BlendZeroSelector::keep;
inline constexpr BlendZeroSelector any_bz = BlendZeroSelector::any;

enum struct BlendSelector : u8 { lhs = 0, rhs = 1, any = 2 };
inline constexpr BlendSelector lhs_bl = BlendSelector::lhs;
inline constexpr BlendSelector rhs_bl = BlendSelector::rhs;
inline constexpr BlendSelector any_bl = BlendSelector::any;

enum struct IterDirection : bool { forward, backward };
inline std::string_view format_as(IterDirection dir) {
  switch (dir) {
    case IterDirection::forward: return "forward";
    case IterDirection::backward: return "backward";
  }
}

template<typename It>
concept MultiByteIterator = requires(It it) {
  typename It::Container;
  { It::Container::element_bytes } -> std::convertible_to<std::size_t>;
  { it.raw() } -> std::convertible_to<const std::byte*>;
};

template<std::size_t I, typename T>
using IdxType = T;

template<std::size_t N>
GREX_ALWAYS_INLINE constexpr decltype(auto) static_apply(auto f) {
  return [&]<std::size_t... I>(std::index_sequence<I...> /*seq*/)
           GREX_ALWAYS_INLINE -> decltype(auto) {
             return f.template operator()<I...>();
           }(std::make_index_sequence<N>{});
}
template<std::size_t Begin, std::size_t End>
GREX_ALWAYS_INLINE constexpr decltype(auto) static_apply(auto f) {
  return [&]<std::size_t... I>(std::index_sequence<I...> /*seq*/)
           GREX_ALWAYS_INLINE -> decltype(auto) {
             return f.template operator()<Begin + I...>();
           }(std::make_index_sequence<End - Begin>{});
}
} // namespace grex

#endif // INCLUDE_GREX_BASE_HPP
