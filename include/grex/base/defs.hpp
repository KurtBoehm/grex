// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BASE_DEFS_HPP
#define INCLUDE_GREX_BASE_DEFS_HPP

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <utility>

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
} // namespace primitives
using namespace primitives;

template<typename T>
concept UnsignedIntVectorizable =
  std::same_as<T, u8> || std::same_as<T, u16> || std::same_as<T, u32> || std::same_as<T, u64>;
template<typename T>
concept SignedIntVectorizable =
  std::same_as<T, i8> || std::same_as<T, i16> || std::same_as<T, i32> || std::same_as<T, i64>;
template<typename T>
concept FloatVectorizable = std::same_as<T, f32> || std::same_as<T, f64>;

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
    using Unsigned = U; \
    using Signed = S; \
  }; \
  template<> \
  struct SignednessTrait<S> { \
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

template<std::size_t tBytes>
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
template<std::size_t tBytes>
using UnsignedInt = SizedIntegerTrait<tBytes>::Unsigned;
template<std::size_t tBytes>
using SignedInt = SizedIntegerTrait<tBytes>::Signed;
template<FloatVectorizable T>
using FloatSize = UnsignedInt<sizeof(T)>;

enum struct ShuffleIndex : u8 { any = 254, zero = 255 };
inline constexpr ShuffleIndex any_sh = ShuffleIndex::any;
inline constexpr ShuffleIndex zero_sh = ShuffleIndex::zero;
constexpr bool is_index(ShuffleIndex sh) {
  return u8(sh) < u8(any_sh);
}

enum struct BlendZero : u8 { zero = 0, keep = 1, any = 2 };
inline constexpr BlendZero zero_bz = BlendZero::zero;
inline constexpr BlendZero keep_bz = BlendZero::keep;
inline constexpr BlendZero any_bz = BlendZero::any;

namespace literals {
consteval ShuffleIndex operator""_sh(unsigned long long int v) {
  if (v < 254) {
    return ShuffleIndex(v);
  }
  throw std::invalid_argument{"Unsupported value!"};
}
} // namespace literals

template<typename TIt>
concept MultiByteIterator = requires(TIt it) {
  typename TIt::Container;
  { TIt::Container::element_bytes } -> std::convertible_to<std::size_t>;
  { it.raw() } -> std::convertible_to<const std::byte*>;
};

template<std::size_t tIdx, typename T>
using IdxType = T;

template<typename T>
struct TypeTag {};
template<typename T>
inline constexpr TypeTag<T> type_tag{};

template<typename... T>
struct TypeSeq {
  template<typename... TOther>
  using Prepended = TypeSeq<TOther..., T...>;
};

template<typename T, T tVal>
struct ValueTag {
  using Value = T;
  static constexpr T value = tVal;
  constexpr operator T() const { // NOLINT
    return value;
  }
};
template<auto tValue>
using AutoTag = ValueTag<std::decay_t<decltype(tValue)>, tValue>;
template<std::size_t tValue>
using IndexTag = AutoTag<tValue>;
template<bool tValue>
using BoolTag = AutoTag<tValue>;

template<typename T, T tValue>
inline constexpr ValueTag<T, tValue> value_tag{};
template<auto tValue>
inline constexpr AutoTag<tValue> auto_tag{};
template<std::size_t tValue>
inline constexpr IndexTag<tValue> index_tag{};
template<bool tValue>
inline constexpr BoolTag<tValue> bool_tag{};
inline constexpr BoolTag<true> true_tag{};
inline constexpr BoolTag<false> false_tag{};

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
concept AnyBoolTag = TypedValueTag<TTag, bool>;

enum struct IterDirection : bool { forward, backward };
inline std::string_view format_as(IterDirection dir) {
  switch (dir) {
    case IterDirection::forward: return "forward";
    case IterDirection::backward: return "backward";
  }
}

#if defined(__GNUC__)
#define GREX_ALWAYS_INLINE __attribute__((always_inline))
#else
#define GREX_ALWAYS_INLINE
#endif

template<std::size_t tSize>
GREX_ALWAYS_INLINE constexpr decltype(auto) static_apply(auto f) {
  return [&]<std::size_t... tIdxs>(std::index_sequence<tIdxs...> /*seq*/)
           GREX_ALWAYS_INLINE -> decltype(auto) {
             return f.template operator()<tIdxs...>();
           }(std::make_index_sequence<tSize>{});
}
template<std::size_t tBegin, std::size_t tEnd>
GREX_ALWAYS_INLINE constexpr decltype(auto) static_apply(auto f) {
  return [&]<std::size_t... tIdxs>(std::index_sequence<tIdxs...> /*seq*/)
           GREX_ALWAYS_INLINE -> decltype(auto) {
             return f.template operator()<tBegin + tIdxs...>();
           }(std::make_index_sequence<tEnd - tBegin>{});
}
} // namespace grex

#endif // INCLUDE_GREX_BASE_DEFS_HPP
