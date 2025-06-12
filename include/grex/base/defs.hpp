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
#include <utility>

namespace grex {
using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using i8 = std::int8_t;
using i16 = std::int16_t;
using i32 = std::int32_t;
using i64 = std::int64_t;

#if defined(__GNUC__)
#define GREX_ALWAYS_INLINE __attribute__((always_inline))
#else
#define GREX_ALWAYS_INLINE
#endif

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

using f32 = float;
static_assert(std::numeric_limits<f32>::is_iec559 && sizeof(f32) == 4);
using f64 = double;
static_assert(std::numeric_limits<f64>::is_iec559 && sizeof(f64) == 8);

template<typename T>
concept Vectorizable =
  std::same_as<T, u8> || std::same_as<T, i8> || std::same_as<T, u16> || std::same_as<T, i16> ||
  std::same_as<T, u32> || std::same_as<T, i32> || std::same_as<T, u64> || std::same_as<T, i64> ||
  std::same_as<T, f32> || std::same_as<T, f64>;

template<typename T>
struct TypeTag {};
template<typename T>
inline constexpr TypeTag<T> type_tag{};

template<std::size_t tVal>
struct IndexTag {
  static constexpr std::size_t value = tVal;
  constexpr operator std::size_t() const {
    return value;
  }
};
template<std::size_t tVal>
inline constexpr IndexTag<tVal> index_tag;
template<typename T>
concept AnyIndexTag = requires {
  { T::value } -> std::convertible_to<std::size_t>;
};

template<std::size_t tSize>
constexpr decltype(auto) static_apply(auto f) {
  return [&]<std::size_t... tIdxs>(std::index_sequence<tIdxs...> /*seq*/) -> decltype(auto) {
    return f.template operator()<tIdxs...>();
  }(std::make_index_sequence<tSize>{});
}
template<std::size_t tBegin, std::size_t tEnd>
constexpr decltype(auto) static_apply(auto f) {
  return [&]<std::size_t... tIdxs>(std::index_sequence<tIdxs...> /*seq*/) -> decltype(auto) {
    return f.template operator()<tBegin + tIdxs...>();
  }(std::make_index_sequence<tEnd - tBegin>{});
}
} // namespace grex

#endif // INCLUDE_GREX_BASE_DEFS_HPP
