// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_OPERATIONS_HPP
#define INCLUDE_GREX_BACKEND_OPERATIONS_HPP

#include <algorithm>
#include <bit>
#include <climits>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>

#include "grex/backend/defs.hpp"
#include "grex/base.hpp"

namespace grex::backend {
template<Vectorizable T>
inline T abs(T x) {
  return T(std::abs(x));
}

template<Vectorizable T>
inline T min(T a, T b) {
  return std::min(a, b);
}
template<Vectorizable T>
inline T max(T a, T b) {
  return std::max(a, b);
}

inline bool logical_andnot(bool a, bool b) {
  return !a && b;
}

#define GREX_OPS_MASKARITH(NAME, OP) \
  template<Vectorizable T> \
  inline T NAME(bool mask, T a, T b) { \
    return mask ? T(a OP b) : a; \
  }
GREX_OPS_MASKARITH(mask_add, +)
GREX_OPS_MASKARITH(mask_subtract, -)
GREX_OPS_MASKARITH(mask_multiply, *)
GREX_OPS_MASKARITH(mask_divide, /)
#undef GREX_OPS_MASKARITH

template<Vectorizable T>
inline T extract_single(T v) {
  return v;
}

template<Vectorizable T>
inline T blend_zero(bool selector, T v1) {
  return selector ? v1 : T{};
}
template<Vectorizable T>
inline T blend(bool selector, T v0, T v1) {
  return selector ? v1 : v0;
}

template<FloatVectorizable T>
inline bool is_finite(T v) {
  return std::isfinite(v);
}

/** Whether the binary16 fused operations are carried out with a single rounding. */
inline constexpr bool has_f16_fma = GREX_F16_NATIVE_ARITHMETIC;

// Scalar binary16 operations are implemented in terms of GCC/Clang built-ins if `_Float16` is
// supported. If not, they are emulated through bit operations on the underlying `u16`.

#if GREX_NATIVE_F16
template<std::same_as<f16> T>
inline T abs(T x) {
  return __builtin_fabsf16(x);
}
template<std::same_as<f16> T>
inline bool is_finite(T v) {
  return abs(v) < NumericTrait<f16>::infinity();
}
template<std::same_as<f16> T>
inline T make_finite(T v) {
  return is_finite(v) ? v : f16{};
}
#else
template<std::same_as<f16> T>
inline T abs(T x) {
  return f16_from_bits(u16(f16_bits(x) & 0x7FFFU));
}
template<std::same_as<f16> T>
inline bool is_finite(T v) {
  return (f16_bits(v) & 0x7C00U) != 0x7C00U;
}
template<std::same_as<f16> T>
inline T make_finite(T v) {
  return is_finite(v) ? v : f16{};
}
#endif

template<std::size_t SrcBytes>
static UnsignedInt<std::bit_ceil(SrcBytes)> load_multibyte(const std::byte* data,
                                                           IndexTag<SrcBytes> /*tag*/) {
  static constexpr std::size_t dst_bytes = std::bit_ceil(SrcBytes);
  static constexpr std::size_t overhead_bits = (dst_bytes - SrcBytes) * CHAR_BIT;
  using Dst = UnsignedInt<dst_bytes>;
  static constexpr Dst mask = std::numeric_limits<Dst>::max() >> overhead_bits;

  Dst output;
  std::memcpy(&output, data, SrcBytes);
  if constexpr (std::endian::native == std::endian::little) {
    return output & mask;
  }
  if constexpr (std::endian::native == std::endian::big) {
    return output >> overhead_bits;
  }
  return output;
}

template<std::size_t I, typename Head, typename... Tail>
GREX_ALWAYS_INLINE inline Head pack_get(Head head, Tail... tail) {
  if constexpr (I == 0) {
    return head;
  } else {
    return pack_get<I - 1>(tail...);
  }
}

#define GREX_NARY(NAME, OP, SECOP) \
  template<typename Head, typename... Tail> \
  requires((... && std::same_as<Head, Tail>)) \
  GREX_ALWAYS_INLINE inline Head NAME(Head head, Tail... tail) { \
    constexpr std::size_t num = sizeof...(Tail) + 1; \
    if constexpr (num == 1) { \
      return head; \
    } else { \
      const auto rec0 = \
        [&]<std::size_t Off, std::size_t... I>(IndexTag<Off>, std::index_sequence<I...>) \
          GREX_ALWAYS_INLINE { return NAME(pack_get<Off + I>(head, tail...)...); }; \
      const auto rec1 = \
        [&]<std::size_t Off, std::size_t... I>(IndexTag<Off>, std::index_sequence<I...>) \
          GREX_ALWAYS_INLINE { return SECOP(pack_get<Off + I>(head, tail...)...); }; \
\
      if constexpr (std::has_single_bit(num)) { \
        const auto s0 = rec0(index_tag<0>, std::make_index_sequence<num / 2>{}); \
        const auto s1 = rec1(index_tag<num / 2>, std::make_index_sequence<num / 2>{}); \
        return s0 OP s1; \
      } else { \
        constexpr std::size_t lower = std::bit_floor(num); \
        const auto s0 = rec0(index_tag<0>, std::make_index_sequence<lower>{}); \
        const auto s1 = rec1(index_tag<lower>, std::make_index_sequence<num - lower>{}); \
        return s0 OP s1; \
      } \
    } \
  }
GREX_NARY(nary_add, +, nary_add)
GREX_NARY(nary_subtract, -, nary_add)
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_OPERATIONS_HPP
