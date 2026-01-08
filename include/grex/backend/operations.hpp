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

#include "grex/backend/base.hpp"
#include "grex/base.hpp"

namespace grex::backend {
template<Vectorizable T>
inline Scalar<T> abs(Scalar<T> x) {
  return {.value = T(std::abs(x.value))};
}

template<Vectorizable T>
inline Scalar<T> min(Scalar<T> a, Scalar<T> b) {
  return {.value = std::min(a.value, b.value)};
}
template<Vectorizable T>
inline Scalar<T> max(Scalar<T> a, Scalar<T> b) {
  return {.value = std::max(a.value, b.value)};
}

inline bool logical_andnot(bool a, bool b) {
  return !a && b;
}

#define GREX_OPS_MASKARITH(NAME, OP) \
  template<Vectorizable T> \
  inline Scalar<T> NAME(bool mask, Scalar<T> a, Scalar<T> b) { \
    return {.value = (mask ? T(a.value OP b.value) : a.value)}; \
  }
GREX_OPS_MASKARITH(mask_add, +)
GREX_OPS_MASKARITH(mask_subtract, -)
GREX_OPS_MASKARITH(mask_multiply, *)
GREX_OPS_MASKARITH(mask_divide, /)
#undef GREX_OPS_MASKARITH

template<Vectorizable T>
inline Scalar<T> extract_single(Scalar<T> v) {
  return v;
}

template<Vectorizable T>
inline Scalar<T> blend_zero(bool selector, Scalar<T> v1) {
  return selector ? v1 : Scalar<T>{T{}};
}
template<Vectorizable T>
inline Scalar<T> blend(bool selector, Scalar<T> v0, Scalar<T> v1) {
  return selector ? v1 : v0;
}

template<FloatVectorizable T>
inline bool is_finite(Scalar<T> v) {
  return std::isfinite(v.value);
}

template<std::size_t tSrcBytes>
static UnsignedInt<std::bit_ceil(tSrcBytes)> load_multibyte(const std::byte* data,
                                                            IndexTag<tSrcBytes> /*tag*/) {
  static constexpr std::size_t dst_bytes = std::bit_ceil(tSrcBytes);
  static constexpr std::size_t overhead_bits = (dst_bytes - tSrcBytes) * CHAR_BIT;
  using Dst = UnsignedInt<dst_bytes>;
  static constexpr Dst mask = std::numeric_limits<Dst>::max() >> overhead_bits;

  Dst output;
  std::memcpy(&output, data, tSrcBytes);
  if constexpr (std::endian::native == std::endian::little) {
    return output & mask;
  }
  if constexpr (std::endian::native == std::endian::big) {
    return output >> overhead_bits;
  }
  return output;
}

template<std::size_t tIdx, typename THead, typename... TTail>
GREX_ALWAYS_INLINE inline THead pack_get(THead head, TTail... tail) {
  if constexpr (tIdx == 0) {
    return head;
  } else {
    return pack_get<tIdx - 1>(tail...);
  }
}

#define GREX_NARY(NAME, OP, SECOP) \
  template<typename THead, typename... TTail> \
  requires((... && std::same_as<THead, TTail>)) \
  GREX_ALWAYS_INLINE inline THead NAME(THead head, TTail... tail) { \
    constexpr std::size_t num = sizeof...(TTail) + 1; \
    if constexpr (num == 1) { \
      return head; \
    } else { \
      const auto rec0 = \
        [&]<std::size_t tOff, std::size_t... tI>(IndexTag<tOff>, std::index_sequence<tI...>) \
          GREX_ALWAYS_INLINE { return NAME(pack_get<tOff + tI>(head, tail...)...); }; \
      const auto rec1 = \
        [&]<std::size_t tOff, std::size_t... tI>(IndexTag<tOff>, std::index_sequence<tI...>) \
          GREX_ALWAYS_INLINE { return SECOP(pack_get<tOff + tI>(head, tail...)...); }; \
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
