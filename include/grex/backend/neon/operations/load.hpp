// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_NEON_OPERATIONS_LOAD_HPP
#define INCLUDE_GREX_BACKEND_NEON_OPERATIONS_LOAD_HPP

#include <cassert>
#include <cstddef>
#include <cstring>

#include <arm_neon.h>

#include "grex/backend/defs.hpp" // IWYU pragma: keep
#include "grex/backend/macros/for-each.hpp"
#include "grex/backend/neon/macros/types.hpp"
#include "grex/backend/neon/operations/expand.hpp"
#include "grex/backend/neon/operations/reinterpret.hpp"
#include "grex/backend/neon/types.hpp"
#include "grex/base.hpp"

namespace grex::backend {
#define GREX_LOAD(KIND, BITS, SIZE) \
  GREX_ALWAYS_INLINE inline Vector<KIND##BITS, SIZE> load(const KIND##BITS* src, \
                                                          TypeTag<Vector<KIND##BITS, SIZE>>) { \
    return {.r = GREX_ISUFFIXED(vld1q, KIND, BITS)(src)}; \
  } \
  /* This is not actually aligned, but who cares */ \
  GREX_ALWAYS_INLINE inline Vector<KIND##BITS, SIZE> load_aligned( \
    const KIND##BITS* src, TypeTag<Vector<KIND##BITS, SIZE>>) { \
    return {.r = GREX_ISUFFIXED(vld1q, KIND, BITS)(src)}; \
  }
GREX_FOREACH_TYPE(GREX_LOAD, 128)

// 8-bit load
template<typename T>
requires(sizeof(T) == 1)
GREX_ALWAYS_INLINE inline Vector<T, 16> load_first(const T* data, IndexTag<1> /*bytes*/) {
  using Vec = Vector<T, 16>;
  if (__builtin_constant_p(data[0])) {
    auto out = zeros(type_tag<Vec>);
    std::memcpy(&out, data, 1);
    return out;
  }
  Vec out{};
  asm("ldr %b0, %1" : "=w"(out) : "m"(*data) : "memory"); // NOLINT
  return out;
}
// 16-bit load
template<typename T>
requires(sizeof(T) <= 2)
GREX_ALWAYS_INLINE inline Vector<T, 16 / sizeof(T)> load_first(const T* data,
                                                               IndexTag<2> /*bytes*/) {
  using Vec = Vector<T, 16 / sizeof(T)>;
  if (static_apply<Vec::size>(
        [&]<std::size_t... tI>() { return (... && __builtin_constant_p(data[tI])); })) {
    auto out = zeros(type_tag<Vec>);
    std::memcpy(&out, data, 2);
    return out;
  }
  Vec out{};
  asm("ldr %h0, %1" : "=w"(out) : "m"(*data) : "memory"); // NOLINT
  return out;
}
// 32-bit load
template<typename T>
requires(sizeof(T) <= 4)
GREX_ALWAYS_INLINE inline Vector<T, 16 / sizeof(T)> load_first(const T* data,
                                                               IndexTag<4> /*bytes*/) {
  using Vec = Vector<T, 16 / sizeof(T)>;
  if (static_apply<Vec::size>(
        [&]<std::size_t... tI>() { return (... && __builtin_constant_p(data[tI])); })) {
    auto out = zeros(type_tag<Vec>);
    std::memcpy(&out, data, 4);
    return out;
  }
  Vec out{};
  asm("ldr %s0, %1" : "=w"(out) : "m"(*data) : "memory"); // NOLINT
  return out;
}
// 64-bit load
template<typename T>
GREX_ALWAYS_INLINE inline Vector<T, 16 / sizeof(T)> load_first(const T* data,
                                                               IndexTag<8> /*bytes*/) {
  uint64x1_t out;
  std::memcpy(&out, data, 8);
  return {.r = as<T>(expand64(out))};
}

template<std::size_t tBytes, typename T>
GREX_ALWAYS_INLINE inline Vector<T, 16 / sizeof(T)> load_first(const T* data) {
  return load_first(data, index_tag<tBytes>);
}

template<AnyVector TVec, std::size_t tSize>
requires((AnyNativeVector<TVec> || AnySubNativeVector<TVec>) && tSize <= TVec::size)
GREX_ALWAYS_INLINE inline TVec load_part(const typename TVec::Value* ptr, IndexTag<tSize> /*size*/,
                                         TypeTag<TVec> /*tag*/) {
  using Value = TVec::Value;
  using FullVec = Vector<Value, 16 / sizeof(Value)>;
  constexpr std::size_t bytes = tSize * sizeof(Value);

  // Simple cases: 16 and 0
  if constexpr (bytes == 16) {
    return load(ptr, type_tag<FullVec>);
  } else if constexpr (bytes == 0) {
    return undefined(type_tag<TVec>);
  }

  uint8x16_t out;
  if constexpr ((bytes & 8U) != 0) {
    out = as<u8>(load_first<8>(ptr).r);
  }
  if constexpr ((bytes & 4U) != 0) {
    constexpr std::size_t offset = bytes & 8U;
    if constexpr (offset == 0) {
      out = as<u8>(load_first<4>(ptr).r);
    } else {
      asm("ld1.s { %0 }[%1], %2" // NOLINT
          : "+w"(out)
          : "i"(offset / 4), "Q"(ptr[offset / sizeof(Value)])
          : "memory");
    }
  }
  if constexpr ((bytes & 2U) != 0) {
    constexpr std::size_t offset = bytes & 12U;
    if constexpr (offset == 0) {
      out = as<u8>(load_first<2>(ptr).r);
    } else {
      asm("ld1.h { %0 }[%1], %2" // NOLINT
          : "+w"(out)
          : "i"(offset / 2), "Q"(ptr[offset / sizeof(Value)])
          : "memory");
    }
  }
  if constexpr ((bytes & 1U) != 0) {
    constexpr std::size_t offset = bytes & 14U;
    if constexpr (offset == 0) {
      out = as<u8>(load_first<1>(ptr).r);
    } else {
      out = vld1q_lane_u8(reinterpret_cast<const u8*>(ptr) + offset, out, offset);
    }
  }
  return TVec{as<Value>(out)};
}

template<Vectorizable T, std::size_t tPart, std::size_t tSize>
GREX_ALWAYS_INLINE inline SubVector<T, tPart, tSize>
load(const T* src, TypeTag<SubVector<T, tPart, tSize>> /*tag*/) {
  using Dst = SubVector<T, tPart, tSize>;
  return Dst{load_part(src, index_tag<tPart>, type_tag<Dst>)};
}
template<Vectorizable T, std::size_t tPart, std::size_t tSize>
GREX_ALWAYS_INLINE inline SubVector<T, tPart, tSize>
load_aligned(const T* src, TypeTag<SubVector<T, tPart, tSize>> /*tag*/) {
  using Dst = SubVector<T, tPart, tSize>;
  return Dst{load_part(src, index_tag<tPart>, type_tag<Dst>)};
}

template<AnyVector TVec>
requires(AnyNativeVector<TVec> || AnySubNativeVector<TVec>)
GREX_ALWAYS_INLINE inline TVec load_part(const typename TVec::Value* ptr, std::size_t size,
                                         TypeTag<TVec> tag) {
  using Value = TVec::Value;
  constexpr std::size_t bytes = sizeof(Value) * TVec::size;

  if (__builtin_constant_p(size)) {
    auto result = undefined(tag);
    bool matched = false;
    grex::static_apply<TVec::size>([&]<std::size_t... tI>() {
      matched =
        (((tI == size) ? (result = load_part(ptr, index_tag<tI>, tag), true) : false) || ...);
    });
    return matched ? result : load(ptr, tag);
  }

  if (size >= TVec::size) [[unlikely]] {
    return load(ptr, tag);
  }
  auto out = undefined(tag).registr();
  if constexpr (sizeof(Value) == 1) {
    if ((size & (1U / sizeof(Value))) != 0) {
      constexpr std::size_t f = 2 / sizeof(Value);
      out = load_first<1>(ptr + (size / f * f)).r;
    }
  }
  if constexpr (sizeof(Value) <= 2 && bytes > 2) {
    if ((size & (2U / sizeof(Value))) != 0) {
      constexpr std::size_t f = 4 / sizeof(Value);
      const auto lo = load_first<2>(ptr + (size / f * f)).r;
      out = as<Value>(vzip1q_u16(as<u16>(lo), as<u16>(out)));
    }
  }
  if constexpr (sizeof(Value) <= 4 && bytes > 4) {
    if ((size & (4U / sizeof(Value))) != 0) {
      constexpr std::size_t f = 8 / sizeof(Value);
      const auto lo = load_first<4>(ptr + (size / f * f)).r;
      out = as<Value>(vzip1q_u32(as<u32>(lo), as<u32>(out)));
    }
  }
  if ((size & (8U / sizeof(Value))) != 0 && bytes > 8) {
    const auto lo = load_first<8>(ptr).r;
    out = as<Value>(vzip1q_u64(as<u64>(lo), as<u64>(out)));
  }
  return TVec{out};
}
} // namespace grex::backend

#include "grex/backend/shared/operations/load.hpp" // IWYU pragma: export

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_LOAD_HPP
