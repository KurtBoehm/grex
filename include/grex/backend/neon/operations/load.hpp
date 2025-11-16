// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_NEON_OPERATIONS_LOAD_HPP
#define INCLUDE_GREX_BACKEND_NEON_OPERATIONS_LOAD_HPP

#include <cassert>

#include <arm_neon.h>

#include "grex/backend/macros/equals.hpp"
#include "grex/backend/neon/macros/types.hpp"
#include "grex/backend/neon/operations/expand.hpp"
#include "grex/backend/neon/operations/reinterpret.hpp"
#include "grex/backend/neon/types.hpp"

namespace grex::backend {
#define GREX_LOAD(KIND, BITS, SIZE) \
  inline Vector<KIND##BITS, SIZE> load(const KIND##BITS* src, TypeTag<Vector<KIND##BITS, SIZE>>) { \
    return {.r = GREX_ISUFFIXED(vld1q, KIND, BITS)(src)}; \
  } \
  /* This is not actually aligned, but who cares */ \
  inline Vector<KIND##BITS, SIZE> load_aligned(const KIND##BITS* src, \
                                               TypeTag<Vector<KIND##BITS, SIZE>>) { \
    return {.r = GREX_ISUFFIXED(vld1q, KIND, BITS)(src)}; \
  }
GREX_FOREACH_TYPE(GREX_LOAD, 128)

namespace ld1 {
template<typename T, std::size_t tSize>
inline Vector<T, tSize> part(const T* ptr, AnyIndexTag auto bytes) {
  static_assert(bytes.value <= 16);
  const u8* ptr8 = reinterpret_cast<const u8*>(ptr);

  // Simple cases: 16 and 0
  if constexpr (bytes.value == 16) {
    return load(ptr, type_tag<Vector<T, tSize>>);
  } else if constexpr (bytes.value == 0) {
    return undefined(type_tag<Vector<T, tSize>>);
  }

  uint8x16_t out;
  if constexpr ((bytes.value & 8U) != 0) {
    out = expand64(vld1_u8(ptr8));
  }
  if constexpr ((bytes.value & 4U) != 0) {
    constexpr std::size_t offset = (bytes.value & 8U) / 4U;
    if constexpr (offset == 0) {
      asm volatile("ldr %s0, %1" : "=w"(out) : "m"(*ptr)); // NOLINT
    } else {
      const u32* ptr32 = reinterpret_cast<const u32*>(ptr);
      const auto out32 = vreinterpretq_u32_u8(out);
      const auto ins32 = vld1q_lane_u32(ptr32 + offset, out32, offset);
      out = vreinterpretq_u8_u32(ins32);
    }
  }
  if constexpr ((bytes.value & 2U) != 0) {
    constexpr std::size_t offset = (bytes.value & 12U) / 2U;
    if constexpr (offset == 0) {
      asm volatile("ldr %h0, %1" : "=w"(out) : "m"(*ptr)); // NOLINT
    } else {
      const u16* ptr16 = reinterpret_cast<const u16*>(ptr);
      const auto out16 = vreinterpretq_u16_u8(out);
      const auto ins16 = vld1q_lane_u16(ptr16 + offset, out16, offset);
      out = vreinterpretq_u8_u16(ins16);
    }
  }
  if constexpr ((bytes.value & 1U) != 0) {
    constexpr std::size_t offset = bytes.value & 14U;
    if constexpr (offset == 0) {
      asm volatile("ldr %b0, %1" : "=w"(out) : "m"(*ptr)); // NOLINT
    } else {
      out = vld1q_lane_u8(ptr8 + offset, out, offset);
    }
  }
  return {.r = reinterpret<T>(out)};
}
} // namespace ld1

#define GREX_SUBLOAD(KIND, BITS, PART, SIZE) \
  inline SubVector<KIND##BITS, PART, SIZE> load(const KIND##BITS* src, \
                                                TypeTag<SubVector<KIND##BITS, PART, SIZE>>) { \
    const auto v = ld1::part<KIND##BITS, SIZE>(src, index_tag<PART##UZ * BITS##UZ / CHAR_BIT>); \
    return SubVector<KIND##BITS, PART, SIZE>{v}; \
  } \
  inline SubVector<KIND##BITS, PART, SIZE> load_aligned( \
    const KIND##BITS* src, TypeTag<SubVector<KIND##BITS, PART, SIZE>>) { \
    const auto v = ld1::part<KIND##BITS, SIZE>(src, index_tag<PART##UZ * BITS##UZ / CHAR_BIT>); \
    return SubVector<KIND##BITS, PART, SIZE>{v}; \
  }
GREX_FOREACH_SUB(GREX_SUBLOAD)

#define GREX_PARTLOAD_ATTR_0
#define GREX_PARTLOAD_ATTR_1 [[unlikely]]
#define GREX_PARTLOAD_ATTR(INDEX, REF) GREX_CAT(GREX_PARTLOAD_ATTR_, GREX_EQUALS(INDEX, REF))

#define GREX_PARTLOAD_CASE(SIZE, INDEX, KIND, BITS) \
  GREX_PARTLOAD_ATTR(INDEX, 0) case INDEX: \
  return ld1::part<KIND##BITS, SIZE>(ptr, index_tag<INDEX * BITS / CHAR_BIT>);
#define GREX_PARTLOAD(KIND, BITS, SIZE) \
  inline Vector<KIND##BITS, SIZE> load_part(const KIND##BITS* ptr, std::size_t size, \
                                            TypeTag<Vector<KIND##BITS, SIZE>>) { \
    switch (size) { \
      GREX_REPEAT(SIZE, GREX_PARTLOAD_CASE, KIND, BITS) \
      [[unlikely]] GREX_PARTLOAD_CASE(SIZE, SIZE, KIND, BITS) default : std::unreachable(); \
    } \
  }
GREX_FOREACH_TYPE(GREX_PARTLOAD, 128)

#define GREX_SUBPARTLOAD_CASE(PART, INDEX, KIND, BITS, SIZE) \
  GREX_PARTLOAD_ATTR(INDEX, 0) case INDEX: \
  return Dst{ld1::part<KIND##BITS, SIZE>(ptr, index_tag<INDEX * BITS / CHAR_BIT>)};
#define GREX_SUBPARTLOAD(KIND, BITS, PART, SIZE) \
  inline SubVector<KIND##BITS, PART, SIZE> load_part(const KIND##BITS* ptr, std::size_t size, \
                                                     TypeTag<SubVector<KIND##BITS, PART, SIZE>>) { \
    using Dst = SubVector<KIND##BITS, PART, SIZE>; \
    switch (size) { \
      GREX_REPEAT(PART, GREX_SUBPARTLOAD_CASE, KIND, BITS, SIZE) \
      [[unlikely]] GREX_SUBPARTLOAD_CASE(PART, PART, KIND, BITS, SIZE) default \
          : std::unreachable(); \
    } \
  }
GREX_FOREACH_SUB(GREX_SUBPARTLOAD)
} // namespace grex::backend

#include "grex/backend/shared/operations/load.hpp" // IWYU pragma: export

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_LOAD_HPP
