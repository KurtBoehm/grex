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
inline uint8x16_t part8(const u8* ptr, AnyIndexTag auto bytes, uint8x16_t base,
                        AnyIndexTag auto offset) {
  if constexpr (bytes.value >= 1) {
    return vld1q_lane_u8(ptr, base, offset.value);
  } else {
    return base;
  }
}
inline uint8x16_t part16(const u8* ptr, AnyIndexTag auto bytes, uint8x16_t base,
                         AnyIndexTag auto offset) {
  if constexpr (bytes.value >= 2) {
    const auto base16 = vreinterpretq_u16_u8(base);
    const auto ins16 = vld1q_lane_u16(ptr, base16, offset.value / 2);
    const auto ins8 = vreinterpretq_u8_u16(ins16);
    return part8(ptr + 2, index_tag<bytes.value - 2>, ins8, index_tag<offset.value + 2>);
  } else {
    return part8(ptr, bytes, base, offset);
  }
}
inline uint8x16_t part32(const u8* ptr, AnyIndexTag auto bytes, uint8x16_t base,
                         AnyIndexTag auto offset) {
  if constexpr (bytes.value >= 4) {
    const auto base32 = vreinterpretq_u32_u8(base);
    const auto ins32 = vld1q_lane_u32(ptr, base32, offset.value / 4);
    const auto ins8 = vreinterpretq_u8_u32(ins32);
    return part16(ptr + 4, index_tag<bytes.value - 4>, ins8, index_tag<offset.value + 4>);
  } else {
    return part16(ptr, bytes, base, offset);
  }
}
inline uint8x16_t part64(const void* ptr, AnyIndexTag auto bytes) {
  static_assert(bytes.value <= 16);
  const u8* ptr8 = reinterpret_cast<const u8*>(ptr);
  if constexpr (bytes.value == 16) {
    return vld1q_u8(ptr8);
  } else if constexpr (bytes.value >= 8) {
    return part32(ptr8 + 8, index_tag<bytes.value - 8>, expand64(vld1_u8(ptr8)), index_tag<8>);
  } else {
    return part32(ptr8, bytes, make_undefined<uint8x16_t>(), index_tag<0>);
  }
}
} // namespace ld1

#define GREX_SUBLOAD(KIND, BITS, PART, SIZE) \
  inline SubVector<KIND##BITS, PART, SIZE> load(const KIND##BITS* src, \
                                                TypeTag<SubVector<KIND##BITS, PART, SIZE>>) { \
    const auto vu8 = ld1::part64(src, index_tag<PART##UZ * BITS##UZ / CHAR_BIT>); \
    const auto v = reinterpret(vu8, type_tag<GREX_REGISTER(KIND, BITS, SIZE)>); \
    return SubVector<KIND##BITS, PART, SIZE>{v}; \
  } \
  inline SubVector<KIND##BITS, PART, SIZE> load_aligned( \
    const KIND##BITS* src, TypeTag<SubVector<KIND##BITS, PART, SIZE>>) { \
    const auto vu8 = ld1::part64(src, index_tag<PART##UZ * BITS##UZ / CHAR_BIT>); \
    const auto v = reinterpret(vu8, type_tag<GREX_REGISTER(KIND, BITS, SIZE)>); \
    return SubVector<KIND##BITS, PART, SIZE>{v}; \
  }
GREX_FOREACH_SUB(GREX_SUBLOAD)

#define GREX_PARTLOAD_ATTR_0
#define GREX_PARTLOAD_ATTR_1 [[unlikely]]
#define GREX_PARTLOAD_ATTR(INDEX, REF) GREX_CAT(GREX_PARTLOAD_ATTR_, GREX_EQUALS(INDEX, REF))

#define GREX_PARTLOAD_CASE(SIZE, INDEX, KIND, BITS) \
  GREX_PARTLOAD_ATTR(INDEX, 0) case INDEX: \
  return {.r = reinterpret(ld1::part64(ptr, index_tag<INDEX * BITS / CHAR_BIT>), \
                           type_tag<GREX_REGISTER(KIND, BITS, SIZE)>)};
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
  return Dst{reinterpret(ld1::part64(ptr, index_tag<INDEX * BITS / CHAR_BIT>), \
                         type_tag<GREX_REGISTER(KIND, BITS, SIZE)>)};
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
