// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_NEON_OPERATIONS_STORE_HPP
#define INCLUDE_GREX_BACKEND_NEON_OPERATIONS_STORE_HPP

#include <climits>

#include <arm_neon.h>

#include "grex/backend/macros/equals.hpp"
#include "grex/backend/macros/repeat.hpp"
#include "grex/backend/neon/macros/types.hpp"
#include "grex/backend/neon/operations/reinterpret.hpp"
#include "grex/backend/neon/types.hpp"
#include "grex/base/defs.hpp"

namespace grex::backend {
#define GREX_STORE(KIND, BITS, SIZE) \
  inline void store(KIND##BITS* dst, Vector<KIND##BITS, SIZE> src) { \
    GREX_ISUFFIXED(vst1q, KIND, BITS)(dst, src.r); \
  } \
  /* This is not actually aligned, but who cares */ \
  inline void store_aligned(KIND##BITS* dst, Vector<KIND##BITS, SIZE> src) { \
    GREX_ISUFFIXED(vst1q, KIND, BITS)(dst, src.r); \
  }
GREX_FOREACH_TYPE(GREX_STORE, 128)

namespace st1 {
inline void part(void* dst, uint8x16_t src, AnyIndexTag auto bytes) {
  static_assert(bytes.value <= 16);
  u8* dst8 = reinterpret_cast<u8*>(dst);

  // Simple cases: 16 and 0
  if constexpr (bytes.value == 16) {
    vst1q_u8(dst8, src);
    return;
  } else if constexpr (bytes.value == 0) {
    return;
  }

  if constexpr ((bytes.value & 8U) != 0) {
    vst1_u8(dst8, vget_low_u8(src));
  }
  if constexpr ((bytes.value & 4U) != 0) {
    constexpr std::size_t offset = (bytes.value & 8U) / 4U;
    u32* dst32 = reinterpret_cast<u32*>(dst) + offset;
    vst1q_lane_u32(dst32, vreinterpretq_u32_u8(src), offset);
  }
  if constexpr ((bytes.value & 2U) != 0) {
    constexpr std::size_t offset = (bytes.value & 12U) / 2U;
    u16* dst16 = reinterpret_cast<u16*>(dst) + offset;
// Sadly, GCC gobbles 2-byte writes sometimes when using ARM64 intrinsics
// Therefore, inline assembly is used in the case of GCC
#if GREX_GCC
    if constexpr (bytes.value == 2) {
      asm volatile("str %h1, %0" : "=m"(*dst16) : "w"(src) :);
    } else {
      asm volatile("st1.h { %1 }[%2], %0" : "=Q"(*dst16) : "w"(src), "i"(offset) : "memory");
    }
#elif GREX_CLANG
    vst1q_lane_u16(dst16, vreinterpretq_u16_u8(src), offset);
#endif
  }
  if constexpr ((bytes.value & 1U) != 0) {
    constexpr std::size_t offset = bytes.value & 14U;
    vst1q_lane_u8(dst8 + offset, src, offset);
  }
}
} // namespace st1

#define GREX_SUBSTORE(KIND, BITS, PART, SIZE) \
  inline void store(KIND##BITS* dst, SubVector<KIND##BITS, PART, SIZE> src) { \
    const auto src8 = reinterpret<u8>(src.registr()); \
    st1::part(dst, src8, index_tag<PART##UZ * BITS##UZ / CHAR_BIT>); \
  } \
  inline void store_aligned(KIND##BITS* dst, SubVector<KIND##BITS, PART, SIZE> src) { \
    const auto src8 = reinterpret<u8>(src.registr()); \
    st1::part(dst, src8, index_tag<PART##UZ * BITS##UZ / CHAR_BIT>); \
  }
GREX_FOREACH_SUB(GREX_SUBSTORE)

#define GREX_PARTSTORE_ATTR_0
#define GREX_PARTSTORE_ATTR_1 [[unlikely]]
#define GREX_PARTSTORE_ATTR(INDEX, REF) GREX_CAT(GREX_PARTSTORE_ATTR_, GREX_EQUALS(INDEX, REF))

#define GREX_PARTSTORE_CASE(SIZE, INDEX, KIND, BITS) \
  GREX_PARTSTORE_ATTR(INDEX, 0) case INDEX: { \
    st1::part(dst, src8, index_tag<INDEX * BITS / CHAR_BIT>); \
    return; \
  }
#define GREX_PARTSTORE(KIND, BITS, SIZE) \
  inline void store_part(KIND##BITS* dst, Vector<KIND##BITS, SIZE> src, std::size_t size) { \
    const auto src8 = reinterpret<u8>(src.r); \
    switch (size) { \
      GREX_REPEAT(SIZE, GREX_PARTSTORE_CASE, KIND, BITS) \
      [[unlikely]] GREX_PARTSTORE_CASE(SIZE, SIZE, KIND, BITS) default : std::unreachable(); \
    } \
  }
GREX_FOREACH_TYPE(GREX_PARTSTORE, 128)

#define GREX_SUBPARTSTORE(KIND, BITS, PART, SIZE) \
  inline void store_part(KIND##BITS* dst, SubVector<KIND##BITS, PART, SIZE> src, \
                         std::size_t size) { \
    const auto src8 = reinterpret<u8>(src.registr()); \
    switch (size) { \
      GREX_REPEAT(PART, GREX_PARTSTORE_CASE, KIND, BITS) \
      [[unlikely]] GREX_PARTSTORE_CASE(PART, PART, KIND, BITS) default : std::unreachable(); \
    } \
  }
GREX_FOREACH_SUB(GREX_SUBPARTSTORE)
} // namespace grex::backend

#include "grex/backend/shared/operations/store.hpp" // IWYU pragma: export

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_STORE_HPP
