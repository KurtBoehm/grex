// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_NEON_OPERATIONS_LOAD_HPP
#define INCLUDE_GREX_BACKEND_NEON_OPERATIONS_LOAD_HPP

#include <arm_neon.h>

#include "grex/backend/neon/macros/types.hpp"
#include "grex/backend/neon/operations/insert.hpp"
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

#define GREX_PARTLOAD_64(KIND) \
  if (size >= 1) [[likely]] { \
    if (size >= 2) [[unlikely]] { \
      return {.r = GREX_ISUFFIXED(vld1q, KIND, 64)(ptr)}; \
    } \
    const auto lo = GREX_ISUFFIXED(vld1, KIND, 64)(ptr); \
    return {.r = GREX_ISUFFIXED(vcombine, KIND, 64)(lo, GREX_ISUFFIXED(vdup_n, KIND, 64)(0))}; \
  } \
  return {.r = GREX_ISUFFIXED(vdupq_n, KIND, 64)(0)};
#define GREX_PARTLOAD_32(KIND) \
  const auto zero64 = GREX_ISUFFIXED(vdup_n, KIND, 32)(0); \
  if (size >= 2) { \
    if (size >= 4) [[unlikely]] { \
      return {.r = GREX_ISUFFIXED(vld1q, KIND, 32)(ptr)}; \
    } \
    const auto lo = GREX_ISUFFIXED(vld1, KIND, 32)(ptr); \
    if (size == 3) { \
      const auto hi = GREX_ISUFFIXED(vset_lane, KIND, 32)(ptr[2], zero64, 0); \
      return {.r = GREX_ISUFFIXED(vcombine, KIND, 32)(lo, hi)}; \
    } \
    return {.r = GREX_ISUFFIXED(vcombine, KIND, 32)(lo, zero64)}; \
  } \
  if (size == 1) [[likely]] { \
    const auto zero = GREX_ISUFFIXED(vdupq_n, KIND, 32)(0); \
    return {.r = GREX_ISUFFIXED(vsetq_lane, KIND, 32)(ptr[0], zero, 0)}; \
  } \
  return {.r = GREX_ISUFFIXED(vdupq_n, KIND, 32)(0)};
#define GREX_PARTLOAD_16(KIND) \
  const std::size_t size2 = size / 2; \
  const auto* ptr32 = reinterpret_cast<const KIND##32 *>(ptr); \
  const auto out32 = load_part(ptr32, size2, type_tag<Vector<KIND##32, 4>>).r; \
  auto out = GREX_CAT(vreinterpretq_, GREX_ISUFFIX(KIND, 16), _, GREX_ISUFFIX(KIND, 32))(out32); \
  if ((size & 1U) != 0) { \
    const std::size_t i = size - 1; \
    out = insert(Vector<KIND##16, 8>{out}, i, ptr[i]).r; \
  } \
  return {.r = out};
#define GREX_PARTLOAD_8(KIND) \
  const std::size_t size2 = size / 2; \
  const auto* ptr16 = reinterpret_cast<const KIND##16 *>(ptr); \
  const auto out16 = load_part(ptr16, size2, type_tag<Vector<KIND##16, 8>>).r; \
  auto out = GREX_CAT(vreinterpretq_, GREX_ISUFFIX(KIND, 8), _, GREX_ISUFFIX(KIND, 16))(out16); \
  if ((size & 1U) != 0) { \
    const std::size_t i = size - 1; \
    out = insert(Vector<KIND##8, 16>{out}, i, ptr[i]).r; \
  } \
  return {.r = out};

#define GREX_PARTLOAD(KIND, BITS, SIZE) \
  inline Vector<KIND##BITS, SIZE> load_part(const KIND##BITS* ptr, std::size_t size, \
                                            TypeTag<Vector<KIND##BITS, SIZE>>) { \
    GREX_PARTLOAD_##BITS(KIND) \
  }
GREX_FOREACH_TYPE(GREX_PARTLOAD, 128)
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_LOAD_HPP
