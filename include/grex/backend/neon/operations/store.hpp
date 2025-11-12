// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_NEON_OPERATIONS_STORE_HPP
#define INCLUDE_GREX_BACKEND_NEON_OPERATIONS_STORE_HPP

#include <array>

#include <arm_neon.h>

#include "grex/backend/macros/base.hpp"
#include "grex/backend/neon/macros/types.hpp"
#include "grex/backend/neon/types.hpp"

namespace grex::backend {
#define GREX_STORE(KIND, BITS, SIZE) \
  inline void store(KIND##BITS* dst, Vector<KIND##BITS, SIZE> src) { \
    GREX_CAT(vst1q_, GREX_ISUFFIX(KIND, BITS))(dst, src.r); \
  } \
  /* This is not actually aligned, but who cares */ \
  inline void store_aligned(KIND##BITS* dst, Vector<KIND##BITS, SIZE> src) { \
    GREX_CAT(vst1q_, GREX_ISUFFIX(KIND, BITS))(dst, src.r); \
  }
GREX_FOREACH_TYPE(GREX_STORE, 128)

#define GREX_PARTSTORE_64(KIND) \
  switch (size) { \
    [[unlikely]] case 0: \
      return; \
    case 1: \
      return GREX_ISUFFIXED(vst1, KIND, 64)(dst, GREX_ISUFFIXED(vget_low, KIND, 64)(src.r)); \
    [[unlikely]] case 2: \
      return GREX_ISUFFIXED(vst1q, KIND, 64)(dst, src.r); \
    default: std::unreachable(); \
  }
#define GREX_PARTSTORE_32(KIND) \
  switch (size) { \
    [[unlikely]] case 0: \
      return; \
    case 1: return GREX_ISUFFIXED(vst1q_lane, KIND, 32)(dst, src.r, 0); \
    case 2: [[fallthrough]]; \
    case 3: \
      GREX_ISUFFIXED(vst1, KIND, 32)(dst, GREX_ISUFFIXED(vget_low, KIND, 32)(src.r)); \
      if (size == 3) { \
        GREX_ISUFFIXED(vst1q_lane, KIND, 32)(dst + 2, src.r, 2); \
      } \
      return; \
    [[unlikely]] case 4: \
      return GREX_ISUFFIXED(vst1q, KIND, 32)(dst, src.r); \
    default: std::unreachable(); \
  }
#define GREX_PARTSTORE_16(KIND) \
  const std::size_t size2 = size / 2; \
  store_part(reinterpret_cast<KIND##32 *>(dst), Vector<KIND##32, 4>{.r = src.r}, size2); \
  if ((size & 1U) != 0) { \
    switch (size2) { \
      case 0: GREX_ISUFFIXED(vst1q_lane, KIND, 16)(dst + 0, src.r, 0); return; \
      case 1: GREX_ISUFFIXED(vst1q_lane, KIND, 16)(dst + 2, src.r, 2); return; \
      case 2: GREX_ISUFFIXED(vst1q_lane, KIND, 16)(dst + 4, src.r, 4); return; \
      case 3: GREX_ISUFFIXED(vst1q_lane, KIND, 16)(dst + 6, src.r, 6); return; \
      default: break; \
    } \
  }
#define GREX_PARTSTORE_8(KIND) \
  if (size >= 16) [[unlikely]] { \
    store(dst, src); \
    return; \
  } \
  if (size == 0) [[unlikely]] { \
    return; \
  } \
  std::array<KIND##8, 16> buf{}; \
  store(buf.data(), src); \
  std::size_t j = 0; \
  if ((size & 8U) != 0) { \
    reinterpret_cast<u64*>(dst)[0] = reinterpret_cast<u64*>(buf.data())[0]; \
    j += 8; \
  } \
  if ((size & 4U) != 0) { \
    reinterpret_cast<u32*>(dst)[j / 4] = reinterpret_cast<u32*>(buf.data())[j / 4]; \
    j += 4; \
  } \
  if ((size & 2U) != 0) { \
    reinterpret_cast<u16*>(dst)[j / 2] = reinterpret_cast<u16*>(buf.data())[j / 2]; \
    j += 2; \
  } \
  if ((size & 1U) != 0) { \
    dst[j] = buf[j]; \
  }

#define GREX_PARTSTORE(KIND, BITS, SIZE) \
  inline void store_part(KIND##BITS* dst, Vector<KIND##BITS, SIZE> src, std::size_t size) { \
    GREX_PARTSTORE_##BITS(KIND) \
  }
GREX_FOREACH_TYPE(GREX_PARTSTORE, 128)
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_STORE_HPP
