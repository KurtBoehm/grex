// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_NEON_OPERATIONS_STORE_HPP
#define INCLUDE_GREX_BACKEND_NEON_OPERATIONS_STORE_HPP

#include <cassert>
#include <cstddef>
#include <cstring>

#include <arm_neon.h>

#include "grex/backend/defs.hpp" // IWYU pragma: keep
#include "grex/backend/macros/base.hpp"
#include "grex/backend/macros/equals.hpp"
#include "grex/backend/macros/for-each.hpp"
#include "grex/backend/macros/repeat.hpp"
#include "grex/backend/neon/macros/types.hpp"
#include "grex/backend/neon/operations/reinterpret.hpp"
#include "grex/backend/neon/types.hpp"
#include "grex/base.hpp"

namespace grex::backend {
#define GREX_STORE(KIND, BITS, SIZE) \
  GREX_ALWAYS_INLINE inline void store(KIND##BITS* dst, Vector<KIND##BITS, SIZE> src) { \
    GREX_ISUFFIXED(vst1q, KIND, BITS)(dst, src.r); \
  } \
  /* This is not actually aligned, but who cares */ \
  GREX_ALWAYS_INLINE inline void store_aligned(KIND##BITS* dst, Vector<KIND##BITS, SIZE> src) { \
    GREX_ISUFFIXED(vst1q, KIND, BITS)(dst, src.r); \
  }
GREX_FOREACH_TYPE(GREX_STORE, 128)

template<std::size_t tBytes, typename T>
GREX_ALWAYS_INLINE inline void store_first(T* dst, Vector<T, 16 / sizeof(T)> src) {
  std::memcpy(dst, &src.r, tBytes);
}
template<std::size_t tBytes, typename T, std::size_t tPart, std::size_t tSize>
requires(tBytes <= sizeof(T) * tPart)
GREX_ALWAYS_INLINE inline void store_first(T* dst, SubVector<T, tPart, tSize> src) {
  std::memcpy(dst, &src.full.r, tBytes);
}

template<AnyVector TVec, std::size_t tSize>
requires((AnyNativeVector<TVec> || AnySubNativeVector<TVec>) && tSize <= TVec::size)
GREX_ALWAYS_INLINE inline void store_part(typename TVec::Value* dst, TVec src,
                                          IndexTag<tSize> /*size*/) {
  using Value = TVec::Value;
  constexpr std::size_t bytes = tSize * sizeof(Value);

  // Simple cases: 16 and 0
  if constexpr (bytes == 16) {
    store(dst, src);
    return;
  } else if constexpr (bytes == 0) {
    return;
  }

  if constexpr ((bytes & 8U) != 0) {
    store_first<8>(dst, src);
  }
  if constexpr ((bytes & 4U) != 0) {
    constexpr std::size_t offset = bytes & 8U;
    if constexpr (offset == 0) {
      store_first<4>(dst, src);
    } else {
      asm("st1.s { %1 }[%2], %0" // NOLINT
          : "=Q"(dst[offset / sizeof(Value)])
          : "w"(src.registr()), "i"(offset / 4)
          : "memory");
    }
  }
  if constexpr ((bytes & 2U) != 0) {
    constexpr std::size_t offset = bytes & 12U;
    if constexpr (offset == 0) {
      store_first<2>(dst, src);
    } else {
      asm("st1.h { %1 }[%2], %0" // NOLINT
          : "=Q"(dst[offset / sizeof(Value)])
          : "w"(src.registr()), "i"(offset / 2)
          : "memory");
    }
  }
  if constexpr ((bytes & 1U) != 0) {
    constexpr std::size_t offset = bytes & 14U;
    if constexpr (offset == 0) {
      store_first<1>(dst, src);
    } else {
      vst1q_lane_u8(reinterpret_cast<u8*>(dst) + offset, as<u8>(src.registr()), offset);
    }
  }
}

template<Vectorizable T, std::size_t tPart, std::size_t tSize>
GREX_ALWAYS_INLINE inline void store(T* dst, SubVector<T, tPart, tSize> src) {
  store_part(dst, src, index_tag<tPart>);
}
template<Vectorizable T, std::size_t tPart, std::size_t tSize>
GREX_ALWAYS_INLINE inline void store_aligned(T* dst, SubVector<T, tPart, tSize> src) {
  store_part(dst, src, index_tag<tPart>);
}

#define GREX_PARTSTORE_ATTR_0
#define GREX_PARTSTORE_ATTR_1 [[unlikely]]
#define GREX_PARTSTORE_ATTR(INDEX, REF) GREX_CAT(GREX_PARTSTORE_ATTR_, GREX_EQUALS(INDEX, REF))

#define GREX_PARTSTORE_CASE(SIZE, INDEX, KIND, BITS) \
  GREX_PARTSTORE_ATTR(INDEX, 0) case INDEX: { \
    store_part(dst, src, index_tag<INDEX>); \
    return; \
  }
#define GREX_PARTSTORE(KIND, BITS, SIZE) \
  inline void store_part(KIND##BITS* dst, Vector<KIND##BITS, SIZE> src, std::size_t size) { \
    switch (size) { \
      GREX_REPEAT(SIZE, GREX_PARTSTORE_CASE, KIND, BITS) \
      [[unlikely]] GREX_PARTSTORE_CASE(SIZE, SIZE, KIND, BITS) default : std::unreachable(); \
    } \
  }
GREX_FOREACH_TYPE(GREX_PARTSTORE, 128)

#define GREX_SUBPARTSTORE(KIND, BITS, PART, SIZE) \
  inline void store_part(KIND##BITS* dst, SubVector<KIND##BITS, PART, SIZE> src, \
                         std::size_t size) { \
    switch (size) { \
      GREX_REPEAT(PART, GREX_PARTSTORE_CASE, KIND, BITS) \
      [[unlikely]] GREX_PARTSTORE_CASE(PART, PART, KIND, BITS) default : std::unreachable(); \
    } \
  }
GREX_FOREACH_SUB(GREX_SUBPARTSTORE)
} // namespace grex::backend

#include "grex/backend/shared/operations/store.hpp" // IWYU pragma: export

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_STORE_HPP
