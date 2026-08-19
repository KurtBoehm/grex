// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_NEON_OPERATIONS_EXPAND_REGISTER_HPP
#define INCLUDE_GREX_BACKEND_NEON_OPERATIONS_EXPAND_REGISTER_HPP

#include <concepts>

#include <arm_neon.h>

#include "grex/backend/defs.hpp" // IWYU pragma: keep
#include "grex/backend/macros/for-each.hpp"
#include "grex/backend/neon/macros/types.hpp"
#include "grex/base.hpp"

#if GREX_CLANG
#include <cstddef>
#endif

#if GREX_GCC
#include "grex/backend/neon/operations/reinterpret.hpp"
#endif

namespace grex::backend {
/** Casts from `TSrc` to `TDst` with arbitrary values in the upper bits. */
template<IntVectorizable TDst, IntVectorizable TSrc>
inline TDst expand_bits(TSrc src) {
  if (__builtin_constant_p(src)) {
    return TDst(src);
  }
  TDst dst;
  asm("" : "=r"(dst) : "0"(src)); // NOLINT
  return dst;
}

#if GREX_CLANG
#define GREX_EXPAND_REGISTER_IMPL(KIND, BITS, SIZE) \
  using Single = KIND##BITS __attribute__((ext_vector_type(1))); \
  const Single s = x; \
  return static_apply<SIZE>([&]<std::size_t... tIdxs>() -> GREX_REGISTER(KIND, BITS, SIZE) { \
    return __builtin_shufflevector(s, s, ((tIdxs == 0) ? 0 : -1)...); \
  });
#define GREX_EXPAND_REGISTER_f GREX_EXPAND_REGISTER_IMPL
#define GREX_EXPAND_REGISTER_i GREX_EXPAND_REGISTER_IMPL
#define GREX_EXPAND_REGISTER_u GREX_EXPAND_REGISTER_IMPL
#elif GREX_GCC
#define GREX_EXPAND_REGISTER_f(KIND, BITS, SIZE) \
  float##BITS##x##SIZE##_t r; \
  asm("" : "=w"(r) : "0"(x)); \
  return r;

#define GREX_EXPAND_INT_BIG(KIND, BITS, SIZE) \
  return as<KIND##BITS>(expand_register(std::bit_cast<f##BITS>(x)));
#define GREX_EXPAND_INT_SMALL(KIND, BITS, SIZE) \
  const auto expanded = expand_bits<KIND##32>(x); \
  return as<KIND##BITS>(expand_register(std::bit_cast<f32>(expanded)));
#define GREX_EXPAND_INT64 GREX_EXPAND_INT_BIG
#define GREX_EXPAND_INT32 GREX_EXPAND_INT_BIG
#define GREX_EXPAND_INT16 GREX_EXPAND_INT_SMALL
#define GREX_EXPAND_INT8 GREX_EXPAND_INT_SMALL
#define GREX_EXPAND_REGISTER_i(KIND, BITS, SIZE) GREX_EXPAND_INT##BITS(KIND, BITS, SIZE)
#define GREX_EXPAND_REGISTER_u(KIND, BITS, SIZE) GREX_EXPAND_INT##BITS(KIND, BITS, SIZE)
#endif

#define GREX_EXPAND_REGISTER(KIND, BITS, SIZE) \
  template<std::same_as<KIND##BITS> T> \
  GREX_ALWAYS_INLINE inline GREX_REGISTER(KIND, BITS, SIZE) expand_register(T x) { \
    GREX_EXPAND_REGISTER_##KIND(KIND, BITS, SIZE) \
  }
GREX_FOREACH_TYPE_EXT(GREX_EXPAND_REGISTER, 128)
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_EXPAND_REGISTER_HPP
