// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_BIT_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_BIT_HPP

#include <bit>

#include "grex/base/defs.hpp"

namespace grex::backend {
// Casts from TSrc to TDst with arbitrary values in
template<IntVectorizable TDst, IntVectorizable TSrc>
inline TDst expand_any(TSrc src) {
  if (__builtin_constant_p(src)) {
    return TDst(src);
  }
  TDst dst;
  asm("" : "=r"(dst) : "0"(src));
  return dst;
}

#define GREX_BITTEST_COND \
  (__builtin_constant_p(a) != 0 && __builtin_constant_p(b) != 0) || \
    (__builtin_constant_p(a) != 0 && std::has_single_bit(a)) || \
    (__builtin_constant_p(b) != 0 && b == 0)

inline bool bit_test(u64 a, u64 b) {
  if (GREX_BITTEST_COND) {
    return ((a >> b) & 1) != 0;
  }
  bool dst{};
  asm("bt{q %2, %1 | %1, %2}" : "=@ccc"(dst) : "r"(a), "ir"(b) : "cc");
  return dst;
}
inline bool bit_test(u32 a, u32 b) {
  if (GREX_BITTEST_COND) {
    return ((a >> b) & 1) != 0;
  }
  bool dst{};
  asm("bt{l %2, %1 | %1, %2}" : "=@ccc"(dst) : "r"(a), "ir"(b) : "cc");
  return dst;
}
inline bool bit_test(u16 a, u16 b) {
  if (GREX_BITTEST_COND) {
    return ((a >> b) & 1) != 0;
  }
  bool dst{};
  asm("bt{w %2, %1 | %1, %2}" : "=@ccc"(dst) : "r"(a), "ir"(b) : "cc");
  return dst;
}
inline bool bit_test(u8 a, u8 b) {
  return bit_test(expand_any<u16>(a), expand_any<u16>(b));
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_BIT_HPP
