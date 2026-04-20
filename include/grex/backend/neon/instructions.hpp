// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_NEON_INSTRUCTIONS_HPP
#define INCLUDE_GREX_BACKEND_NEON_INSTRUCTIONS_HPP

#include <climits>
#include <concepts>

#include "grex/base.hpp"

namespace grex::isa {
// Generic bitfield insert: (dest & ~mask) | ((src & field_mask) << offset)
template<std::size_t tOffset, std::size_t tWidth, std::unsigned_integral T>
GREX_ALWAYS_INLINE constexpr T bitfield_insert(T dst, T src) {
  static_assert(tWidth > 0, "WIDTH must be > 0");
  static_assert(tOffset + tWidth <= sizeof(T) * CHAR_BIT, "Field out of range");

  constexpr T field_mask = (tWidth == sizeof(T) * CHAR_BIT) ? T(~T{0}) : T((T{1} << tWidth) - 1);
  constexpr T mask = field_mask << tOffset;

  return (dst & ~mask) | ((src & field_mask) << tOffset);
}

// Wrapper that either constant-folds or emits `bfi` on AArch64.
template<std::size_t tOffset, std::size_t tWidth, std::unsigned_integral T>
GREX_ALWAYS_INLINE inline T bfi(T dst, T src) {
  static_assert(tWidth > 0, "WIDTH must be > 0");
  static_assert(tOffset + tWidth <= sizeof(T) * CHAR_BIT, "Field out of range");

  // Only use inline assembly when not folded.
  if (!__builtin_constant_p(dst) && !__builtin_constant_p(src)) {
    if constexpr (sizeof(T) == 8) { // NOLINT
      asm("bfi %0, %1, %2, %3" : "+r"(dst) : "r"(src), "i"(tOffset), "i"(tWidth)); // NOLINT
      return dst;
    } else if constexpr (sizeof(T) == 4) {
      asm("bfi %w0, %w1, %2, %3" : "+r"(dst) : "r"(src), "i"(tOffset), "i"(tWidth)); // NOLINT
      return dst;
    }
  }

  // Fallback: generic implementation.
  return bitfield_insert<tOffset, tWidth>(dst, src);
}

template<std::size_t tShift>
struct LSL {};
template<std::size_t tShift>
inline constexpr LSL<tShift> lsl{};

// (32, 32) → 32: purely arithmetic, no inline asm needed.
template<std::size_t tShift>
GREX_ALWAYS_INLINE inline u32 orr32(u32 a, u32 b, LSL<tShift> /*lsl*/) {
  // asm("orr %w0, %w1, %w2, lsl #%c3" : "=r"(r) : "r"(a), "r"(b), "i"(tShift));
  return a | (b << tShift);
}

// (32, 32) → 64: assembly path for non-constant inputs.
template<std::size_t tShift>
GREX_ALWAYS_INLINE inline u64 orr64(u32 a, u32 b, LSL<tShift> /*lsl*/) {
  if (!__builtin_constant_p(a) && !__builtin_constant_p(b)) {
    u64 r{};
    asm("orr %w0, %w1, %w2, lsl #%c3" : "=r"(r) : "r"(a), "r"(b), "i"(tShift)); // NOLINT
    return r;
  }

  return a | (b << tShift);
}

// (64, 64) → 64: purely arithmetic, no inline asm needed.
template<std::size_t tShift>
GREX_ALWAYS_INLINE inline u64 orr64(u64 a, u64 b, LSL<tShift> /*lsl*/) {
  // asm("orr %0, %1, %2, lsl #%c3" : "=r"(r) : "r"(a), "r"(b), "i"(tShift));
  return a | (b << tShift);
}
} // namespace grex::isa

#endif // INCLUDE_GREX_BACKEND_NEON_INSTRUCTIONS_HPP
