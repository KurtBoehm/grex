// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_TO_ARRAY_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_TO_ARRAY_HPP

#include "grex/backend/base.hpp"
#include "grex/backend/x86/operations/store.hpp"
#include "grex/backend/x86/types.hpp"
#include "grex/base.hpp"

namespace grex::backend {
#if GREX_X86_64_LEVEL >= 4
inline void to_array(bool* dst, NativeMask<u8, 16> m) {
  const auto masked = _mm_maskz_mov_epi8(m.r, _mm_set1_epi8(1));
  _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), masked);
}
inline void to_array(bool* dst, SubMask<u8, 8, 16> m) {
  const auto masked = _mm_maskz_mov_epi8(m.full.r, _mm_set1_epi8(1));
  _mm_storeu_si64(static_cast<void*>(dst), masked);
}
inline void to_array(bool* dst, SubMask<u8, 4, 16> m) {
  const auto masked = _mm_maskz_mov_epi8(m.full.r, _mm_set1_epi8(1));
  _mm_storeu_si32(static_cast<void*>(dst), masked);
}
inline void to_array(bool* dst, SubMask<u8, 2, 16> m) {
  const auto masked = _mm_maskz_mov_epi8(m.full.r, _mm_set1_epi8(1));
  _mm_storeu_si16(static_cast<void*>(dst), masked);
}
inline void to_array(bool* dst, NativeMask<u8, 32> m) {
  const auto masked = _mm256_maskz_mov_epi8(m.r, _mm256_set1_epi8(1));
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst), masked);
}
inline void to_array(bool* dst, NativeMask<u8, 64> m) {
  const auto masked = _mm512_maskz_mov_epi8(m.r, _mm512_set1_epi8(1));
  _mm512_storeu_si512(reinterpret_cast<void*>(dst), masked);
}
#else
inline void to_array(bool* dst, NativeMask<u8, 16> m) {
  const auto masked = _mm_and_si128(m.r, _mm_set1_epi8(1));
  _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), masked);
}
inline void to_array(bool* dst, SubMask<u8, 8, 16> m) {
  const auto masked = _mm_and_si128(m.full.r, _mm_set1_epi8(1));
  _mm_storeu_si64(static_cast<void*>(dst), masked);
}
inline void to_array(bool* dst, SubMask<u8, 4, 16> m) {
  const auto masked = _mm_and_si128(m.full.r, _mm_set1_epi8(1));
  _mm_storeu_si32(static_cast<void*>(dst), masked);
}
inline void to_array(bool* dst, SubMask<u8, 2, 16> m) {
  const auto masked = _mm_and_si128(m.full.r, _mm_set1_epi8(1));
  _mm_storeu_si16(static_cast<void*>(dst), masked);
}
#if GREX_X86_64_LEVEL >= 3
inline void to_array(bool* dst, NativeMask<u8, 32> m) {
  const auto masked = _mm256_and_si256(m.r, _mm256_set1_epi8(1));
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst), masked);
}
#endif
#endif
} // namespace grex::backend

#include "grex/backend/shared/operations/to-array.hpp" // IWYU pragma: export

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_TO_ARRAY_HPP
