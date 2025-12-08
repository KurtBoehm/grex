// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_TO_ARRAY_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_TO_ARRAY_HPP

#include <array>
#include <concepts>
#include <cstddef>
#include <utility>

#include "grex/backend/base.hpp"
#include "grex/backend/x86/operations/convert.hpp"
#include "grex/backend/x86/operations/store.hpp"
#include "grex/backend/x86/types.hpp"
#include "grex/base.hpp"

namespace grex::backend {
template<AnyVector TVec>
inline void to_array(typename TVec::Value* dst, TVec v) {
  store(dst, v);
}

#if GREX_X86_64_LEVEL >= 4
inline void to_array(bool* dst, Mask<u8, 16> m) {
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
inline void to_array(bool* dst, Mask<u8, 32> m) {
  const auto masked = _mm256_maskz_mov_epi8(m.r, _mm256_set1_epi8(1));
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst), masked);
}
inline void to_array(bool* dst, Mask<u8, 64> m) {
  const auto masked = _mm512_maskz_mov_epi8(m.r, _mm512_set1_epi8(1));
  _mm512_storeu_si512(reinterpret_cast<void*>(dst), masked);
}
#else
inline void to_array(bool* dst, Mask<u8, 16> m) {
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
inline void to_array(bool* dst, Mask<u8, 32> m) {
  const auto masked = _mm256_and_si256(m.r, _mm256_set1_epi8(1));
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst), masked);
}
#endif
#endif

template<AnyMask TMask>
inline void to_array(bool* dst, TMask m) {
  using VectorValue = TMask::VectorValue;
  constexpr std::size_t size = TMask::size;

  if constexpr (!std::same_as<VectorValue, u8>) {
    // elements are bigger than 1 byte → convert to 1-byte mask
    to_array(dst, convert<u8>(m));
  } else if constexpr (is_supernative<VectorValue, size>) {
    // super-native → store in halves
    to_array(dst, m.lower);
    to_array(dst + size / 2, m.upper);
  } else {
    static_assert(false, "Unsupported argument!");
    std::unreachable();
  }
}

template<AnyVector TVec>
inline std::array<typename TVec::Value, TVec::size> to_array(TVec v) {
  std::array<typename TVec::Value, TVec::size> buf{};
  to_array(buf.data(), v);
  return buf;
}
template<AnyMask TMask>
inline std::array<bool, TMask::size> to_array(TMask m) {
  std::array<bool, TMask::size> buf{};
  to_array(buf.data(), m);
  return buf;
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_TO_ARRAY_HPP
