// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_SHUFFLE_128_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_SHUFFLE_128_HPP

#include <array>
#include <optional>

#include "thesauros/types/value-tag.hpp"

#include "grex/backend/x86/instruction-sets.hpp"
#include "grex/backend/x86/operations/shuffle/aux.hpp"
#include "grex/backend/x86/operations/shuffle/check.hpp"
#include "grex/base/defs.hpp"

namespace grex::backend {
inline __m128i shuffle2x64(const __m128i v,
                           thes::TypedValueTag<std::array<ShuffleIndex, 2>> auto tag) {
  static constexpr auto idxs = tag.value;
  static constexpr ShuffleChecker<8, 2> flags{idxs};
  static_assert(!flags.out_of_range(), "Index out of range in permute function");

  __m128i out = v;

  if constexpr (flags.noop()) {
    return v;
  }
  if constexpr (flags.all_zero()) {
    return _mm_setzero_si128();
  }

  if constexpr (flags.permute()) {
    static constexpr auto rotate = flags.rotate();

    if constexpr (rotate && rotate->shleft != none_oz) {
      // pslldq does both permutation and zeroing
      static_assert(rotate->shleft == no_zeroing_oz);
      return _mm_bslli_si128(v, 8);
    }
    if constexpr (rotate && rotate->shright != none_oz) {
      // psrldq does both permutation and zeroing
      static_assert(rotate->shright == no_zeroing_oz);
      return _mm_bsrli_si128(v, 8);
    }
    if constexpr (flags.unpackhi()) {
      out = _mm_unpackhi_epi64(v, v);
    } else if constexpr (flags.unpacklo()) {
      out = _mm_unpacklo_epi64(v, v);
    } else {
      out = _mm_shuffle_epi32(v, u32(idxs[0]) * 0x0A + u32(idxs[1]) * 0xA0 + 0x44);
    }
  }
  if constexpr (flags.zeroing()) {
#if GREX_X86_64_LEVEL >= 4
    out = _mm_maskz_mov_epi64(zero_mask<2>(idxs), out);
#else
    // use unpack to avoid using data cache
    if constexpr (idxs[0] == zero_sh) {
      out = _mm_unpackhi_epi64(_mm_setzero_si128(), out);
    } else if constexpr (idxs[1] == zero_sh) {
      out = _mm_unpacklo_epi64(out, _mm_setzero_si128());
    }
#endif
  }
  return out;
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_SHUFFLE_128_HPP
