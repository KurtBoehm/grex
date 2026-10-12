// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_SLIDE_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_SLIDE_HPP

#include <array>
#include <cstddef>

#include <immintrin.h>

#include "grex/backend/base.hpp"
#include "grex/backend/shared/defs.hpp"
#include "grex/backend/x86/instruction-sets.hpp"
#include "grex/backend/x86/operations/merge.hpp"
#include "grex/backend/x86/operations/reinterpret.hpp"
#include "grex/base.hpp"

namespace grex::backend {
namespace sld {
/**
 * Byte indices `[ByteShift, ByteShift + 1, ..., ByteShift + ByteSize - 1]` for `vpermt2b` and
 * `vpermb`.
 */
template<std::size_t ByteSize, std::size_t ByteShift>
alignas(ByteSize) inline constexpr auto byte_indices = [] {
  std::array<u8, ByteSize> idx{};
  for (std::size_t i = 0; i < ByteSize; ++i) {
    idx[i] = static_cast<u8>(ByteShift + i);
  }
  return idx;
}();

/**
 * Bit `i` is set iff byte `i` of a vector shifted by `ByteShift` bytes in direction `Dir` is not
 * shifted in, i.e. iff `i + ByteShift < ByteSize` (`down`) or `i ≥ ByteShift` (`up`). Bits beyond
 * `ByteSize` are ignored by the masked instructions.
 */
template<std::size_t ByteSize, ShiftDirection Dir, std::size_t ByteShift>
requires(ByteShift < ByteSize)
inline constexpr u64 shift_mask =
  (Dir == shift_down) ? (~u64{0} >> (64 - (ByteSize - ByteShift))) : (~u64{0} << ByteShift);

/** The native vector of type `V` with the bytes in `r`. */
template<AnyNativeVector V, typename R>
[[nodiscard, GREX_ALWAYS_INLINE]] inline V from_byte_register(R r) noexcept {
  return as<ValueOf<V>>(NativeVector<u8, sizeof(ValueOf<V>) * size_of<V>>{.r = r});
}
} // namespace sld

// The costs below mirror the implementations further down branch by branch, so the two have to be
// kept in sync. The µop counts are based on Zen 5; Intel CPUs can use more µops, e.g. three for
// `vpermt2b`. Constant loads (indices and masks) are not counted, as they are hoisted out of loops,
// and neither are zero idioms, which are handled at register renaming.

//==================================================================================================
// Costs for various instructions on Zen 5 based on uops.info
//==================================================================================================

namespace sld::cost {
// Zen 5, measured on uops.info. For instructions whose operands have different latencies, the
// latency is the larger one, as both sources of the two-source instructions are data operands.

/** No instruction, e.g. a shift by zero or one returning (only) zeros. */
inline constexpr Cost none{.inv_throughput = 0, .latency = 0};
/** `pshufd` */
inline constexpr Cost shuf_xmm{.inv_throughput = 0.25, .latency = 2};
/** `[v]ps[rl]ldq` (`xmm`/`ymm`), `[v]palignr` (`xmm`/`ymm`/`zmm`) */
inline constexpr Cost bshift{.inv_throughput = 0.5, .latency = 2};
/** `vextracti128` */
inline constexpr Cost extract128{.inv_throughput = 0.25, .latency = 2};
/** `vperm2i128` */
inline constexpr Cost perm2x128{.inv_throughput = 0.5, .latency = 3};
/** `valignq ymm` */
inline constexpr Cost alignq_ymm{.inv_throughput = 0.5, .latency = 3};
/** `valignd ymm`, `valignd/q zmm` */
inline constexpr Cost align{.inv_throughput = 0.5, .latency = 4};
/** `vpermq ymm imm`, `vpermd ymm`, `vpermb ymm`/`zmm` (also zero-masked), `vpermt2b ymm`/`zmm` */
inline constexpr Cost perm{.inv_throughput = 0.5, .latency = 4};

/**
 * `[v]psrldq` and `[v]pslldq` in parallel (FP12) followed by `[v]por` (FP0123): the two shifts
 * occupy FP12 for one cycle.
 */
inline constexpr Cost shift_or{.inv_throughput = 1, .latency = 4};
/** `vextracti128` (FP0123) followed by `vpalignr`/`vpsrldq` (FP12): 1 µop on FP12. */
inline constexpr Cost extract_bshift{.inv_throughput = 0.5, .latency = 4};
} // namespace sld::cost

namespace sld {
template<std::size_t ByteSize, std::size_t ByteShift>
requires(ByteShift <= ByteSize)
consteval Cost alignr_cost() {
  using namespace cost;

  if constexpr (ByteShift == 0 || ByteShift == ByteSize) {
    return none;
  } else if constexpr (ByteSize == 16) {
    return x86_64_level >= 2 ? bshift : shift_or;
  } else if constexpr (ByteSize == 32) {
    if constexpr (x86_64_level >= 4 && ByteShift % 8 == 0) {
      return alignq_ymm;
    } else if constexpr (x86_64_level >= 4 && ByteShift % 4 == 0) {
      return align;
    } else if constexpr (x86_64_level >= 4 && has_avx512vbmi) {
      return perm;
    } else if constexpr (ByteShift == 16) {
      return perm2x128;
    } else {
      return perm2x128 + bshift;
    }
  } else {
    static_assert(ByteSize == 64);
    if constexpr (ByteShift % 4 == 0) {
      return align;
    } else if constexpr (has_avx512vbmi) {
      return perm;
    } else {
      // One or two independent `valignd` and one `vpalignr`, all on FP12.
      static constexpr std::size_t q = ByteShift / 16;
      static constexpr f64 aligns = f64{q > 0} + f64{q < 3};
      return {.inv_throughput = 0.5 * (aligns + 1), .latency = align.latency + bshift.latency};
    }
  }
}

template<ShiftDirection Dir, std::size_t ByteSize, std::size_t ByteShift>
requires(ByteShift <= ByteSize)
consteval Cost shift_bytes_cost() {
  using namespace cost;
  static constexpr bool down = Dir == shift_down;

  if constexpr (ByteShift == 0 || ByteShift == ByteSize) {
    return none;
  } else if constexpr (ByteSize == 16) {
    return bshift;
  } else if constexpr (ByteSize == 32) {
    if constexpr (x86_64_level >= 4 && ByteShift % 4 == 0) {
      static constexpr std::size_t align_shift = down ? ByteShift : 32 - ByteShift;
      return alignr_cost<32, align_shift>();
    } else if constexpr (x86_64_level >= 4 && has_avx512vbmi) {
      return perm;
    } else if constexpr (down) {
      return ByteShift == 16 ? extract128 : extract_bshift;
    } else {
      return ByteShift == 16 ? perm2x128 : perm2x128 + bshift;
    }
  } else {
    static_assert(ByteSize == 64);
    if constexpr (ByteShift % 4 != 0 && has_avx512vbmi) {
      return perm;
    } else {
      static constexpr std::size_t align_shift = down ? ByteShift : 64 - ByteShift;
      return alignr_cost<64, align_shift>();
    }
  }
}

template<std::size_t ByteSize, std::size_t ByteShift>
requires(ByteShift < ByteSize)
consteval Cost rotr_cost() {
  using namespace cost;

  if constexpr (ByteShift == 0) {
    return none;
  } else if constexpr (ByteSize == 16) {
    return ByteShift % 4 == 0 ? shuf_xmm : alignr_cost<16, ByteShift>();
  } else if constexpr (ByteSize == 32) {
    if constexpr (x86_64_level >= 4 && ByteShift % 4 == 0) {
      return alignr_cost<32, ByteShift>();
    } else if constexpr (ByteShift % 4 == 0) {
      // `vpermq` with an immediate or `vpermd`.
      return perm;
    } else if constexpr (x86_64_level >= 4 && has_avx512vbmi) {
      return perm;
    } else {
      return alignr_cost<32, ByteShift>();
    }
  } else {
    static_assert(ByteSize == 64);
    if constexpr (ByteShift % 4 != 0 && has_avx512vbmi) {
      return perm;
    } else {
      return alignr_cost<64, ByteShift>();
    }
  }
}
} // namespace sld

/** The cost of `concat_slide_down<Shift>` on native vectors of `N` values of `ValueBytes` bytes. */
template<std::size_t ValueBytes, std::size_t N, std::size_t Shift>
requires(Shift <= N && ValueBytes * N <= register_bytes.back())
consteval Cost concat_slide_down_cost() {
  return sld::alignr_cost<ValueBytes * N, ValueBytes * Shift>();
}

/** The cost of `slide<Dir, Shift>` on native vectors of `N` values of `ValueBytes` bytes. */
template<std::size_t ValueBytes, std::size_t N, ShiftDirection Dir, std::size_t Shift>
requires(Shift <= N)
consteval Cost slide_cost() {
  return sld::shift_bytes_cost<Dir, ValueBytes * N, ValueBytes * Shift>();
}

/** The cost of `rotate_down<Shift>` on native vectors of `N` values of `ValueBytes` bytes. */
template<std::size_t ValueBytes, std::size_t N, std::size_t Shift>
requires(Shift <= N)
consteval Cost rotate_down_cost() {
  static constexpr std::size_t shift = Shift % N;
  return sld::rotr_cost<ValueBytes * N, ValueBytes * shift>();
}

//==================================================================================================
// Slide the concatenated vector down and take the lower half
//==================================================================================================

namespace sld {
template<std::size_t ByteShift>
requires(ByteShift <= 16)
[[nodiscard, GREX_ALWAYS_INLINE]] inline __m128i alignr(__m128i lo, __m128i hi) noexcept {
  if constexpr (ByteShift == 0) {
    return lo;
  } else if constexpr (ByteShift == 16) {
    return hi;
  } else if constexpr (x86_64_level >= 2) {
    return _mm_alignr_epi8(hi, lo, ByteShift);
  } else {
    return _mm_or_si128(_mm_bslli_si128(hi, 16 - ByteShift), _mm_bsrli_si128(lo, ByteShift));
  }
}

#if GREX_X86_64_LEVEL >= 3
template<std::size_t ByteShift>
requires(ByteShift <= 32)
[[nodiscard, GREX_ALWAYS_INLINE]] inline __m256i alignr(__m256i lo, __m256i hi) noexcept {
  if constexpr (ByteShift == 0) {
    return lo;
  } else if constexpr (ByteShift == 32) {
    return hi;
  } else if constexpr (x86_64_level >= 4 && ByteShift % 8 == 0) {
    return _mm256_alignr_epi64(hi, lo, ByteShift / 8);
  } else if constexpr (x86_64_level >= 4 && ByteShift % 4 == 0) {
    return _mm256_alignr_epi32(hi, lo, ByteShift / 4);
  } else if constexpr (x86_64_level >= 4 && has_avx512vbmi) {
    const __m256i shufi =
      _mm256_load_si256(reinterpret_cast<const __m256i*>(byte_indices<32, ByteShift>.data()));
    return _mm256_permutex2var_epi8(lo, shufi, hi);
  } else {
    // `[lo[16:], hi[:16]]`.
    const __m256i mid = _mm256_permute2x128_si256(lo, hi, 0x21);
    if constexpr (ByteShift == 16) {
      return mid;
    } else if constexpr (ByteShift < 16) {
      return _mm256_alignr_epi8(mid, lo, ByteShift);
    } else {
      return _mm256_alignr_epi8(hi, mid, ByteShift - 16);
    }
  }
}
#endif

#if GREX_X86_64_LEVEL >= 4
template<std::size_t ByteShift>
requires(ByteShift <= 64)
[[nodiscard, GREX_ALWAYS_INLINE]] inline __m512i alignr(__m512i lo, __m512i hi) noexcept {
  if constexpr (ByteShift == 0) {
    return lo;
  } else if constexpr (ByteShift == 64) {
    return hi;
  } else if constexpr (ByteShift % 8 == 0) {
    return _mm512_alignr_epi64(hi, lo, ByteShift / 8);
  } else if constexpr (ByteShift % 4 == 0) {
    return _mm512_alignr_epi32(hi, lo, ByteShift / 4);
  } else if constexpr (has_avx512vbmi) {
    return _mm512_permutex2var_epi8(lo, _mm512_load_si512(byte_indices<64, ByteShift>.data()), hi);
  } else {
    // `a` and `b` are `[lo, hi]` shifted by `q` and `q + 1` whole lanes; the remaining shift is
    // `ByteShift % 16`.
    static constexpr std::size_t q = ByteShift / 16;
    __m512i a = lo;
    if constexpr (q > 0) {
      a = _mm512_alignr_epi32(hi, lo, 4 * q);
    }
    __m512i b = hi;
    if constexpr (q < 3) {
      b = _mm512_alignr_epi32(hi, lo, 4 * (q + 1));
    }
    return _mm512_alignr_epi8(b, a, ByteShift % 16);
  }
}
#endif
} // namespace sld

template<std::size_t Shift, AnyNativeVector V>
requires(Shift <= size_of<V>)
[[nodiscard, GREX_ALWAYS_INLINE]] inline V concat_slide_down(V lo, V hi) noexcept {
  const auto r = sld::alignr<sizeof(ValueOf<V>) * Shift>(as<u8>(lo).r, as<u8>(hi).r);
  return sld::from_byte_register<V>(r);
}

//==================================================================================================
// Move lanes down or up, filling the vacated elements with zero
//==================================================================================================

namespace sld {
template<ShiftDirection Dir, std::size_t ByteShift>
requires(ByteShift <= 16)
[[nodiscard, GREX_ALWAYS_INLINE]] inline __m128i shift_bytes(__m128i v) noexcept {
  if constexpr (ByteShift == 0) {
    return v;
  } else if constexpr (ByteShift == 16) {
    return _mm_setzero_si128();
  } else if constexpr (Dir == shift_down) {
    return _mm_bsrli_si128(v, ByteShift);
  } else {
    return _mm_bslli_si128(v, ByteShift);
  }
}

#if GREX_X86_64_LEVEL >= 3
template<ShiftDirection Dir, std::size_t ByteShift>
requires(ByteShift <= 32)
[[nodiscard, GREX_ALWAYS_INLINE]] inline __m256i shift_bytes(__m256i v) noexcept {
  static constexpr bool down = Dir == shift_down;
  if constexpr (ByteShift == 0) {
    return v;
  } else if constexpr (ByteShift == 32) {
    return _mm256_setzero_si256();
  } else if constexpr (x86_64_level >= 4 && ByteShift % 4 == 0) {
    // `valignd`/`valignq` with zero.
    const __m256i zero = _mm256_setzero_si256();
    return down ? alignr<ByteShift>(v, zero) : alignr<32 - ByteShift>(zero, v);
  } else if constexpr (x86_64_level >= 4 && has_avx512vbmi) {
    // The zero fill allows the single-source `vpermb` instead of `vpermt2b`. As `vpermb` only uses
    // the low five index bits, the indices for shifting up are `(i - ByteShift) % 32`.
    static constexpr std::size_t index_shift = down ? ByteShift : (32 - ByteShift);
    const __m256i shufi =
      _mm256_load_si256(reinterpret_cast<const __m256i*>(byte_indices<32, index_shift>.data()));
    return _mm256_maskz_permutexvar_epi8(static_cast<__mmask32>(shift_mask<32, Dir, ByteShift>),
                                         shufi, v);
  } else if constexpr (down) {
    // `[v[16:], 0]`, which compilers emit more reliably than `alignr` with zero.
    const __m256i mid = _mm256_zextsi128_si256(_mm256_extracti128_si256(v, 1));
    if constexpr (ByteShift == 16) {
      return mid;
    } else if constexpr (ByteShift < 16) {
      return _mm256_alignr_epi8(mid, v, ByteShift);
    } else {
      return _mm256_bsrli_epi128(mid, ByteShift - 16);
    }
  } else {
    // `[0, v[:16]]`.
    const __m256i mid = _mm256_permute2x128_si256(v, v, 0x08);
    if constexpr (ByteShift == 16) {
      return mid;
    } else if constexpr (ByteShift < 16) {
      return _mm256_alignr_epi8(v, mid, 16 - ByteShift);
    } else {
      return _mm256_bslli_epi128(mid, ByteShift - 16);
    }
  }
}
#endif

#if GREX_X86_64_LEVEL >= 4
template<ShiftDirection Dir, std::size_t ByteShift>
requires(ByteShift <= 64)
[[nodiscard, GREX_ALWAYS_INLINE]] inline __m512i shift_bytes(__m512i v) noexcept {
  static constexpr bool down = Dir == shift_down;
  if constexpr (ByteShift == 0) {
    return v;
  } else if constexpr (ByteShift == 64) {
    return _mm512_setzero_si512();
  } else if constexpr (ByteShift % 4 != 0 && has_avx512vbmi) {
    // The zero fill allows the single-source `vpermb` instead of `vpermt2b`. As `vpermb` only uses
    // the low six index bits, the indices for shifting up are `(i - ByteShift) % 64`.
    static constexpr std::size_t index_shift = down ? ByteShift : (64 - ByteShift);
    const __m512i shufi = _mm512_load_si512(byte_indices<64, index_shift>.data());
    return _mm512_maskz_permutexvar_epi8(shift_mask<64, Dir, ByteShift>, shufi, v);
  } else {
    // `valignd`/`valignq` with zero if `ByteShift % 4 == 0`, otherwise the emulation.
    const __m512i zero = _mm512_setzero_si512();
    return down ? alignr<ByteShift>(v, zero) : alignr<64 - ByteShift>(zero, v);
  }
}
#endif
} // namespace sld

template<ShiftDirection Dir, std::size_t Shift, AnyNativeVector V>
requires(Shift <= size_of<V>)
[[nodiscard, GREX_ALWAYS_INLINE]] inline V slide(V v) noexcept {
  const auto r = sld::shift_bytes<Dir, sizeof(ValueOf<V>) * Shift>(as<u8>(v).r);
  return sld::from_byte_register<V>(r);
}

//==================================================================================================
// Rotate, i.e. slide the vector concatenated with itself
//==================================================================================================

namespace sld {
/** The immediate for `pshufd`/`vpermq` that rotates four elements down by `Q`. */
template<std::size_t Q>
inline constexpr int rotate4_imm =
  static_cast<int>((Q % 4) | ((Q + 1) % 4) << 2 | ((Q + 2) % 4) << 4 | ((Q + 3) % 4) << 6);

template<std::size_t ByteShift>
requires(ByteShift < 16)
[[nodiscard, GREX_ALWAYS_INLINE]] inline __m128i rotr(__m128i v) noexcept {
  if constexpr (ByteShift == 0) {
    return v;
  } else if constexpr (ByteShift % 4 == 0) {
    // `pshufd`: one instruction on every level, and non-destructive without AVX.
    return _mm_shuffle_epi32(v, rotate4_imm<ByteShift / 4>);
  } else {
    return alignr<ByteShift>(v, v);
  }
}

#if GREX_X86_64_LEVEL >= 3
template<std::size_t ByteShift>
requires(ByteShift < 32)
[[nodiscard, GREX_ALWAYS_INLINE]] inline __m256i rotr(__m256i v) noexcept {
  if constexpr (ByteShift == 0) {
    return v;
  } else if constexpr (x86_64_level >= 4 && ByteShift % 4 == 0) {
    // `valignd`/`valignq`.
    return alignr<ByteShift>(v, v);
  } else if constexpr (ByteShift % 8 == 0) {
    // `vpermq` with an immediate.
    return _mm256_permute4x64_epi64(v, rotate4_imm<ByteShift / 8>);
  } else if constexpr (ByteShift % 4 == 0) {
    // `vpermd` only uses the low three index bits, so these indices are `(i + q) % 8`.
    static constexpr int q = ByteShift / 4;
    const __m256i shufi = _mm256_setr_epi32(q, q + 1, q + 2, q + 3, q + 4, q + 5, q + 6, q + 7);
    return _mm256_permutevar8x32_epi32(v, shufi);
  } else if constexpr (x86_64_level >= 4 && has_avx512vbmi) {
    // A single source allows `vpermb` instead of `vpermt2b`; as it only uses the low five index
    // bits, these indices are `(i + ByteShift) % 32`.
    const __m256i shufi =
      _mm256_load_si256(reinterpret_cast<const __m256i*>(byte_indices<32, ByteShift>.data()));
    return _mm256_permutexvar_epi8(shufi, v);
  } else {
    return alignr<ByteShift>(v, v);
  }
}
#endif

#if GREX_X86_64_LEVEL >= 4
template<std::size_t ByteShift>
requires(ByteShift < 64)
[[nodiscard, GREX_ALWAYS_INLINE]] inline __m512i rotr(__m512i v) noexcept {
  if constexpr (ByteShift == 0) {
    return v;
  } else if constexpr (ByteShift % 4 != 0 && has_avx512vbmi) {
    // A single source allows `vpermb` instead of `vpermt2b`; as it only uses the low six index
    // bits, these indices are `(i + ByteShift) % 64`.
    return _mm512_permutexvar_epi8(_mm512_load_si512(byte_indices<64, ByteShift>.data()), v);
  } else {
    // `valignd`/`valignq` if `ByteShift % 4 == 0`, otherwise the emulation.
    return alignr<ByteShift>(v, v);
  }
}
#endif
} // namespace sld

template<std::size_t Shift, AnyNativeVector V>
requires(Shift <= size_of<V>)
[[nodiscard, GREX_ALWAYS_INLINE]] inline V rotate_down(V v) noexcept {
  const auto r = sld::rotr<sizeof(ValueOf<V>) * (Shift % size_of<V>)>(as<u8>(v).r);
  return sld::from_byte_register<V>(r);
}

//==================================================================================================
// Non-native vectors
//==================================================================================================

/**
 * Slide within the active part of the register, so that zeros (rather than the contents of the
 * inactive lanes) are moved in.
 */
template<ShiftDirection Dir, std::size_t Shift, AnySubNativeVector V>
requires(Shift <= size_of<V>)
[[nodiscard, GREX_ALWAYS_INLINE]] inline V slide(V v) noexcept {
  static constexpr bool down = Dir == shift_down;
  static constexpr std::size_t byte_size = sizeof(ValueOf<V>) * size_of<V>;
  static constexpr int bit_shift = 8 * sizeof(ValueOf<V>) * Shift;

  const __m128i ur = as<u8>(v.full).r;
  const __m128i r = [&] [[GREX_ALWAYS_INLINE]] {
    if constexpr (byte_size == 8) {
      return down ? _mm_srli_epi64(ur, bit_shift) : _mm_slli_epi64(ur, bit_shift);
    } else if constexpr (byte_size == 4) {
      return down ? _mm_srli_epi32(ur, bit_shift) : _mm_slli_epi32(ur, bit_shift);
    } else {
      static_assert(byte_size == 2);
      return down ? _mm_srli_epi16(ur, bit_shift) : _mm_slli_epi16(ur, bit_shift);
    }
  }();
  return V{sld::from_byte_register<typename V::Full>(r)};
}
} // namespace grex::backend

#include "grex/backend/shared/operations/slide.hpp" // IWYU pragma: keep

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_SLIDE_HPP
