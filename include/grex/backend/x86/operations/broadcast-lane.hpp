// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_BROADCAST_LANE_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_BROADCAST_LANE_HPP

#include <cstddef>

#include "grex/backend/base.hpp"
#include "grex/backend/shared/defs.hpp"
#include "grex/backend/x86/instruction-sets.hpp"
#include "grex/backend/x86/types.hpp"
#include "grex/base.hpp"

namespace grex::backend {
namespace bcast_lane {
consteval int splat(int i) {
  return i * 0x55;
}

/** SSE2 16-bit broadcast: `pshuf[lh]w` + `pshufd`. 2 µops. */
template<std::size_t I>
requires(I < 8)
[[nodiscard, GREX_ALWAYS_INLINE]] inline __m128i sse2_bcast16(__m128i v) {
  if constexpr (I < 4) {
    static constexpr int imm = splat(I);
    return _mm_shuffle_epi32(_mm_shufflelo_epi16(v, imm), 0x00);
  } else {
    static constexpr int imm = splat(I - 4);
    return _mm_shuffle_epi32(_mm_shufflehi_epi16(v, imm), 0xFF);
  }
}
} // namespace bcast_lane

// The µop counts are based on Zen 5; even modern Intel CPUs can use more µops, e.g. for `vpermw`.

//==================================================================================================
// Costs for various instructions on Zen 5 base on uops.info
//==================================================================================================

namespace bcast_lane::cost {
// Zen 5, measured on uops.info. For instructions whose operands have different latencies, the
// latency is that of the data operand.

// `vpshufd`, `vpermilps`, `vpshuf[lh]w` (`xmm`/`ymm`), `punpck*`, `movddup`, `unpck[lh]pd`,
// `shufps`, `vpbroadcastb/w xmm←xmm`
inline constexpr Cost shuf_xmm{.inv_throughput = 0.25, .latency = 2};
// `vpshufb xmm`/`ymm`, `vpsrldq xmm`
inline constexpr Cost pshufb{.inv_throughput = 0.5, .latency = 2};
// `vpbroadcastb/w`, `vbroadcastss ymm`←`xmm`
inline constexpr Cost bcast_ymm{.inv_throughput = 0.5, .latency = 2};
// `vpbroadcastb/w`, `vbroadcastss/sd zmm`←`xmm`
inline constexpr Cost bcast_zmm{.inv_throughput = 0.5, .latency = 1};
// `vpbroadcastd/q zmm`←`xmm` (also used for `f32`/`f64`, which take 1 cycle)
inline constexpr Cost bcast_zmm_dq{.inv_throughput = 0.5, .latency = 2};
// `vpermd/ps/w/b ymm`, `vpermq/pd ymm imm`
inline constexpr Cost perm_ymm{.inv_throughput = 0.5, .latency = 4};
// `vpermd/ps/pd zmm` (`vpermq zmm`: 4)
inline constexpr Cost perm_zmm{.inv_throughput = 0.5, .latency = 5};
// `vpermw/b zmm`
inline constexpr Cost perm_zmm_bw{.inv_throughput = 0.5, .latency = 4};
// `vshufi32x4 zmm` with the same register as both sources
inline constexpr Cost shuf128_zmm{.inv_throughput = 0.5, .latency = 5};
// `vextracti128`
inline constexpr Cost extract128{.inv_throughput = 0.25, .latency = 2};
// `vextracti32x4`
inline constexpr Cost extract_zmm{.inv_throughput = 0.5, .latency = 4};
// `vpshufb zmm`
inline constexpr Cost pshufb_zmm{.inv_throughput = 0.5, .latency = 1};
} // namespace bcast_lane::cost

template<std::size_t ValueBytes, std::size_t N, std::size_t I>
[[nodiscard]] consteval Cost broadcast_lane_cost() {
  using namespace bcast_lane::cost;

  if constexpr (ValueBytes * N == 16) {
    if constexpr (ValueBytes >= 4) {
      return shuf_xmm;
    } else if constexpr (I == 0 && x86_64_level >= 3) {
      // `vpbroadcastb/w` `xmm←xmm`
      return shuf_xmm;
    } else if constexpr (x86_64_level >= 2) {
      return pshufb;
    } else if constexpr (ValueBytes == 2) {
      return shuf_xmm + shuf_xmm;
    } else {
      return shuf_xmm + shuf_xmm + shuf_xmm;
    }
  } else if constexpr (ValueBytes * N == 32) {
    if constexpr (ValueBytes == 8) {
      return perm_ymm;
    } else if constexpr (I == 0) {
      return bcast_ymm;
    } else if constexpr (ValueBytes == 4) {
      return perm_ymm;
    } else if constexpr (ValueBytes == 2) {
      // `vpshuf[lh]w` (FP0123) + `vpermq` (FP12): 2 µops over 4 pipes, 1 on FP12.
      return x86_64_level >= 4 ? perm_ymm : Cost{.inv_throughput = 0.5, .latency = 6};
    } else if constexpr (has_avx512vbmi && x86_64_level >= 4) {
      return perm_ymm;
    } else if constexpr (I < 16) {
      // `vpsrldq` + `vpbroadcastb`
      return pshufb + bcast_ymm;
    } else if constexpr (I == 16) {
      // `vextracti128` (FP0123) + `vpbroadcastb` (FP12).
      return Cost{.inv_throughput = 0.5, .latency = 4};
    } else {
      return pshufb + perm_ymm;
    }
  } else {
    if constexpr (I == 0) {
      return ValueBytes >= 4 ? bcast_zmm_dq : bcast_zmm;
    } else if constexpr (ValueBytes >= 4) {
      return perm_zmm;
    } else if constexpr (ValueBytes == 2 || has_avx512vbmi) {
      return perm_zmm_bw;
    } else if constexpr (I < 16) {
      // `vpsrldq` + `vpbroadcastb`
      return pshufb + bcast_zmm;
    } else if constexpr (I % 16 == 0) {
      return extract_zmm + bcast_zmm;
    } else {
      // `vshufi32x4` + `vpshufb`
      return shuf128_zmm + pshufb_zmm;
    }
  }
}

//==================================================================================================
// 128-bit
//==================================================================================================

/** `f64`: `movddup` starting on level 2, `unpcklpd`/`unpckhpd` on level 1. 1 µop. */
template<std::size_t I>
requires(I < 2)
[[nodiscard, GREX_ALWAYS_INLINE]] inline f64x2 broadcast_lane(f64x2 v, IndexTag<I> /*idx*/) {
  if constexpr (I == 0 && x86_64_level >= 2) {
    return {.r = _mm_movedup_pd(v.r)};
  } else if constexpr (I == 0) {
    return {.r = _mm_unpacklo_pd(v.r, v.r)};
  } else {
    return {.r = _mm_unpackhi_pd(v.r, v.r)};
  }
}

/** `f32`: `shufps` before level 3, `vpermilps` afterwards (non-destructive). 1 µop. */
template<std::size_t I>
requires(I < 4)
[[nodiscard, GREX_ALWAYS_INLINE]] inline f32x4 broadcast_lane(f32x4 v, IndexTag<I> /*idx*/) {
  static constexpr int imm = bcast_lane::splat(I);
  if constexpr (x86_64_level >= 3) {
    return {.r = _mm_permute_ps(v.r, imm)};
  } else {
    return {.r = _mm_shuffle_ps(v.r, v.r, imm)};
  }
}

/** `i64`/`u64`: `pshufd`. 1 µop. */
template<std::size_t I, Int64 T>
requires(I < 2)
[[nodiscard, GREX_ALWAYS_INLINE]] inline NativeVector<T, 2> broadcast_lane(NativeVector<T, 2> v,
                                                                           IndexTag<I> /*idx*/) {
  static constexpr int imm = (I > 0) ? 0xEE : 0x44;
  return {.r = _mm_shuffle_epi32(v.r, imm)};
}

/** `i32`/`u32`: `pshufd`. 1 µop. */
template<std::size_t I, Int32 T>
requires(I < 4)
[[nodiscard, GREX_ALWAYS_INLINE]] inline NativeVector<T, 4> broadcast_lane(NativeVector<T, 4> v,
                                                                           IndexTag<I> /*idx*/) {
  static constexpr int imm = bcast_lane::splat(I);
  return {.r = _mm_shuffle_epi32(v.r, imm)};
}

/**
 * `i16`/`u16`:
 * - level 1: `pshuf[lh]w` + `pshufd`. 2 µops.
 * - level 2+: `pshufb` + constant load. 1 µop.
 * - level 3, lane 0: `vpbroadcastw`. 1 µop.
 */
template<std::size_t I, Any16 T>
requires(I < 8)
[[nodiscard, GREX_ALWAYS_INLINE]] inline NativeVector<T, 8> broadcast_lane(NativeVector<T, 8> v,
                                                                           IndexTag<I> /*idx*/) {
  if constexpr (I == 0 && x86_64_level >= 3) {
    return {.r = _mm_broadcastw_epi16(v.r)};
  } else if constexpr (x86_64_level >= 2) {
    return {.r = _mm_shuffle_epi8(v.r, _mm_set1_epi16(static_cast<i16>((2 * I + 1) << 8 | 2 * I)))};
  } else {
    return {.r = bcast_lane::sse2_bcast16<I>(v.r)};
  }
}

/**
 * `i8`/`u8`:
 * - level 1: `punpck[lh]bw` + `pshuf[lh]w` + `pshufd`. 3 µops.
 * - level 2+: `pshufb` + constant load. 1 µop.
 * - level 3, lane 0: `vpbroadcastb`. 1 µop.
 */
template<std::size_t I, Int8 T>
requires(I < 16)
[[nodiscard, GREX_ALWAYS_INLINE]] inline NativeVector<T, 16> broadcast_lane(NativeVector<T, 16> v,
                                                                            IndexTag<I> /*idx*/) {
  if constexpr (I == 0 && x86_64_level >= 3) {
    return {.r = _mm_broadcastb_epi8(v.r)};
  } else if constexpr (x86_64_level >= 2) {
    return {.r = _mm_shuffle_epi8(v.r, _mm_set1_epi8(static_cast<i8>(I)))};
  } else if constexpr (I < 8) {
    return {.r = bcast_lane::sse2_bcast16<I % 8>(_mm_unpacklo_epi8(v.r, v.r))};
  } else {
    return {.r = bcast_lane::sse2_bcast16<I % 8>(_mm_unpackhi_epi8(v.r, v.r))};
  }
}

#if GREX_X86_64_LEVEL >= 3
//==================================================================================================
// 256-bit (level 3+)
//==================================================================================================

/** `f64`: `vpermpd`. 1 µop. */
template<std::size_t I>
requires(I < 4)
[[nodiscard, GREX_ALWAYS_INLINE]] inline f64x4 broadcast_lane(f64x4 v, IndexTag<I> /*idx*/) {
  static constexpr int imm = bcast_lane::splat(I);
  return {.r = _mm256_permute4x64_pd(v.r, imm)};
}

/** `f32`: `vbroadcastss` (lane 0)/`vpermps` + constant load. 1 µop. */
template<std::size_t I>
requires(I < 8)
[[nodiscard, GREX_ALWAYS_INLINE]] inline f32x8 broadcast_lane(f32x8 v, IndexTag<I> /*idx*/) {
  if constexpr (I == 0) {
    return {.r = _mm256_broadcastss_ps(_mm256_castps256_ps128(v.r))};
  } else {
    return {.r = _mm256_permutevar8x32_ps(v.r, _mm256_set1_epi32(I))};
  }
}

/** `i64`/`u64`: `vpermq`. 1 µop. */
template<std::size_t I, Int64 T>
requires(I < 4)
[[nodiscard, GREX_ALWAYS_INLINE]] inline NativeVector<T, 4> broadcast_lane(NativeVector<T, 4> v,
                                                                           IndexTag<I> /*idx*/) {
  static constexpr int imm = bcast_lane::splat(I);
  return {.r = _mm256_permute4x64_epi64(v.r, imm)};
}

/** `i32`/`u32`: `vpbroadcastd` (lane 0)/`vpermd` + constant load. 1 µop. */
template<std::size_t I, Int32 T>
requires(I < 8)
[[nodiscard, GREX_ALWAYS_INLINE]] inline NativeVector<T, 8> broadcast_lane(NativeVector<T, 8> v,
                                                                           IndexTag<I> /*idx*/) {
  if constexpr (I == 0) {
    return {.r = _mm256_broadcastd_epi32(_mm256_castsi256_si128(v.r))};
  } else {
    return {.r = _mm256_permutevar8x32_epi32(v.r, _mm256_set1_epi32(I))};
  }
}

/**
 * `i16`/`u16`:
 * - lane 0: `vpbroadcastw`. 1 µop.
 * - level 3: `vpshuf[lh]w` + `vpermq`. 2 µops.
 * - level 4: `vpermw` + constant load. 1 µop.
 */
template<std::size_t I, Any16 T>
requires(I < 16)
[[nodiscard, GREX_ALWAYS_INLINE]] inline NativeVector<T, 16> broadcast_lane(NativeVector<T, 16> v,
                                                                            IndexTag<I> /*idx*/) {
  if constexpr (I == 0) {
    return {.r = _mm256_broadcastw_epi16(_mm256_castsi256_si128(v.r))};
  } else if constexpr (x86_64_level >= 4) {
    return {.r = _mm256_permutexvar_epi16(_mm256_set1_epi16(static_cast<i16>(I)), v.r)};
  } else {
    // Broadcast the word within its qword (in-lane), then that qword.
    static constexpr int w = bcast_lane::splat(I % 4);
    static constexpr int q = bcast_lane::splat(I / 4);
    if constexpr ((I / 4) % 2 == 0) {
      return {.r = _mm256_permute4x64_epi64(_mm256_shufflelo_epi16(v.r, w), q)};
    } else {
      return {.r = _mm256_permute4x64_epi64(_mm256_shufflehi_epi16(v.r, w), q)};
    }
  }
}

/**
 * `i8`/`u8`:
 * - lane 0: `vpbroadcastb`. 1 µop.
 * - VBMI: `vpermb` + constant load. 1 µop.
 * - lanes 1-15: `vpsrldq` + `vpbroadcastb`. 2 µops.
 * - lane 16: `vextracti128` + `vpbroadcastb`. 2 µops.
 * - lanes 17-31: `vpshufb` + constant load + `vpermq`. 2 µops.
 */
template<std::size_t I, Int8 T>
requires(I < 32)
[[nodiscard, GREX_ALWAYS_INLINE]] inline NativeVector<T, 32> broadcast_lane(NativeVector<T, 32> v,
                                                                            IndexTag<I> /*idx*/) {
  if constexpr (I == 0) {
    return {.r = _mm256_broadcastb_epi8(_mm256_castsi256_si128(v.r))};
  } else if constexpr (x86_64_level >= 4 && has_avx512vbmi) {
    return {.r = _mm256_permutexvar_epi8(_mm256_set1_epi8(static_cast<i8>(I)), v.r)};
  } else if constexpr (I < 16) {
    const auto shifted = _mm_srli_si128(_mm256_castsi256_si128(v.r), I);
    return {.r = _mm256_broadcastb_epi8(shifted)};
  } else if constexpr (I == 16) {
    return {.r = _mm256_broadcastb_epi8(_mm256_extracti128_si256(v.r, 1))};
  } else {
    const auto shuffled = _mm256_shuffle_epi8(v.r, _mm256_set1_epi8(static_cast<i8>(I - 16)));
    return {.r = _mm256_permute4x64_epi64(shuffled, 0xEE)};
  }
}
#endif

#if GREX_X86_64_LEVEL >= 4
//==================================================================================================
// 512-bit (level 4)
//==================================================================================================

/**
 * `f64`: `vbroadcastsd` (lane 0)/`vpermpd` + constant load. 1 µop.
 * Alas, `vpermpd zmm, imm` only permutes within 256-bit halves and is not applicable here.
 */
template<std::size_t I>
requires(I < 8)
[[nodiscard, GREX_ALWAYS_INLINE]] inline f64x8 broadcast_lane(f64x8 v, IndexTag<I> /*idx*/) {
  if constexpr (I == 0) {
    return {.r = _mm512_broadcastsd_pd(_mm512_castpd512_pd128(v.r))};
  } else {
    return {.r = _mm512_permutexvar_pd(_mm512_set1_epi64(I), v.r)};
  }
}

/** `f32`: `vbroadcastss` (lane 0)/`vpermps` + constant load. 1 µop. */
template<std::size_t I>
requires(I < 16)
[[nodiscard, GREX_ALWAYS_INLINE]] inline f32x16 broadcast_lane(f32x16 v, IndexTag<I> /*idx*/) {
  if constexpr (I == 0) {
    return {.r = _mm512_broadcastss_ps(_mm512_castps512_ps128(v.r))};
  } else {
    return {.r = _mm512_permutexvar_ps(_mm512_set1_epi32(I), v.r)};
  }
}

/**
 * `i64`/`u64`: `vpbroadcastq` (lane 0)/`vpermq` + constant load. 1 µop.
 * Alas, `vpermq zmm, imm` only permutes within 256-bit halves and is not applicable here.
 */
template<std::size_t I, Int64 T>
requires(I < 8)
[[nodiscard, GREX_ALWAYS_INLINE]] inline NativeVector<T, 8> broadcast_lane(NativeVector<T, 8> v,
                                                                           IndexTag<I> /*idx*/) {
  if constexpr (I == 0) {
    return {.r = _mm512_broadcastq_epi64(_mm512_castsi512_si128(v.r))};
  } else {
    return {.r = _mm512_permutexvar_epi64(_mm512_set1_epi64(I), v.r)};
  }
}

/** `i32`/`u32`: `vpbroadcastd` (lane 0)/`vpermd` + constant load. 1 µop. */
template<std::size_t I, Int32 T>
requires(I < 16)
[[nodiscard, GREX_ALWAYS_INLINE]] inline NativeVector<T, 16> broadcast_lane(NativeVector<T, 16> v,
                                                                            IndexTag<I> /*idx*/) {
  if constexpr (I == 0) {
    return {.r = _mm512_broadcastd_epi32(_mm512_castsi512_si128(v.r))};
  } else {
    return {.r = _mm512_permutexvar_epi32(_mm512_set1_epi32(I), v.r)};
  }
}

/** `i16`/`u16`: `vpbroadcastw` (lane 0)/`vpermw` + constant load. 1 µop. */
template<std::size_t I, Any16 T>
requires(I < 32)
[[nodiscard, GREX_ALWAYS_INLINE]] inline NativeVector<T, 32> broadcast_lane(NativeVector<T, 32> v,
                                                                            IndexTag<I> /*idx*/) {
  if constexpr (I == 0) {
    return {.r = _mm512_broadcastw_epi16(_mm512_castsi512_si128(v.r))};
  } else {
    return {.r = _mm512_permutexvar_epi16(_mm512_set1_epi16(static_cast<i16>(I)), v.r)};
  }
}

/**
 * `i8`/`u8`:
 * - lane 0: `vpbroadcastb`. 1 µop.
 * - VBMI: `vpermb` + constant load. 1 µop.
 * - lanes 1-15: `vpsrldq` + `vpbroadcastb`. 2 µops.
 * - lanes 16, 32, 48: `vextracti32x4` + `vpbroadcastb`. 2 µops.
 * - other lanes: `vshufi32x4` + `vpshufb` + constant load. 2 µops.
 */
template<std::size_t I, Int8 T>
requires(I < 64)
[[nodiscard, GREX_ALWAYS_INLINE]] inline NativeVector<T, 64> broadcast_lane(NativeVector<T, 64> v,
                                                                            IndexTag<I> /*idx*/) {
  if constexpr (I == 0) {
    return {.r = _mm512_broadcastb_epi8(_mm512_castsi512_si128(v.r))};
  } else if constexpr (has_avx512vbmi) {
    return {.r = _mm512_permutexvar_epi8(_mm512_set1_epi8(static_cast<i8>(I)), v.r)};
  } else if constexpr (I < 16) {
    const auto shuffled = _mm_srli_si128(_mm512_castsi512_si128(v.r), I);
    return {.r = _mm512_broadcastb_epi8(shuffled)};
  } else if constexpr (I % 16 == 0) {
    return {.r = _mm512_broadcastb_epi8(_mm512_extracti32x4_epi32(v.r, I / 16))};
  } else {
    // Copy the 128-bit block holding the byte to all four blocks, then pshufb.
    static constexpr int blk = bcast_lane::splat(I / 16);
    const auto shuffled = _mm512_shuffle_i32x4(v.r, v.r, blk);
    return {.r = _mm512_shuffle_epi8(shuffled, _mm512_set1_epi8(static_cast<i8>(I % 16)))};
  }
}
#endif
} // namespace grex::backend

#include "grex/backend/shared/operations/broadcast-lane.hpp" // IWYU pragma: keep

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_BROADCAST_LANE_HPP
