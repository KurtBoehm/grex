// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_CONVERT_128_VECTOR_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_CONVERT_128_VECTOR_HPP

#include <immintrin.h>

#include "grex/backend/x86/instruction-sets.hpp"
#include "grex/backend/x86/operations/convert/base.hpp"

#if GREX_X86_64_LEVEL == 1
#include "grex/backend/macros/base.hpp"
#include "grex/backend/macros/math.hpp"
#include "grex/backend/x86/macros/types.hpp"
#include "grex/backend/x86/types.hpp"
#include "grex/base.hpp"
#endif
#if GREX_X86_64_LEVEL < 4
#include "grex/backend/base.hpp"
#include "grex/backend/x86/operations/extract.hpp"
#endif

namespace grex::backend {
//////////////////////
// Integer widening //
//////////////////////

#if GREX_X86_64_LEVEL >= 2
// On x86-64-v2+, use the generic EPUI helper (pmovzx/pmovsx) for all widening conversions.
#define GREX_CVT_IMPL_Ux2 GREX_CVT_INTRINSIC_EPUI
#define GREX_CVT_IMPL_HALFINCR GREX_CVT_INTRINSIC_EPUI
#else
// Widening by ×2 for unsigned integers:
// [a0, a1, ...] (SRCBITS) → [a0, 0, a1, 0, ...] (DSTBITS = 2 * SRCBITS) via unpacklo and zero.
#define GREX_CVT_IMPL_Ux2(DSTKIND, DSTBITS, SRCKIND, SRCBITS, SIZE, BITPREFIX, REGISTERBITS) \
  return {.r = _mm_unpacklo_epi##SRCBITS(v.registr(), _mm_setzero_si128())};

// Widening by >2:
//  1) cast to an intermediate integer type (Half) using a doubled lane count,
//  2) cast from that intermediate type to the final destination.
#define GREX_CVT_IMPL_HALFINCR(DSTKIND, DSTBITS, SRCKIND, SRCBITS, SIZE, BITPREFIX, REGISTERBITS) \
  using Double = GREX_VECTOR_TYPE(SRCKIND, SRCBITS, GREX_MULTIPLY(SIZE, 2)); \
  using Half = GREX_CAT(DSTKIND, GREX_DIVIDE(DSTBITS, 2)); \
  const __m128i half = convert(Double{v.registr()}, type_tag<Half>).r; \
  return convert(GREX_VECTOR_TYPE(DSTKIND, GREX_DIVIDE(DSTBITS, 2), SIZE){half}, \
                 type_tag<DSTKIND##DSTBITS>);
#endif

#if GREX_X86_64_LEVEL >= 2
// Direct pmovsx/pmovzx implementations where available.
// Double integer size.
#define GREX_CVT_IMPL_i16_i8_8 GREX_CVT_INTRINSIC_EPUI
#define GREX_CVT_IMPL_i32_i16_4 GREX_CVT_INTRINSIC_EPUI
#define GREX_CVT_IMPL_i64_i32_2 GREX_CVT_INTRINSIC_EPUI
#define GREX_CVT_IMPL_u16_u8_8 GREX_CVT_INTRINSIC_EPUI
#define GREX_CVT_IMPL_u32_u16_4 GREX_CVT_INTRINSIC_EPUI
#define GREX_CVT_IMPL_u64_u32_2 GREX_CVT_INTRINSIC_EPUI
// Quadruple integer size (8→32, 16→64).
#define GREX_CVT_IMPL_i32_i8_4 GREX_CVT_INTRINSIC_EPUI
#define GREX_CVT_IMPL_u32_u8_4 GREX_CVT_INTRINSIC_EPUI
#define GREX_CVT_IMPL_i64_i16_2 GREX_CVT_INTRINSIC_EPUI
#define GREX_CVT_IMPL_u64_u16_2 GREX_CVT_INTRINSIC_EPUI
// Octuple integer size (8→64).
#define GREX_CVT_IMPL_i64_i8_2 GREX_CVT_INTRINSIC_EPUI
#define GREX_CVT_IMPL_u64_u8_2 GREX_CVT_INTRINSIC_EPUI
#else
// i8→i16: duplicate bytes and sign-extend from the high byte via arithmetic shift.
#define GREX_CVT_IMPL_i16_i8_8(...) \
  const __m128i r = v.registr(); \
  const __m128i unpacklo = _mm_unpacklo_epi8(r, r); \
  return {.r = _mm_srai_epi16(unpacklo, 8)};

// i16→i32: duplicate 16-bit values and sign-extend by shifting the high word into position.
#define GREX_CVT_IMPL_i32_i16_4(...) \
  return {.r = _mm_srai_epi32(_mm_unpacklo_epi16(v.registr(), v.registr()), 16)};

// i32→i64: build a per-lane sign mask via arithmetic shift, then interleave value/sign.
#define GREX_CVT_IMPL_i64_i32_2(...) \
  return {.r = _mm_unpacklo_epi32(v.registr(), _mm_srai_epi32(v.registr(), 31))};

// Unsigned ×2 widening: zero-extend via the generic Ux2 helper.
#define GREX_CVT_IMPL_u16_u8_8 GREX_CVT_IMPL_Ux2
#define GREX_CVT_IMPL_u32_u16_4 GREX_CVT_IMPL_Ux2
#define GREX_CVT_IMPL_u64_u32_2 GREX_CVT_IMPL_Ux2

// Larger widening steps fall back to recursive half-step widening.
// Quadruple integer size.
#define GREX_CVT_IMPL_i32_i8_4 GREX_CVT_IMPL_HALFINCR
#define GREX_CVT_IMPL_u32_u8_4 GREX_CVT_IMPL_HALFINCR
#define GREX_CVT_IMPL_i64_i16_2 GREX_CVT_IMPL_HALFINCR
#define GREX_CVT_IMPL_u64_u16_2 GREX_CVT_IMPL_HALFINCR
// Octuple integer size.
#define GREX_CVT_IMPL_i64_i8_2 GREX_CVT_IMPL_HALFINCR
#define GREX_CVT_IMPL_u64_u8_2 GREX_CVT_IMPL_HALFINCR
#endif

////////////////////////
// Integer truncation //
////////////////////////

// Truncation drops the high bits. Signedness does not matter for the bit pattern.

#if GREX_X86_64_LEVEL >= 4
// Prefer dedicated narrowing intrinsics where available.
#define GREX_CVT_IMPL_u8_u16_8 GREX_CVT_INTRINSIC_EPI
#define GREX_CVT_IMPL_u16_u32_4 GREX_CVT_INTRINSIC_EPI
#define GREX_CVT_IMPL_u16_u32_2 GREX_CVT_INTRINSIC_EPI
#define GREX_CVT_IMPL_u32_u64_2 GREX_CVT_INTRINSIC_EPI
#define GREX_CVT_IMPL_u8_u32_4 GREX_CVT_INTRINSIC_EPI
#define GREX_CVT_IMPL_u16_u64_2 GREX_CVT_INTRINSIC_EPI
#define GREX_CVT_IMPL_u8_u64_2 GREX_CVT_INTRINSIC_EPI
#elif GREX_X86_64_LEVEL >= 2
// Use pshufb-based element selection patterns for truncation.

// u16x8 → u8x8: keep every second byte of the low 8 elements.
#define GREX_CVT_IMPL_u8_u16_8(...) \
  const auto m = _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1); \
  return SubVector<u8, 8, 16>{_mm_shuffle_epi8(v.r, m)};

// u32x4 → u16x4: select the low 2 bytes from each 32-bit lane.
#define GREX_CVT_IMPL_u16_u32_4(DSTKIND, DSTBITS, SRCKIND, SRCBITS, SIZE, ...) \
  const auto m = _mm_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1); \
  return SubVector<u16, SIZE, 8>{_mm_shuffle_epi8(v.registr(), m)};

// 2-lane variant reuses the same shuffle pattern.
#define GREX_CVT_IMPL_u16_u32_2 GREX_CVT_IMPL_u16_u32_4

// u64x2 → u32x2: keep the low 32 bits from each 64-bit lane via insertps.
#define GREX_CVT_IMPL_u32_u64_2(...) \
  const __m128 i = _mm_insert_ps(_mm_castsi128_ps(v.r), _mm_castsi128_ps(v.r), 0b10011100); \
  return SubVector<u32, 2, 4>{_mm_castps_si128(i)};

// u32x4 → u8x4: keep the lowest byte of each 32-bit element.
#define GREX_CVT_IMPL_u8_u32_4(...) \
  const auto m = _mm_setr_epi8(0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1); \
  return SubVector<u8, 4, 16>{_mm_shuffle_epi8(v.r, m)};

// u64x2 → u16x2: keep the low 16 bits of each 64-bit element.
#define GREX_CVT_IMPL_u16_u64_2(...) \
  const auto m = _mm_setr_epi8(0, 1, 8, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1); \
  return SubVector<u16, 2, 8>{_mm_shuffle_epi8(v.r, m)};

// u64x2 → u8x2: keep the lowest byte from each 64-bit element.
#define GREX_CVT_IMPL_u8_u64_2(...) \
  const auto m = _mm_setr_epi8(0, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1); \
  return SubVector<u8, 2, 16>{_mm_shuffle_epi8(v.r, m)};
#else
// x86-64-v1: emulate truncation using older shuffle and pack instructions.

// u16x8 → u8x8: mask out upper 8 bits and use saturated pack as a truncation helper.
#define GREX_CVT_IMPL_u8_u16_8(...) \
  const auto r = _mm_packus_epi16(_mm_and_si128(v.r, _mm_set1_epi16(0xFF)), _mm_setzero_si128()); \
  return SubVector<u8, 8, 16>{r};

// super-native variant: u16x16 → u8x16 across both 128-bit halves.
#define GREX_CVT_IMPL_u8_u16_16(...) \
  const auto m = _mm_set1_epi16(0xFF); \
  const auto r = _mm_packus_epi16(_mm_and_si128(v.lower.r, m), _mm_and_si128(v.upper.r, m)); \
  return u8x16{r};

// u32x4 → u16x4: drop the upper half of each 32-bit lane via 16-bit shuffles.
#define GREX_CVT_IMPL_u16_u32_4(...) \
  /* lo = [u16(v[0]), u16(v[1]), -, ...] */ \
  const auto lo = _mm_shufflelo_epi16(v.r, 0b1000); \
  /* hi = [u16(v[0]), u16(v[1]), -, -, u16(v[2]), u16(v[3]), -, -] */ \
  const auto hi = _mm_shufflehi_epi16(lo, 0b1000); \
  /* sh = [u16(v[0]), u16(v[1]), u16(v[2]), u16(v[3]), -, -, -, -] */ \
  return SubVector<u16, 4, 8>{_mm_shuffle_epi32(hi, 0b1000)};

// u32x2 → u16x2: same pattern restricted to the low two lanes.
#define GREX_CVT_IMPL_u16_u32_2(...) \
  /* lo = [u16(v[0]), u16(v[1]), -, ...] */ \
  return SubVector<u16, 2, 8>{_mm_shufflelo_epi16(v.registr(), 0b1000)};

// u64x2 → u32x2: cast via shufps, keeping the low 32 bits of each 64-bit lane.
#define GREX_CVT_IMPL_u32_u64_2(...) \
  /* [u32(v[0]), u32(v[1]), 0, 0] */ \
  const __m128 sh = _mm_shuffle_ps(_mm_castsi128_ps(v.r), _mm_setzero_ps(), 0b1000); \
  return SubVector<u32, 2, 4>{_mm_castps_si128(sh)};

// u32x4 → u8x4 via intermediate u16/u8 packing.
#define GREX_CVT_IMPL_u8_u32_4(...) \
  /* [u8(v[0]), u8(v[1]), u8(v[2]), u8(v[3])] as u32x4 (masked low byte) */ \
  const __m128i vu32 = _mm_and_si128(v.r, _mm_set1_epi32(0xFF)); \
  /* [u8(v[0..3]), 0, 0, 0, 0] as u16x8 */ \
  const __m128i vu16 = _mm_packus_epi16(vu32, _mm_setzero_si128()); \
  /* [u8(v[0..3]), 0, …, 0] as u8x16 */ \
  const auto r = _mm_packus_epi16(vu16, _mm_setzero_si128()); \
  return SubVector<u8, 4, 16>{r};

// super-native variants for u32 narrowing at level 1, using two or four 128-bit pieces.
#define GREX_CVT_IMPL_u8_u32_8(...) \
  const __m128i mask = _mm_set1_epi32(0xFF); \
  const __m128i lu32 = _mm_and_si128(v.lower.r, mask); \
  const __m128i hu32 = _mm_and_si128(v.upper.r, mask); \
  const __m128i vu16 = _mm_packus_epi16(lu32, hu32); \
  const auto r = _mm_packus_epi16(vu16, _mm_setzero_si128()); \
  return SubVector<u8, 8, 16>{r};

#define GREX_CVT_IMPL_u8_u32_16(...) \
  const __m128i mask = _mm_set1_epi32(0xFF); \
  const __m128i u32i0 = _mm_and_si128(v.lower.lower.r, mask); \
  const __m128i u32i1 = _mm_and_si128(v.lower.upper.r, mask); \
  const __m128i u32i2 = _mm_and_si128(v.upper.lower.r, mask); \
  const __m128i u32i3 = _mm_and_si128(v.upper.upper.r, mask); \
  const __m128i lu16 = _mm_packus_epi16(u32i0, u32i1); \
  const __m128i hu16 = _mm_packus_epi16(u32i2, u32i3); \
  const auto r = _mm_packus_epi16(lu16, hu16); \
  return u8x16{r};

// sub-native variant: u64x2 → u16x2 via u32x2 intermediate.
#define GREX_CVT_IMPL_u16_u64_2(...) \
  /* lo = [u32(v0), u32(v1), -, -] as u32x4 */ \
  const auto lo = _mm_shuffle_epi32(v.r, 0b1000); \
  /* [u16(v0), u16(v1), -, …, -] */ \
  return SubVector<u16, 2, 8>{_mm_shufflelo_epi16(lo, 0b1000)};

// u64x2 → u8x2 via staged narrowing through 32- and 16-bit widths.
#define GREX_CVT_IMPL_u8_u64_2(...) \
  /* [u8(v[0]), u8(v[1])] as u64x2 (masked low byte of each lane) */ \
  const __m128i vu64 = _mm_and_si128(v.r, _mm_set1_epi64x(0xFF)); \
  /* [u8(v[0]), u8(v[1]), 0, 0] as u32x4 */ \
  const __m128i vu32 = _mm_shuffle_epi32(vu64, 0b11011000); \
  /* [u8(v[0]), u8(v[1]), 0, …, 0] as u16x8 */ \
  const __m128i vu16 = _mm_shufflelo_epi16(vu32, 0b11011000); \
  /* final pack to u8x16, only first two bytes are used */ \
  return SubVector<u8, 2, 16>{_mm_packus_epi16(vu16, _mm_setzero_si128())};
#endif

#if GREX_X86_64_LEVEL < 3
// 256-bit super-native widening from 64→32 bit via two 128-bit halves.
#define GREX_CVT_IMPL_i32_i64_4(DSTKIND, ...) \
  /* take low 32 bits from lower/upper halves and merge into a single 128-bit register */ \
  /* [u32(v[0]), u32(v[1]), 0, 0] */ \
  const __m128 sh = \
    _mm_shuffle_ps(_mm_castsi128_ps(v.lower.r), _mm_castsi128_ps(v.upper.r), 0b1000'1000); \
  return DSTKIND##32x4 {_mm_castps_si128(sh)};

#define GREX_CVT_IMPL_i32_u64_4 GREX_CVT_IMPL_i32_i64_4
#define GREX_CVT_IMPL_u32_i64_4 GREX_CVT_IMPL_i32_i64_4
#define GREX_CVT_IMPL_u32_u64_4 GREX_CVT_IMPL_i32_i64_4
#endif

/////////////////////////////////////
// Floating-point ↔ floating-point //
/////////////////////////////////////

// f32↔f64 conversions use native cvt intrinsics.
#define GREX_CVT_IMPL_f64_f32_2 GREX_CVT_INTRINSIC_EPU
#define GREX_CVT_IMPL_f32_f64_2 GREX_CVT_INTRINSIC_EPU

//////////////////////////////
// Integer → floating-point //
//////////////////////////////
// SMALLI2F helpers handle small integer types via i32 intermediates.

//                   //
// Conversion to f64 //
//                   //
#if GREX_X86_64_LEVEL >= 4
// Direct packed cvt intrinsics for 32- and 64-bit integer sources.
#define GREX_CVT_IMPL_f64_i64_2 GREX_CVT_INTRINSIC_EPU
#define GREX_CVT_IMPL_f64_u64_2 GREX_CVT_INTRINSIC_EPU
#define GREX_CVT_IMPL_f64_u32_2 GREX_CVT_INTRINSIC_EPU
#else
// u64→f64: split each 64-bit value into low/high 32-bit parts, encode each as a f64 using exponent
// fiddling, then recombine. This is exact for all u64.
#define GREX_CVT_IMPL_f64_u64_2(...) \
  /* isolate the lower 32 bits of each 64-bit value */ \
  const __m128i lu32 = _mm_and_si128(v.r, _mm_set1_epi64x(0xFFFFFFFF)); \
  /* encode lower 32 bits as mantissa with exponent 52; hidden bit adds 2^52 */ \
  const __m128d lf64 = _mm_castsi128_pd(_mm_or_si128(lu32, _mm_set1_epi64x(0x4330000000000000))); \
  /* shift away lower 32 bits, leaving upper 32 bits */ \
  const __m128i hu32 = _mm_srli_epi64(v.r, 32); \
  /* encode high 32 bits with exponent 84 = 52 + 32; hidden bit adds 2^84 */ \
  const __m128d hf64 = _mm_castsi128_pd(_mm_or_si128(hu32, _mm_set1_epi64x(0x4530000000000000))); \
  /* subtract 2^84 * (1 + 2^-32) = 2^84 + 2^52 to cancel hidden bits */ \
  const __m128d hsub = _mm_sub_pd(hf64, _mm_castsi128_pd(_mm_set1_epi64x(0x4530000000100000))); \
  /* sum both contributions */ \
  return {.r = _mm_add_pd(lf64, hsub)};

// i64 → f64: convert each lane with scalar cvtsi2sd and repack.
#define GREX_CVT_IMPL_f64_i64_2(...) \
  const __m128d lf64 = _mm_cvtsi64_sd(_mm_undefined_pd(), _mm_cvtsi128_si64(v.r)); \
  const __m128d hf64 = _mm_cvtsi64_sd(_mm_undefined_pd(), extract(v, 1)); \
  return {.r = _mm_unpacklo_pd(lf64, hf64)};

// u32 → f64: pack into mantissa, set exponent to 52, then subtract the hidden bit.
// Exact for all u32.
#define GREX_CVT_IMPL_f64_u32_2(...) \
  /* [u64(v[0]), u64(v[1])], integer bits in low mantissa */ \
  const __m128 unpack64 = _mm_unpacklo_ps(_mm_castsi128_ps(v.full.r), _mm_setzero_ps()); \
  /* exponent 52 encoded in IEEE-754 f64 */ \
  const __m128i exponent = _mm_set1_epi64x(0x4330000000000000); \
  /* set exponent to 52 → value = v + 2^52 due to hidden bit */ \
  const __m128i iv64 = _mm_or_si128(_mm_castps_si128(unpack64), exponent); \
  /* subtract hidden bit to recover original integer */ \
  return {.r = _mm_sub_pd(_mm_castsi128_pd(iv64), _mm_castsi128_pd(exponent))};
#endif

// i32→f64 uses standard cvt; smaller integer types go via i32 intermediates.
#define GREX_CVT_IMPL_f64_i32_2 GREX_CVT_INTRINSIC_EPU
#define GREX_CVT_IMPL_f64_i16_2 GREX_CVT_IMPL_SMALLI2F
#define GREX_CVT_IMPL_f64_u16_2 GREX_CVT_IMPL_SMALLI2F
#define GREX_CVT_IMPL_f64_i8_2 GREX_CVT_IMPL_SMALLI2F
#define GREX_CVT_IMPL_f64_u8_2 GREX_CVT_IMPL_SMALLI2F

//                   //
// Conversion to f32 //
//                   //
#if GREX_X86_64_LEVEL >= 4
// Direct packed cvt for 64-bit integer sources.
#define GREX_CVT_IMPL_f32_i64_2 GREX_CVT_INTRINSIC_EPU
#define GREX_CVT_IMPL_f32_u64_2 GREX_CVT_INTRINSIC_EPU
#else
// i64 → f32: scalar-convert each lane and repack to a 2-lane sub-vector.
#define GREX_CVT_IMPL_f32_i64_2(...) \
  const __m128 lf32 = _mm_cvtsi64_ss(_mm_undefined_ps(), _mm_cvtsi128_si64(v.r)); \
  const __m128 hf32 = _mm_cvtsi64_ss(_mm_undefined_ps(), extract(v, 1)); \
  return SubVector<f32, 2, 4>{_mm_unpacklo_ps(lf32, hf32)};

// u64 → f32: compute v/2 (rounded to even) as i64, convert to f32, then multiply by 2.
#define GREX_CVT_IMPL_f32_u64_2(...) \
  /* v / 2 with truncation */ \
  const __m128i thalf = _mm_srli_epi64(v.r, 1); \
  /* v % 2 */ \
  const __m128i mod2 = _mm_and_si128(v.r, _mm_set1_epi64x(1)); \
  /* v / 2 with rounding to even */ \
  const __m128i half = _mm_or_si128(thalf, mod2); \
  /* (v / 2) as f32 */ \
  const __m128 fhalf = convert(i64x2{half}, type_tag<f32>).full.r; \
  /* restore v by multiplying by 2 in f32 */ \
  return SubVector<f32, 2, 4>{_mm_add_ps(fhalf, fhalf)};
#endif

// u32→f32
#if GREX_X86_64_LEVEL >= 4
#define GREX_CVT_IMPL_f32_u32_4 GREX_CVT_INTRINSIC_EPU
#elif GREX_X86_64_LEVEL >= 2
// x86-64-v2: split into low/high 16 bits, encode via exponent fiddling, and recombine.
// This avoids losing precision for large unsigned values.
// TODO This uses pblendw, which is comparatively slow.
#define GREX_CVT_IMPL_f32_u32_4(...) \
  /* combine exponent 23 with lower 16 bits; hidden bit adds 2^23 */ \
  const __m128i a = _mm_blend_epi16(_mm_set1_epi32(0x4B000000), v.r, 0b01010101); \
  /* shift away lower 16 bits, leaving high 16 bits */ \
  const __m128i b = _mm_srli_epi32(v.r, 16); \
  /* set exponent to 39 = 23 + 16 for upper part; hidden bit adds 2^39 */ \
  const __m128i c = _mm_blend_epi16(b, _mm_set1_epi32(0x53000000), 0b10101010); \
  /* subtract 2^39 * (1 + 2^-16) = 2^39 + 2^23 to cancel hidden bits */ \
  const __m128 d = _mm_sub_ps(_mm_castsi128_ps(c), _mm_castsi128_ps(_mm_set1_epi32(0x53000080))); \
  /* sum lower and corrected upper parts */ \
  return {.r = _mm_add_ps(_mm_castsi128_ps(a), d)};
#else
// Same idea as above but without pblendw: explicit masking and OR.
#define GREX_CVT_IMPL_f32_u32_4(...) \
  /* isolate the lower 16 bits of each 32-bit value */ \
  const __m128i lu16 = _mm_and_si128(v.r, _mm_set1_epi32(0xFFFF)); \
  /* add exponent 23; hidden bit adds 2^23 */ \
  const __m128 lf32 = _mm_castsi128_ps(_mm_or_si128(lu16, _mm_set1_epi32(0x4B000000))); \
  /* shift away the lower 16 bits, leaving upper 16 bits */ \
  const __m128i hu16 = _mm_srli_epi32(v.r, 16); \
  /* exponent 39 = 23 + 16; hidden bit adds 2^39 */ \
  const __m128 hf32 = _mm_castsi128_ps(_mm_or_si128(hu16, _mm_set1_epi32(0x53000000))); \
  /* subtract 2^39 * (1 + 2^-16) = 2^39 + 2^23 to cancel hidden bits */ \
  const __m128 hsub = _mm_sub_ps(hf32, _mm_castsi128_ps(_mm_set1_epi32(0x53000080))); \
  /* add both parts together */ \
  return {.r = _mm_add_ps(lf32, hsub)};
#endif

// i32→f32 and smaller integer types use intrinsics and widening to 32 bits if needed.
#define GREX_CVT_IMPL_f32_i32_4 GREX_CVT_INTRINSIC_EPU
#define GREX_CVT_IMPL_f32_i16_4 GREX_CVT_IMPL_SMALLI2F
#define GREX_CVT_IMPL_f32_u16_4 GREX_CVT_IMPL_SMALLI2F
#define GREX_CVT_IMPL_f32_i8_4 GREX_CVT_IMPL_SMALLI2F
#define GREX_CVT_IMPL_f32_u8_4 GREX_CVT_IMPL_SMALLI2F

//////////////////////////////
// Floating-point → integer //
//////////////////////////////
// CVTT variants use truncation semantics (round toward zero).

//                     //
// Conversion from f64 //
//                     //
#if GREX_X86_64_LEVEL >= 4
// Direct packed cvtt intrinsics for 32- and 64-bit integer destinations.
#define GREX_CVT_IMPL_i64_f64_2 GREX_CVTT_INTRINSIC_EPU
#define GREX_CVT_IMPL_u64_f64_2 GREX_CVTT_INTRINSIC_EPU
#define GREX_CVT_IMPL_u32_f64_2 GREX_CVTT_INTRINSIC_EPU
#else
// f64x2 → i64x2 via scalar truncating conversions for both lanes.
#define GREX_CVT_IMPL_i64_f64_2(...) \
  const i64 v0 = _mm_cvttsd_si64(v.r); \
  const i64 v1 = _mm_cvttsd_si64(_mm_unpackhi_pd(v.r, v.r)); \
  return {.r = _mm_set_epi64x(v1, v0)};

// f64x2 → u64x2: offset by 2^63 to map u64 range into i64, convert, then select
// between plain and offset result based on the sign/overflow flag.
#define GREX_CVT_IMPL_u64_f64_2(...) \
  /* v - [2^63, 2^63] as f64x2, mapping u64→i64 domain */ \
  const __m128d voff = \
    _mm_sub_pd(v.registr(), _mm_castsi128_pd(_mm_set1_epi64x(0x43e0000000000000))); \
  /* [i64(v[i] - 2^63)] */ \
  const __m128i oi64 = convert(f64x2{voff}, type_tag<i64>).r; \
  /* [i64(v[i])] with 0x8000000000000000 = 2^63 in place of values ≥ 2^63 */ \
  const __m128i vi64 = convert(v, type_tag<i64>).r; \
  /* sign[i] is true if v[i] < 0 or v[i] ≥ 2^63 */ \
  const __m128i sign = _mm_shuffle_epi32(_mm_srai_epi32(vi64, 31), 0b11110101); \
  /* 0 if 0 ≤ v[i] < 2^63, i64(v[i] - 2^63) if v[i] ≥ 2^63 */ \
  const __m128i mi64 = _mm_and_si128(oi64, sign); \
  /* i64(v[i]) = u64(v[i]) if 0 ≤ v[i] < 2^63 */ \
  /* i64(v[i] - 2^63) | 2^63 = u64(v[i]) if v[i] ≥ 2^63 */ \
  return {.r = _mm_or_si128(vi64, mi64)};

// f64x2 → u32x2: apply the same offset trick at 32-bit width.
#define GREX_CVT_IMPL_u32_f64_2(...) \
  /* [i32(v[0]), i32(v[1]), -, -] with 0x80000000 = 2^31 in place of values ≥ 2^31 */ \
  const __m128i vi32 = _mm_cvttpd_epi32(v.r); \
  /* v - [2^31, 2^31] as f64x2, mapping u32→i32 domain */ \
  const __m128d voff = _mm_sub_pd(v.r, _mm_castsi128_pd(_mm_set1_epi64x(0x41E0000000000000))); \
  /* [i32(v[i] - 2^31), -, -] */ \
  const __m128i oi32 = _mm_cvttpd_epi32(voff); \
  /* sign[i] is true if v[i] < 0 or v[i] ≥ 2^31 */ \
  const __m128i sign = _mm_srai_epi32(vi32, 31); \
  /* 0 if 0 ≤ v[i] < 2^31, i32(v[i] - 2^31) if v[i] ≥ 2^31 */ \
  const __m128i mi32 = _mm_and_si128(oi32, sign); \
  /* i32(v[i]) = u32(v[i]) if 0 ≤ v[i] < 2^31 */ \
  /* i32(v[i] - 2^31) | 2^31 = u32(v[i]) if v[i] ≥ 2^31 */ \
  return SubVector<u32, 2, 4>{_mm_or_si128(vi32, mi32)};
#endif

// Smaller integer destinations via a temporary i32 conversion.
#define GREX_CVT_IMPL_i32_f64_2 GREX_CVTT_INTRINSIC_EPU
#define GREX_CVT_IMPL_i16_f64_2 GREX_CVT_IMPL_F2SMALLI
#define GREX_CVT_IMPL_u16_f64_2 GREX_CVT_IMPL_F2SMALLI
#define GREX_CVT_IMPL_i8_f64_2 GREX_CVT_IMPL_F2SMALLI
#define GREX_CVT_IMPL_u8_f64_2 GREX_CVT_IMPL_F2SMALLI

// Conversion from f32.
#if GREX_X86_64_LEVEL >= 4
#define GREX_CVT_IMPL_i64_f32_2 GREX_CVTT_INTRINSIC_EPU
#define GREX_CVT_IMPL_u64_f32_2 GREX_CVTT_INTRINSIC_EPU
#define GREX_CVT_IMPL_u32_f32_4 GREX_CVTT_INTRINSIC_EPU
#else
// f32x2 → i64x2: scalar cvttss2si per lane followed by repack.
#define GREX_CVT_IMPL_i64_f32_2(...) \
  const i64 li64 = _mm_cvttss_si64(v.registr()); \
  const __m128 hf32 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v.registr()), 0b01)); \
  const i64 hi64 = _mm_cvttss_si64(hf32); \
  return {.r = _mm_set_epi64x(hi64, li64)};

// f32x2 → u64x2: offset by 2^63, convert as i64, then select offset result where needed.
#define GREX_CVT_IMPL_u64_f32_2(...) \
  /* v - [2^63, 2^63] as f32x4, mapping u64→i64 domain */ \
  const __m128 voff = _mm_sub_ps(v.registr(), _mm_castsi128_ps(_mm_set1_epi32(0x5f000000))); \
  /* [i64(v[i] - 2^63)] */ \
  const __m128i oi64 = convert(SubVector<f32, 2, 4>{voff}, type_tag<i64>).r; \
  /* [i64(v[i])] with 0x8000000000000000 = 2^63 in place of values ≥ 2^63 */ \
  const __m128i vi64 = convert(v, type_tag<i64>).r; \
  /* sign[i] is true if v[i] < 0 or v[i] ≥ 2^63 */ \
  const __m128i sign = _mm_shuffle_epi32(_mm_srai_epi32(vi64, 31), 0b11110101); \
  /* 0 if 0 ≤ v[i] < 2^63, i64(v[i] - 2^63) if v[i] ≥ 2^63 */ \
  const __m128i mi64 = _mm_and_si128(oi64, sign); \
  /* i64(v[i]) = u64(v[i]) if 0 ≤ v[i] < 2^63 */ \
  /* i64(v[i] - 2^63) | 2^63 = u64(v[i]) if v[i] ≥ 2^63 */ \
  return {.r = _mm_or_si128(vi64, mi64)};

// f32x4 → u32x4: same offset trick at 32-bit width.
#define GREX_CVT_IMPL_u32_f32_4(...) \
  /* [i32(v[0..3])] with 0x80000000 = 2^31 in place of values ≥ 2^31 */ \
  const __m128i vi32 = _mm_cvttps_epi32(v.r); \
  /* v - [2^31, 2^31, 2^31, 2^31] as f32x4, mapping u32→i32 domain */ \
  const __m128 voff = _mm_sub_ps(v.r, _mm_castsi128_ps(_mm_set1_epi32(0x4f000000))); \
  /* [i32(v[i] - 2^31)] */ \
  const __m128i oi32 = _mm_cvttps_epi32(voff); \
  /* sign[i] is true if v[i] < 0 or v[i] ≥ 2^31 */ \
  const __m128i sign = _mm_srai_epi32(vi32, 31); \
  /* 0 if 0 ≤ v[i] < 2^31, i32(v[i] - 2^31) if v[i] ≥ 2^31 */ \
  const __m128i mi32 = _mm_and_si128(oi32, sign); \
  /* i32(v[i]) = u32(v[i]) if 0 ≤ v[i] < 2^31 */ \
  /* i32(v[i] - 2^31) | 2^31 = u32(v[i]) if v[i] ≥ 2^31 */ \
  return {.r = _mm_or_si128(vi32, mi32)};
#endif

// Smaller integer destinations via temporary i32.
#define GREX_CVT_IMPL_i32_f32_4 GREX_CVTT_INTRINSIC_EPU
#define GREX_CVT_IMPL_i16_f32_4 GREX_CVT_IMPL_F2SMALLI
#define GREX_CVT_IMPL_u16_f32_4 GREX_CVT_IMPL_F2SMALLI
#define GREX_CVT_IMPL_i8_f32_4 GREX_CVT_IMPL_F2SMALLI
#define GREX_CVT_IMPL_u8_f32_4 GREX_CVT_IMPL_F2SMALLI

////////////////////////////////
// Macro-driven instantiation //
////////////////////////////////

GREX_CVT_DEF_ALL(_mm, 128)

// Extra variant: u32x2 → u16x2 (sub-native, 128-bit).
GREX_CVT(u, 16, u, 32, 2, _mm, 128)

#if GREX_X86_64_LEVEL == 1
// Level-1 super-native variants: operate on large logical vectors via 128-bit pieces.
GREX_CVT_SUPER(u, 8, u, 16, 16, _mm, 128)
GREX_CVT_SUPER(u, 8, u, 32, 8, _mm, 128)
GREX_CVT_SUPER(u, 8, u, 32, 16, _mm, 128)
#endif

#if GREX_X86_64_LEVEL < 3
// 256-bit super-native widening from 64→32 bit using two 128-bit lanes.
GREX_CVT_SUPER(i, 32, i, 64, 4, _mm, 128)
GREX_CVT_SUPER(i, 32, u, 64, 4, _mm, 128)
GREX_CVT_SUPER(u, 32, i, 64, 4, _mm, 128)
GREX_CVT_SUPER(u, 32, u, 64, 4, _mm, 128)
#endif
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_CONVERT_128_VECTOR_HPP
