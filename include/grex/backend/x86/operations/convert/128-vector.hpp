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
#include "grex/base/defs.hpp"
#endif
#if GREX_X86_64_LEVEL < 4
#include "grex/backend/defs.hpp"
#include "grex/backend/x86/operations/extract.hpp"
#endif

namespace grex::backend {
// Increasing integer size
#if GREX_X86_64_LEVEL >= 2
#define GREX_CVT_IMPL_Ux2 GREX_CVT_INTRINSIC_EPUI
#define GREX_CVT_IMPL_HALFINCR GREX_CVT_INTRINSIC_EPUI
#else
// Doubling unsigned: unpacklo with zero as the upper part
#define GREX_CVT_IMPL_Ux2(DSTKIND, DSTBITS, SRCKIND, SRCBITS, SIZE, BITPREFIX, REGISTERBITS) \
  return {.r = _mm_unpacklo_epi##SRCBITS(v.registr(), _mm_setzero_si128())};
// Recursive case for larger increases: Cast to the next smaller size first, then cast from there
#define GREX_CVT_IMPL_HALFINCR(DSTKIND, DSTBITS, SRCKIND, SRCBITS, SIZE, BITPREFIX, REGISTERBITS) \
  using Double = GREX_VECTOR_TYPE(SRCKIND, SRCBITS, GREX_MULTIPLY(SIZE, 2)); \
  using Half = GREX_CAT(DSTKIND, GREX_DIVIDE(DSTBITS, 2)); \
  const __m128i half = convert(Double{v.registr()}, type_tag<Half>).r; \
  return convert(GREX_VECTOR_TYPE(DSTKIND, GREX_DIVIDE(DSTBITS, 2), SIZE){half}, \
                 type_tag<DSTKIND##DSTBITS>);
#endif
// Double integer size
#if GREX_X86_64_LEVEL >= 2
#define GREX_CVT_IMPL_i16_i8_8 GREX_CVT_INTRINSIC_EPUI
#define GREX_CVT_IMPL_i32_i16_4 GREX_CVT_INTRINSIC_EPUI
#define GREX_CVT_IMPL_i64_i32_2 GREX_CVT_INTRINSIC_EPUI
#define GREX_CVT_IMPL_u16_u8_8 GREX_CVT_INTRINSIC_EPUI
#define GREX_CVT_IMPL_u32_u16_4 GREX_CVT_INTRINSIC_EPUI
#define GREX_CVT_IMPL_u64_u32_2 GREX_CVT_INTRINSIC_EPUI
// Quadruple integer size
#define GREX_CVT_IMPL_i32_i8_4 GREX_CVT_INTRINSIC_EPUI
#define GREX_CVT_IMPL_u32_u8_4 GREX_CVT_INTRINSIC_EPUI
#define GREX_CVT_IMPL_i64_i16_2 GREX_CVT_INTRINSIC_EPUI
#define GREX_CVT_IMPL_u64_u16_2 GREX_CVT_INTRINSIC_EPUI
// Octuple integer size
#define GREX_CVT_IMPL_i64_i8_2 GREX_CVT_INTRINSIC_EPUI
#define GREX_CVT_IMPL_u64_u8_2 GREX_CVT_INTRINSIC_EPUI
#else
// i8→i16: unpacklo with itself, then srai to extend sign
#define GREX_CVT_IMPL_i16_i8_8(...) \
  const __m128i r = v.registr(); \
  const __m128i unpacklo = _mm_unpacklo_epi8(r, r); \
  return {.r = _mm_srai_epi16(unpacklo, 8)};
// i16→i32: duplicate each value and extend the sign using srai
#define GREX_CVT_IMPL_i32_i16_4(...) \
  return {.r = _mm_srai_epi32(_mm_unpacklo_epi16(v.registr(), v.registr()), 16)};
// i32→i64: unpacklo with sign mask computed using srai
#define GREX_CVT_IMPL_i64_i32_2(...) \
  return {.r = _mm_unpacklo_epi32(v.registr(), _mm_srai_epi32(v.registr(), 31))};
#define GREX_CVT_IMPL_u16_u8_8 GREX_CVT_IMPL_Ux2
#define GREX_CVT_IMPL_u32_u16_4 GREX_CVT_IMPL_Ux2
#define GREX_CVT_IMPL_u64_u32_2 GREX_CVT_IMPL_Ux2
// Quadruple integer size
#define GREX_CVT_IMPL_i32_i8_4 GREX_CVT_IMPL_HALFINCR
#define GREX_CVT_IMPL_u32_u8_4 GREX_CVT_IMPL_HALFINCR
#define GREX_CVT_IMPL_i64_i16_2 GREX_CVT_IMPL_HALFINCR
#define GREX_CVT_IMPL_u64_u16_2 GREX_CVT_IMPL_HALFINCR
// Octuple integer size
#define GREX_CVT_IMPL_i64_i8_2 GREX_CVT_IMPL_HALFINCR
#define GREX_CVT_IMPL_u64_u8_2 GREX_CVT_IMPL_HALFINCR
#endif

// Decreasing integer size: Truncation works the same for signed and unsigned types
#if GREX_X86_64_LEVEL >= 4
#define GREX_CVT_IMPL_u8_u16_8 GREX_CVT_INTRINSIC_EPI
#define GREX_CVT_IMPL_u16_u32_4 GREX_CVT_INTRINSIC_EPI
#define GREX_CVT_IMPL_u16_u32_2 GREX_CVT_INTRINSIC_EPI
#define GREX_CVT_IMPL_u32_u64_2 GREX_CVT_INTRINSIC_EPI
#define GREX_CVT_IMPL_u8_u32_4 GREX_CVT_INTRINSIC_EPI
#define GREX_CVT_IMPL_u16_u64_2 GREX_CVT_INTRINSIC_EPI
#define GREX_CVT_IMPL_u8_u64_2 GREX_CVT_INTRINSIC_EPI
#elif GREX_X86_64_LEVEL >= 2
#define GREX_CVT_IMPL_u8_u16_8(...) \
  const auto m = _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1); \
  return SubVector<u8, 8, 16>{_mm_shuffle_epi8(v.r, m)};
#define GREX_CVT_IMPL_u16_u32_4(DSTKIND, DSTBITS, SRCKIND, SRCBITS, SIZE, ...) \
  const auto m = _mm_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1); \
  return SubVector<u16, SIZE, 8>{_mm_shuffle_epi8(v.registr(), m)};
#define GREX_CVT_IMPL_u16_u32_2 GREX_CVT_IMPL_u16_u32_4
#define GREX_CVT_IMPL_u32_u64_2(...) \
  const __m128 i = _mm_insert_ps(_mm_castsi128_ps(v.r), _mm_castsi128_ps(v.r), 0b10011100); \
  return SubVector<u32, 2, 4>{_mm_castps_si128(i)};

#define GREX_CVT_IMPL_u8_u32_4(...) \
  const auto m = _mm_setr_epi8(0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1); \
  return SubVector<u8, 4, 16>{_mm_shuffle_epi8(v.r, m)};
#define GREX_CVT_IMPL_u16_u64_2(...) \
  const auto m = _mm_setr_epi8(0, 1, 8, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1); \
  return SubVector<u16, 2, 8>{_mm_shuffle_epi8(v.r, m)};

#define GREX_CVT_IMPL_u8_u64_2(...) \
  const auto m = _mm_setr_epi8(0, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1); \
  return SubVector<u8, 2, 16>{_mm_shuffle_epi8(v.r, m)};
#else
// u16→u8 on level 1: Mask out upper 8 bits and use a saturated cast
#define GREX_CVT_IMPL_u8_u16_8(...) \
  const auto r = _mm_packus_epi16(_mm_and_si128(v.r, _mm_set1_epi16(0xFF)), _mm_setzero_si128()); \
  return SubVector<u8, 8, 16>{r};
// super-native variant
#define GREX_CVT_IMPL_u8_u16_16(...) \
  const auto m = _mm_set1_epi16(0xFF); \
  const auto r = _mm_packus_epi16(_mm_and_si128(v.lower.r, m), _mm_and_si128(v.upper.r, m)); \
  return u8x16{r};
#define GREX_CVT_IMPL_u16_u32_4(...) \
  /* lo = [u16(v[0]), u16(v[1]), -, ...] */ \
  const auto lo = _mm_shufflelo_epi16(v.r, 0b1000); \
  /* hi = [u16(v[0]), u16(v[1]), -, -, u16(v[2]), u16(v[3]), -, -] */ \
  const auto hi = _mm_shufflehi_epi16(lo, 0b1000); \
  /* sh = [u16(v[0]), u16(v[1]), u16(v[2]), u16(v[3]), -, -, -, -] */ \
  return SubVector<u16, 4, 8>{_mm_shuffle_epi32(hi, 0b1000)};
#define GREX_CVT_IMPL_u16_u32_2(...) \
  /* lo = [u16(v[0]), u16(v[1]), -, ...] */ \
  return SubVector<u16, 2, 8>{_mm_shufflelo_epi16(v.registr(), 0b1000)};
#define GREX_CVT_IMPL_u32_u64_2(...) \
  /* [u32(v[0]), u32(v[1]), 0, 0] */ \
  const __m128 sh = _mm_shuffle_ps(_mm_castsi128_ps(v.r), _mm_setzero_ps(), 0b1000); \
  return SubVector<u32, 2, 4>{_mm_castps_si128(sh)};

#define GREX_CVT_IMPL_u8_u32_4(...) \
  /* [u8(v[0]), u8(v[1]), u8(v[2]), u8(v[3])] as u32x4 */ \
  const __m128i vu32 = _mm_and_si128(v.r, _mm_set1_epi32(0xFF)); \
  /* [u8(v[0]), u8(v[1]), u8(v[2]), u8(v[3]), 0, 0, 0, 0] as u16x8 */ \
  const __m128i vu16 = _mm_packus_epi16(vu32, _mm_setzero_si128()); \
  /* [u8(v[0]), u8(v[1]), u8(v[2]), u8(v[3]), 0, …, 0] as u8x16 */ \
  const auto r = _mm_packus_epi16(vu16, _mm_setzero_si128()); \
  return SubVector<u8, 4, 16>{r};
// super-native variants at level 1
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
// sub-native variant
#define GREX_CVT_IMPL_u16_u64_2(...) \
  /* lo = [u32(v0), u32(v1), -, -] as u32x4 */ \
  const auto lo = _mm_shuffle_epi32(v.r, 0b1000); \
  /* [u16(v0), u16(v1), -, …, -]  */ \
  return SubVector<u16, 2, 8>{_mm_shufflelo_epi16(lo, 0b1000)};

#define GREX_CVT_IMPL_u8_u64_2(...) \
  /* [u8(v[0]), u8(v[1])] as u64x2 */ \
  const __m128i vu64 = _mm_and_si128(v.r, _mm_set1_epi64x(0xFF)); \
  /* [u8(v[0]), u8(v[1]), 0, 0] as u32x4 */ \
  const __m128i vu32 = _mm_shuffle_epi32(vu64, 0b11011000); \
  /* [u8(v[0]), u8(v[1]), 0, …, 0] as u16x8 */ \
  const __m128i vu16 = _mm_shufflelo_epi16(vu32, 0b11011000); \
  /* [u8(v[0]), u8(v[1]), 0, …, 0] as u8x16 */ \
  return SubVector<u8, 2, 16>{_mm_packus_epi16(vu16, _mm_setzero_si128())};
#endif
#if GREX_X86_64_LEVEL < 3
// 256-bit super-native variant
#define GREX_CVT_IMPL_i32_i64_4(DSTKIND, ...) \
  /* [u32(v[0]), u32(v[1]), 0, 0] */ \
  const __m128 sh = \
    _mm_shuffle_ps(_mm_castsi128_ps(v.lower.r), _mm_castsi128_ps(v.upper.r), 0b1000'1000); \
  return DSTKIND##32x4 {_mm_castps_si128(sh)};
#define GREX_CVT_IMPL_i32_u64_4 GREX_CVT_IMPL_i32_i64_4
#define GREX_CVT_IMPL_u32_i64_4 GREX_CVT_IMPL_i32_i64_4
#define GREX_CVT_IMPL_u32_u64_4 GREX_CVT_IMPL_i32_i64_4
#endif

// Floating-point conversion
#define GREX_CVT_IMPL_f64_f32_2 GREX_CVT_INTRINSIC_EPU
#define GREX_CVT_IMPL_f32_f64_2 GREX_CVT_INTRINSIC_EPU

// Integer → floating-point
// f64
#if GREX_X86_64_LEVEL >= 4
#define GREX_CVT_IMPL_f64_i64_2 GREX_CVT_INTRINSIC_EPU
#define GREX_CVT_IMPL_f64_u64_2 GREX_CVT_INTRINSIC_EPU
#define GREX_CVT_IMPL_f64_u32_2 GREX_CVT_INTRINSIC_EPU
#else
// u64→f64: Convert the upper/lower 32 bits separately and add them (with some hidden bit fixes)
#define GREX_CVT_IMPL_f64_u64_2(...) \
  /* isolate the lower 32 bits of each 64 bit value */ \
  const __m128i lu32 = _mm_and_si128(v.r, _mm_set1_epi64x(0xFFFFFFFF)); \
  /* add an exponent of 52, i.e. the lower 32 bits have integer values again \
   * however, the hidden bit adds 2^52 to these values */ \
  const __m128d lf64 = _mm_castsi128_pd(_mm_or_si128(lu32, _mm_set1_epi64x(0x4330000000000000))); \
  /* shift away the lower 32 bit, leaving only the upper 32 bit */ \
  const __m128i hu32 = _mm_srli_epi64(v.r, 32); \
  /* set the exponent for the upper 32 bit to 84 = 52 + 32, leading to integer values again \
   * however, the hidden bit adds 2^84 to the values */ \
  const __m128d hf64 = _mm_castsi128_pd(_mm_or_si128(hu32, _mm_set1_epi64x(0x4530000000000000))); \
  /* subtract 2^84 * (1 + 2^-32) = 2^84 + 2^52, i.e. the hidden bits of both parts */ \
  const __m128d hsub = _mm_sub_pd(hf64, _mm_castsi128_pd(_mm_set1_epi64x(0x4530000000100000))); \
  /* add both parts together */ \
  return {.r = _mm_add_pd(lf64, hsub)};
// i64 → f64: Extraction and scalar conversion
#define GREX_CVT_IMPL_f64_i64_2(...) \
  const __m128d lf64 = _mm_cvtsi64_sd(_mm_undefined_pd(), _mm_cvtsi128_si64(v.r)); \
  const __m128d hf64 = _mm_cvtsi64_sd(_mm_undefined_pd(), extract(v, 1)); \
  return {.r = _mm_unpacklo_pd(lf64, hf64)};
// u32 → f64: Relatively simple because the conversion is precise
#define GREX_CVT_IMPL_f64_u32_2(...) \
  /* [u64(v[0]), u64(v[1])], i.e. the integer is in the lower bits of the mantissa */ \
  const __m128 unpack64 = _mm_unpacklo_ps(_mm_castsi128_ps(v.full.r), _mm_setzero_ps()); \
  /* exponent 52, i.e. 2^52 if interpreted as f64 */ \
  const __m128i exponent = _mm_set1_epi64x(0x4330000000000000); \
  /* set the exponent to 52, i.e. the correct value for the integer to be an integer again \
   * however, the hidden bit leads to the double value being the original value + 2^52  */ \
  const __m128i iv64 = _mm_or_si128(_mm_castps_si128(unpack64), exponent); \
  /* subtract the value of the hidden bit to get the correct value */ \
  return {.r = _mm_sub_pd(_mm_castsi128_pd(iv64), _mm_castsi128_pd(exponent))};
#endif
#define GREX_CVT_IMPL_f64_i32_2 GREX_CVT_INTRINSIC_EPU
#define GREX_CVT_IMPL_f64_i16_2 GREX_CVT_IMPL_SMALLI2F
#define GREX_CVT_IMPL_f64_u16_2 GREX_CVT_IMPL_SMALLI2F
#define GREX_CVT_IMPL_f64_i8_2 GREX_CVT_IMPL_SMALLI2F
#define GREX_CVT_IMPL_f64_u8_2 GREX_CVT_IMPL_SMALLI2F
// f32
#if GREX_X86_64_LEVEL >= 4
#define GREX_CVT_IMPL_f32_i64_2 GREX_CVT_INTRINSIC_EPU
#define GREX_CVT_IMPL_f32_u64_2 GREX_CVT_INTRINSIC_EPU
#else
// i64 → f32: Extraction and scalar conversion
#define GREX_CVT_IMPL_f32_i64_2(...) \
  const __m128 lf32 = _mm_cvtsi64_ss(_mm_undefined_ps(), _mm_cvtsi128_si64(v.r)); \
  const __m128 hf32 = _mm_cvtsi64_ss(_mm_undefined_ps(), extract(v, 1)); \
  return SubVector<f32, 2, 4>{_mm_unpacklo_ps(lf32, hf32)};
// u64 → f32: Divide by 2 with rounding to clear the sign bit, convert i64 to f32, and multiply by 2
#define GREX_CVT_IMPL_f32_u64_2(...) \
  /* v / 2 with truncation */ \
  const __m128i thalf = _mm_srli_epi64(v.r, 1); \
  /* v % 2 */ \
  const __m128i mod2 = _mm_and_si128(v.r, _mm_set1_epi64x(1)); \
  /* v / 2 with rounding to even */ \
  const __m128i half = _mm_or_si128(thalf, mod2); \
  /* v / 2 as f32 */ \
  const __m128 fhalf = convert(i64x2{half}, type_tag<f32>).full.r; \
  /* double v as f32 */ \
  return SubVector<f32, 2, 4>{_mm_add_ps(fhalf, fhalf)};
#endif
// u32→f32
#if GREX_X86_64_LEVEL >= 4
#define GREX_CVT_IMPL_f32_u32_4 GREX_CVT_INTRINSIC_EPU
#elif GREX_X86_64_LEVEL >= 2
// Convert the upper/lower 16 bits separately and add them (with some hidden bit fixes)
// TODO This uses pblendw, which is comparatively slow and consumes a lot of resources, it seems
#define GREX_CVT_IMPL_f32_u32_4(...) \
  /* combine an exponent of 23 with the lower 16 bits, i.e. these have integer values again \
   * however, the hidden bit adds 2^23 to these values */ \
  const __m128i a = _mm_blend_epi16(_mm_set1_epi32(0x4B000000), v.r, 0b01010101); \
  /* shift away the lower 16 bit, leaving only the upper 16 bit */ \
  const __m128i b = _mm_srli_epi32(v.r, 16); \
  /* set the exponent for the upper 16 bit to 39 = 23 + 16, leading to integer values again \
   * however, the hidden bit adds 2^39 to the values */ \
  const __m128i c = _mm_blend_epi16(b, _mm_set1_epi32(0x53000000), 0b10101010); \
  /* subtract 2^39 * (1 + 2^-16) = 2^39 + 2^23, i.e. the hidden bits of both parts */ \
  const __m128 d = _mm_sub_ps(_mm_castsi128_ps(c), _mm_castsi128_ps(_mm_set1_epi32(0x53000080))); \
  /* add both parts together */ \
  return {.r = _mm_add_ps(_mm_castsi128_ps(a), d)};
#else
// Convert the upper/lower 16 bits separately and add them (with some hidden bit fixes)
#define GREX_CVT_IMPL_f32_u32_4(...) \
  /* isolate the lower 16 bits of each 32 bit value */ \
  const __m128i lu16 = _mm_and_si128(v.r, _mm_set1_epi32(0xFFFF)); \
  /* add an exponent of 23, i.e. the lower 16 bits have integer values again \
   * however, the hidden bit adds 2^23 to these values */ \
  const __m128 lf32 = _mm_castsi128_ps(_mm_or_si128(lu16, _mm_set1_epi32(0x4B000000))); \
  /* shift away the lower 16 bit, leaving only the upper 16 bit */ \
  const __m128i hu16 = _mm_srli_epi32(v.r, 16); \
  /* set the exponent for the upper 16 bit to 39 = 23 + 16, leading to integer values again \
   * however, the hidden bit adds 2^39 to the values */ \
  const __m128 hf32 = _mm_castsi128_ps(_mm_or_si128(hu16, _mm_set1_epi32(0x53000000))); \
  /* subtract 2^39 * (1 + 2^-16) = 2^39 + 2^23, i.e. the hidden bits of both parts */ \
  const __m128 hsub = _mm_sub_ps(hf32, _mm_castsi128_ps(_mm_set1_epi32(0x53000080))); \
  /* add both parts together */ \
  return {.r = _mm_add_ps(lf32, hsub)};
#endif
#define GREX_CVT_IMPL_f32_i32_4 GREX_CVT_INTRINSIC_EPU
#define GREX_CVT_IMPL_f32_i16_4 GREX_CVT_IMPL_SMALLI2F
#define GREX_CVT_IMPL_f32_u16_4 GREX_CVT_IMPL_SMALLI2F
#define GREX_CVT_IMPL_f32_i8_4 GREX_CVT_IMPL_SMALLI2F
#define GREX_CVT_IMPL_f32_u8_4 GREX_CVT_IMPL_SMALLI2F

// Floating-point → integer
// f64
#if GREX_X86_64_LEVEL >= 4
#define GREX_CVT_IMPL_i64_f64_2 GREX_CVTT_INTRINSIC_EPU
#define GREX_CVT_IMPL_u64_f64_2 GREX_CVTT_INTRINSIC_EPU
#define GREX_CVT_IMPL_u32_f64_2 GREX_CVTT_INTRINSIC_EPU
#else
#define GREX_CVT_IMPL_i64_f64_2(...) \
  const i64 v0 = _mm_cvttsd_si64(v.r); \
  const i64 v1 = _mm_cvttsd_si64(_mm_unpackhi_pd(v.r, v.r)); \
  return {.r = _mm_set_epi64x(v1, v0)};
#define GREX_CVT_IMPL_u64_f64_2(...) \
  /* v - [2^63, 2^63, 2^63, 2^63] as f64x2, transforming the range of u64 to that of i64 */ \
  const __m128d voff = \
    _mm_sub_pd(v.registr(), _mm_castsi128_pd(_mm_set1_epi64x(0x43e0000000000000))); \
  /* [i64(v[0] - 2^63), i64(v[1] - 2^63)] */ \
  const __m128i oi64 = convert(f64x2{voff}, type_tag<i64>).r; \
  /* [i64(v[0]), i64(v[1])] */ \
  const __m128i vi64 = convert(v, type_tag<i64>).r; \
  /* sign[i] is true if v[i] < 0 or v[i] in [2^63, 2^64) → we need the offset version */ \
  const __m128i sign = _mm_shuffle_epi32(_mm_srai_epi32(vi64, 31), 0b11110101); \
  /* mask out the element for which the offset version is required */ \
  const __m128i mi64 = _mm_and_si128(oi64, sign); \
  /* due to modular arithmetic, it suffices to update using bitwise OR */ \
  return {.r = _mm_or_si128(vi64, mi64)};
#define GREX_CVT_IMPL_u32_f64_2(...) \
  /* [i32(v[0]), i32(v[1]), -, -] */ \
  const __m128i vi32 = _mm_cvttpd_epi32(v.r); \
  /* v - [2^31, 2^31, -, -] as f64x2, transforming the range of u32 to that of i32 */ \
  const __m128d voff = _mm_sub_pd(v.r, _mm_castsi128_pd(_mm_set1_epi64x(0x41E0000000000000))); \
  /* [i32(v[0] - 2^31), i32(v[1] - 2^31), -, -] */ \
  const __m128i oi32 = _mm_cvttpd_epi32(voff); \
  /* sign[i] is true if v[i] < 0 or v[i] in [2^31, 2^32) → we need the offset version */ \
  const __m128i sign = _mm_srai_epi32(vi32, 31); \
  /* mask out the element for which the offset version is required */ \
  const __m128i mi32 = _mm_and_si128(oi32, sign); \
  /* due to modular arithmetic, it suffices to update using bitwise OR */ \
  return SubVector<u32, 2, 4>{_mm_or_si128(vi32, mi32)};
#endif
#define GREX_CVT_IMPL_i32_f64_2 GREX_CVTT_INTRINSIC_EPU
#define GREX_CVT_IMPL_i16_f64_2 GREX_CVT_IMPL_F2SMALLI
#define GREX_CVT_IMPL_u16_f64_2 GREX_CVT_IMPL_F2SMALLI
#define GREX_CVT_IMPL_i8_f64_2 GREX_CVT_IMPL_F2SMALLI
#define GREX_CVT_IMPL_u8_f64_2 GREX_CVT_IMPL_F2SMALLI
// f32
#if GREX_X86_64_LEVEL >= 4
#define GREX_CVT_IMPL_i64_f32_2 GREX_CVTT_INTRINSIC_EPU
#define GREX_CVT_IMPL_u64_f32_2 GREX_CVTT_INTRINSIC_EPU
#define GREX_CVT_IMPL_u32_f32_4 GREX_CVTT_INTRINSIC_EPU
#else
#define GREX_CVT_IMPL_i64_f32_2(...) \
  const i64 li64 = _mm_cvttss_si64(v.registr()); \
  const __m128 hf32 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v.registr()), 0b01)); \
  const i64 hi64 = _mm_cvttss_si64(hf32); \
  return {.r = _mm_set_epi64x(hi64, li64)};
#define GREX_CVT_IMPL_u64_f32_2(...) \
  /* v - [2^63, 2^63, 2^63, 2^63] as f32x4, transforming the range of u64 to that of i64 */ \
  const __m128 voff = _mm_sub_ps(v.registr(), _mm_castsi128_ps(_mm_set1_epi32(0x5f000000))); \
  /* [i64(v[0] - 2^63), i64(v[1] - 2^63)] */ \
  const __m128i oi64 = convert(SubVector<f32, 2, 4>{voff}, type_tag<i64>).r; \
  /* [i64(v[0]), i64(v[1])] */ \
  const __m128i vi64 = convert(v, type_tag<i64>).r; \
  /* sign[i] is true if v[i] < 0 or v[i] in [2^63, 2^64) → we need the offset version */ \
  const __m128i sign = _mm_shuffle_epi32(_mm_srai_epi32(vi64, 31), 0b11110101); \
  /* mask out the element for which the offset version is required */ \
  const __m128i mi64 = _mm_and_si128(oi64, sign); \
  /* due to modular arithmetic, it suffices to update using bitwise OR */ \
  return {.r = _mm_or_si128(vi64, mi64)};
#define GREX_CVT_IMPL_u32_f32_4(...) \
  /* [i32(v[0]), i32(v[1]), i32(v[2]), i32(v[3])] */ \
  const __m128i vi32 = _mm_cvttps_epi32(v.r); \
  /* v - [2^31, 2^31, 2^31, 2^31] as f32x4, transforming the range of u32 to that of i32 */ \
  const __m128 voff = _mm_sub_ps(v.r, _mm_castsi128_ps(_mm_set1_epi32(0x4f000000))); \
  /* [i32(v[0] - 2^31), …, i32(v[3] - 2^31)] */ \
  const __m128i oi32 = _mm_cvttps_epi32(voff); \
  /* sign[i] is true if v[i] < 0 or v[i] in [2^31, 2^32) → we need the offset version */ \
  const __m128i sign = _mm_srai_epi32(vi32, 31); \
  /* mask out the element for which the offset version is required */ \
  const __m128i mi32 = _mm_and_si128(oi32, sign); \
  /* due to modular arithmetic, it suffices to update using bitwise OR */ \
  return {.r = _mm_or_si128(vi32, mi32)};
#endif
#define GREX_CVT_IMPL_i32_f32_4 GREX_CVTT_INTRINSIC_EPU
#define GREX_CVT_IMPL_i16_f32_4 GREX_CVT_IMPL_F2SMALLI
#define GREX_CVT_IMPL_u16_f32_4 GREX_CVT_IMPL_F2SMALLI
#define GREX_CVT_IMPL_i8_f32_4 GREX_CVT_IMPL_F2SMALLI
#define GREX_CVT_IMPL_u8_f32_4 GREX_CVT_IMPL_F2SMALLI

GREX_CVT_DEF_ALL(_mm, 128)
GREX_CVT(u, 16, u, 32, 2, _mm, 128)
#if GREX_X86_64_LEVEL == 1
GREX_CVT_SUPER(u, 8, u, 16, 16, _mm, 128)
GREX_CVT_SUPER(u, 8, u, 32, 8, _mm, 128)
GREX_CVT_SUPER(u, 8, u, 32, 16, _mm, 128)
#endif
#if GREX_X86_64_LEVEL < 3
GREX_CVT_SUPER(i, 32, i, 64, 4, _mm, 128)
GREX_CVT_SUPER(i, 32, u, 64, 4, _mm, 128)
GREX_CVT_SUPER(u, 32, i, 64, 4, _mm, 128)
GREX_CVT_SUPER(u, 32, u, 64, 4, _mm, 128)
#endif
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_CONVERT_128_VECTOR_HPP
