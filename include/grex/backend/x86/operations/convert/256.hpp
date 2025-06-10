// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_CONVERT_256_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_CONVERT_256_HPP

#include "grex/backend/x86/instruction-sets.hpp"
#include "grex/backend/x86/operations/convert/128.hpp" // IWYU pragma: keep
#include "grex/backend/x86/operations/convert/base.hpp"

#if GREX_X86_64_LEVEL >= 3
#include "grex/backend/defs.hpp"
#include "grex/backend/x86/types.hpp"
#include "grex/base/defs.hpp"
#endif

// AVX/AVX2 come with a lot more intrinsics built in than baseline x86-64,
// simplifying the implementation somewhat

namespace grex::backend {
// Increasing integer size
// Double integer size
#define GREX_CVT_IMPL_i16_i8_16 GREX_CVT_INTRINSIC_EPUI
#define GREX_CVT_IMPL_u16_u8_16 GREX_CVT_INTRINSIC_EPUI
#define GREX_CVT_IMPL_i32_i16_8 GREX_CVT_INTRINSIC_EPUI
#define GREX_CVT_IMPL_u32_u16_8 GREX_CVT_INTRINSIC_EPUI
#define GREX_CVT_IMPL_i64_i32_4 GREX_CVT_INTRINSIC_EPUI
#define GREX_CVT_IMPL_u64_u32_4 GREX_CVT_INTRINSIC_EPUI
// Quadruple integer size
#define GREX_CVT_IMPL_i32_i8_8 GREX_CVT_INTRINSIC_EPUI
#define GREX_CVT_IMPL_u32_u8_8 GREX_CVT_INTRINSIC_EPUI
#define GREX_CVT_IMPL_i64_i16_4 GREX_CVT_INTRINSIC_EPUI
#define GREX_CVT_IMPL_u64_u16_4 GREX_CVT_INTRINSIC_EPUI
// Octuple integer size
#define GREX_CVT_IMPL_i64_i8_4 GREX_CVT_INTRINSIC_EPUI
#define GREX_CVT_IMPL_u64_u8_4 GREX_CVT_INTRINSIC_EPUI

// Decreasing integer size: Truncation works the same for signed and unsigned types
#if GREX_X86_64_LEVEL >= 4
#define GREX_CVT_IMPL_u8_u16_16 GREX_CVT_INTRINSIC_EPI
#define GREX_CVT_IMPL_u16_u32_8 GREX_CVT_INTRINSIC_EPI
#define GREX_CVT_IMPL_u32_u64_4 GREX_CVT_INTRINSIC_EPI
#define GREX_CVT_IMPL_u8_u32_8 GREX_CVT_INTRINSIC_EPI
#define GREX_CVT_IMPL_u16_u64_4 GREX_CVT_INTRINSIC_EPI
#define GREX_CVT_IMPL_u8_u64_4 GREX_CVT_INTRINSIC_EPI
#elif GREX_X86_64_LEVEL >= 3
#define GREX_CVT_IMPL_u8_u16_16(...) \
  const __m256i idxs8 = \
    _mm256_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1, 0, 2, 4, 6, 8, 10, \
                     12, 14, -1, -1, -1, -1, -1, -1, -1, -1); \
  /* [u8(v[0]), …, u8(v[7]), 0, …, 0, u8(v[8]), …, u8(v[15]), 0, …, 0] */ \
  const __m256i shuf = _mm256_shuffle_epi8(v.r, idxs8); \
  /* [u8(v[0]), …, u8(v[15]), 0, …, 0] */ \
  return {.r = _mm256_castsi256_si128(_mm256_permute4x64_epi64(shuf, 0b11011000))};
#define GREX_CVT_IMPL_u16_u32_8(...) \
  const __m256i idxs16 = \
    _mm256_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 4, 5, 8, 9, \
                     12, 13, -1, -1, -1, -1, -1, -1, -1, -1); \
  /* [u16(v[0]), …, u16(v[3]), 0, …, 0, u16(v[4]), …, u16(v[7]), 0, …, 0] */ \
  const __m256i shuf = _mm256_shuffle_epi8(v.r, idxs16); \
  /* [u16(v[0]), …, u16(v[7]), 0, …, 0] */ \
  return {.r = _mm256_castsi256_si128(_mm256_permute4x64_epi64(shuf, 0b11011000))};
#define GREX_CVT_IMPL_u32_u64_4(...) \
  const __m256i idxs = _mm256_setr_epi32(0, 2, 4, 6, 4, 5, 6, 7); \
  /* [u32(v[0]), …, u32(v[3]), -, …, -] */ \
  return {.r = _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(v.r, idxs))};
#define GREX_CVT_IMPL_u8_u32_8(...) \
  const __m128i idxs = _mm_setr_epi8(0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1); \
  /* [u8(v[0]), …, u8(v[3]), 0, …, 0] */ \
  const __m128i v0 = _mm_shuffle_epi8(_mm256_castsi256_si128(v.r), idxs); \
  /* [u8(v[4]), …, u8(v[7]), 0, …, 0] */ \
  const __m128i v1 = _mm_shuffle_epi8(_mm256_extracti128_si256(v.r, 1), idxs); \
  /* [u8(v[0]), …, u8(v[7]), 0, …, 0] */ \
  return SubVector<u8, 8, 16>{_mm_unpacklo_epi32(v0, v1)};
#define GREX_CVT_IMPL_u16_u64_4(...) \
  /* [u16(v[0]), …, u16(v[3])] as u64x4 */ \
  const __m256i blended = _mm256_blend_epi16(v.r, _mm256_setzero_si256(), 0b11101110); \
  /* [u16(v[2]), u16(v[3])] as u64x2 */ \
  const __m128i hu128 = _mm256_extracti128_si256(blended, 1); \
  /* [u16(v0), …, u16(v3)] as u32x4 */ \
  const __m128i vu32 = _mm_packus_epi32(_mm256_castsi256_si128(blended), hu128); \
  /* [u16(v0), …, u16(v3), 0, …, 0] */ \
  return SubVector<u16, 4, 8>{_mm_packus_epi32(vu32, _mm_setzero_si128())};
// super-native variant
#define GREX_CVT_IMPL_u16_u64_8(...) \
  const __m256i lb = _mm256_blend_epi16(v.lower.r, _mm256_setzero_si256(), 0b11101110); \
  const __m256i hb = _mm256_blend_epi16(v.upper.r, _mm256_setzero_si256(), 0b11101110); \
  const __m128i lhu128 = _mm256_extracti128_si256(lb, 1); \
  const __m128i hhu128 = _mm256_extracti128_si256(hb, 1); \
  const __m128i lu32 = _mm_packus_epi32(_mm256_castsi256_si128(lb), lhu128); \
  const __m128i hu32 = _mm_packus_epi32(_mm256_castsi256_si128(hb), hhu128); \
  return u16x8{_mm_packus_epi32(lu32, hu32)};
#define GREX_CVT_IMPL_u8_u64_4(...) \
  const __m128i idxs = \
    _mm_setr_epi8(0, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1); \
  /* [u8(v[0]), u8(v[1]), 0, …, 0] */ \
  const __m128i lshf = _mm_shuffle_epi8(_mm256_castsi256_si128(v.r), idxs); \
  /* [u8(v[2]), u8(v[3]), 0, …, 0] */ \
  const __m128i hshf = _mm_shuffle_epi8(_mm256_extracti128_si256(v.r, 1), idxs); \
  /* [u8(v[0]), …, u8(v[3]), 0, …, 0] */ \
  return SubVector<u8, 4, 16>{_mm_unpacklo_epi16(lshf, hshf)};
#endif

// Floating-point conversion
#define GREX_CVT_IMPL_f64_f32_4 GREX_CVT_INTRINSIC_EPU
#define GREX_CVT_IMPL_f32_f64_4 GREX_CVT_INTRINSIC_EPU

// Integer → floating-point
// f64
#if GREX_X86_64_LEVEL >= 4
#define GREX_CVT_IMPL_f64_u64_4 GREX_CVT_INTRINSIC_EPU
#define GREX_CVT_IMPL_f64_i64_4 GREX_CVT_INTRINSIC_EPU
#define GREX_CVT_IMPL_f64_u32_4 GREX_CVT_INTRINSIC_EPU
#else
// u64→f64: Convert the upper/lower 32 bits separately and add them (with some hidden bit fixes)
#define GREX_CVT_IMPL_f64_u64_4(...) \
  /* isolate the lower 32 bits of each 64 bit value */ \
  const __m256i lu32 = _mm256_blend_epi32(v.r, _mm256_setzero_si256(), 0b10101010); \
  /* add an exponent of 52, i.e. the lower 32 bits have integer values again \
   * however, the hidden bit adds 2^52 to these values */ \
  const __m256d lf64 = \
    _mm256_castsi256_pd(_mm256_or_si256(lu32, _mm256_set1_epi64x(0x4330000000000000))); \
  /* shift away the lower 32 bit, leaving only the upper 32 bit */ \
  const __m256i hu32 = _mm256_srli_epi64(v.r, 32); \
  /* set the exponent for the upper 32 bit to 84 = 52 + 32, leading to integer values again \
   * however, the hidden bit adds 2^84 to the values */ \
  const __m256d hf64 = \
    _mm256_castsi256_pd(_mm256_or_si256(hu32, _mm256_set1_epi64x(0x4530000000000000))); \
  /* subtract 2^84 * (1 + 2^-32) = 2^84 + 2^52, i.e. the hidden bits of both parts */ \
  const __m256d hsub = \
    _mm256_sub_pd(hf64, _mm256_castsi256_pd(_mm256_set1_epi64x(0x4530000000100000))); \
  /* add both parts together */ \
  return {.r = _mm256_add_pd(lf64, hsub)};
// i64 → f64: Extraction and scalar conversion
#define GREX_CVT_IMPL_f64_i64_4(...) \
  const f64 v0 = f64(_mm256_extract_epi64(v.r, 0)); \
  const f64 v1 = f64(_mm256_extract_epi64(v.r, 1)); \
  const f64 v2 = f64(_mm256_extract_epi64(v.r, 2)); \
  const f64 v3 = f64(_mm256_extract_epi64(v.r, 3)); \
  return {.r = _mm256_setr_pd(v0, v1, v2, v3)};
// u32 → f64: Relatively simple because the conversion is precise
#define GREX_CVT_IMPL_f64_u32_4(...) \
  /* convert to 64 bits to be used as the mantissa */ \
  const __m256i vi64 = _mm256_cvtepu32_epi64(v.r); \
  /* 2^52 as an exponent */ \
  const __m256i exponent = _mm256_set1_epi64x(0x4330000000000000); \
  /* adjust the exponent to make the original integers integer-valued again \
   * howerver, the hidden bit adds 2^52 to the actual value */ \
  const __m256i vf64 = _mm256_or_si256(vi64, exponent); \
  /* subtract 2^52 to compensate for the hidden bit */ \
  return {.r = _mm256_sub_pd(_mm256_castsi256_pd(vf64), _mm256_castsi256_pd(exponent))};
#endif
#define GREX_CVT_IMPL_f64_i32_4 GREX_CVT_INTRINSIC_EPU
#define GREX_CVT_IMPL_f64_i16_4 GREX_CVT_IMPL_SMALLI2F
#define GREX_CVT_IMPL_f64_u16_4 GREX_CVT_IMPL_SMALLI2F
#define GREX_CVT_IMPL_f64_i8_4 GREX_CVT_IMPL_SMALLI2F
#define GREX_CVT_IMPL_f64_u8_4 GREX_CVT_IMPL_SMALLI2F
// f32
#if GREX_X86_64_LEVEL >= 4
#define GREX_CVT_IMPL_f32_i64_4 GREX_CVT_INTRINSIC_EPU
#define GREX_CVT_IMPL_f32_u64_4 GREX_CVT_INTRINSIC_EPU
#define GREX_CVT_IMPL_f32_u32_8 GREX_CVT_INTRINSIC_EPU
#else
#define GREX_CVT_IMPL_f32_i64_4(...) \
  const f32 v0 = f32(_mm256_extract_epi64(v.r, 0)); \
  const f32 v1 = f32(_mm256_extract_epi64(v.r, 1)); \
  const f32 v2 = f32(_mm256_extract_epi64(v.r, 2)); \
  const f32 v3 = f32(_mm256_extract_epi64(v.r, 3)); \
  return {.r = _mm_setr_ps(v0, v1, v2, v3)};
#define GREX_CVT_IMPL_f32_u64_4(...) \
  /* v % 2 */ \
  const __m256i mod2 = _mm256_and_si256(v.r, _mm256_set1_epi64x(1)); \
  /* v / 2 with truncation */ \
  const __m256i thalf = _mm256_srli_epi64(v.r, 1); \
  /* v / 2 with rounding to even */ \
  const __m256i half = _mm256_or_si256(thalf, mod2); \
  /* v / 2 as f32 */ \
  const __m128 fhalf = convert(i64x4{half}, type_tag<f32>).registr(); \
  /* double v as f32 */ \
  return {.r = _mm_add_ps(fhalf, fhalf)};
#define GREX_CVT_IMPL_f32_u32_8(...) \
  /* replace the upper 16 bit with an exponent of 23, leading to integer values again \
   * however, the hidden bit adds 2^23 to these values */ \
  const __m256i lf32 = _mm256_blend_epi16(v.r, _mm256_set1_epi32(0x4B000000), 0b10101010); \
  /* shift away the lower 16 bit, leaving only the upper 16 bit */ \
  const __m256i hu16 = _mm256_srli_epi32(v.r, 16); \
  /* set the exponent for the upper 16 bit to 39 = 23 + 16, leading to integer values again \
   * however, the hidden bit adds 2^39 to the values */ \
  const __m256i hf32 = _mm256_blend_epi16(hu16, _mm256_set1_epi32(0x53000000), 0b10101010); \
  /* subtract 2^39 * (1 + 2^-16) = 2^39 + 2^23, i.e. the hidden bits of both parts */ \
  const __m256 hsub = \
    _mm256_sub_ps(_mm256_castsi256_ps(hf32), _mm256_castsi256_ps(_mm256_set1_epi32(0x53000080))); \
  /* add both parts together */ \
  return {.r = _mm256_add_ps(_mm256_castsi256_ps(lf32), hsub)};
#endif
#define GREX_CVT_IMPL_f32_i32_8 GREX_CVT_INTRINSIC_EPU
#define GREX_CVT_IMPL_f32_i16_8 GREX_CVT_IMPL_SMALLI2F
#define GREX_CVT_IMPL_f32_u16_8 GREX_CVT_IMPL_SMALLI2F
#define GREX_CVT_IMPL_f32_i8_8 GREX_CVT_IMPL_SMALLI2F
#define GREX_CVT_IMPL_f32_u8_8 GREX_CVT_IMPL_SMALLI2F

// Floating-point → integer
// f64
#if GREX_X86_64_LEVEL >= 4
#define GREX_CVT_IMPL_i64_f64_4 GREX_CVTT_INTRINSIC_EPU
#define GREX_CVT_IMPL_u64_f64_4 GREX_CVTT_INTRINSIC_EPU
#define GREX_CVT_IMPL_u32_f64_4 GREX_CVTT_INTRINSIC_EPU
#else
#define GREX_CVT_IMPL_i64_f64_4(...) \
  const __m128d lo = _mm256_castpd256_pd128(v.r); \
  const __m128d hi = _mm256_extractf128_pd(v.r, 1); \
  const i64 v0 = _mm_cvttsd_si64(lo); \
  const i64 v1 = _mm_cvttsd_si64(_mm_unpackhi_pd(lo, lo)); \
  const i64 v2 = _mm_cvttsd_si64(hi); \
  const i64 v3 = _mm_cvttsd_si64(_mm_unpackhi_pd(hi, hi)); \
  return {.r = _mm256_set_epi64x(v3, v2, v1, v0)};
#define GREX_CVT_IMPL_u64_f64_4(...) \
  /* v - 2^63 as f64x4, transforming the range of u64 to that of i64 */ \
  const __m256d voff = \
    _mm256_sub_pd(v.registr(), _mm256_castsi256_pd(_mm256_set1_epi64x(0x43e0000000000000))); \
  /* [i64(v[0] - 2^63), …, i64(v[3] - 2^63)] */ \
  const __m256i oi64 = convert(f64x4{voff}, type_tag<i64>).r; \
  /* [i64(v[0]), …, i64(v[3])] */ \
  const __m256i vi64 = convert(v, type_tag<i64>).r; \
  /* sign[i] is true if v[i] < 0 or v[i] in [2^63, 2^64) → we need the offset version */ \
  const __m256i sign = _mm256_shuffle_epi32(_mm256_srai_epi32(vi64, 31), 0b11110101); \
  /* mask out the element for which the offset version is required */ \
  const __m256i mi64 = _mm256_and_si256(oi64, sign); \
  /* due to modular arithmetic, it suffices to update using bitwise OR */ \
  return {.r = _mm256_or_si256(vi64, mi64)};
#define GREX_CVT_IMPL_u32_f64_4(...) \
  /* [i32(v[0]), …, i32(v[4])] */ \
  const __m128i vi32 = _mm256_cvttpd_epi32(v.r); \
  /* 2^31 as f64 */ \
  const __m256i exponent = _mm256_set1_epi64x(0x41E0000000000000); \
  /* v - 2^31 as f64x4, transforming the range of u32 to that of i32 */ \
  const __m256d voff = _mm256_sub_pd(v.r, _mm256_castsi256_pd(exponent)); \
  /* [i32(v[0] - 2^31), …, i32(v[3] - 2^31)] */ \
  const __m128i oi32 = _mm256_cvttpd_epi32(voff); \
  /* sign[i] is true if v[i] < 0 or v[i] in [2^31, 2^32) → we need the offset version */ \
  const __m128i sign = _mm_srai_epi32(vi32, 31); \
  /* mask out the element for which the offset version is required */ \
  const __m128i mi32 = _mm_and_si128(oi32, sign); \
  /* due to modular arithmetic, it suffices to update using bitwise OR */ \
  return {.r = _mm_or_si128(vi32, mi32)};
#endif
#define GREX_CVT_IMPL_i32_f64_4 GREX_CVTT_INTRINSIC_EPU
#define GREX_CVT_IMPL_i16_f64_4 GREX_CVT_IMPL_F2SMALLI
#define GREX_CVT_IMPL_u16_f64_4 GREX_CVT_IMPL_F2SMALLI
#define GREX_CVT_IMPL_i8_f64_4 GREX_CVT_IMPL_F2SMALLI
#define GREX_CVT_IMPL_u8_f64_4 GREX_CVT_IMPL_F2SMALLI
// f32
#if GREX_X86_64_LEVEL >= 4
#define GREX_CVT_IMPL_i64_f32_4 GREX_CVTT_INTRINSIC_EPU
#define GREX_CVT_IMPL_u64_f32_4 GREX_CVTT_INTRINSIC_EPU
#define GREX_CVT_IMPL_u32_f32_8 GREX_CVTT_INTRINSIC_EPU
#else
#define GREX_CVT_IMPL_i64_f32_4(...) \
  const i64 i64i0 = _mm_cvttss_si64(v.registr()); \
  const i64 i64i1 = \
    _mm_cvttss_si64(_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v.registr()), 0b01))); \
  const i64 i64i2 = \
    _mm_cvttss_si64(_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v.registr()), 0b10))); \
  const i64 i64i3 = \
    _mm_cvttss_si64(_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v.registr()), 0b11))); \
  return {.r = _mm256_set_epi64x(i64i3, i64i2, i64i1, i64i0)};
#define GREX_CVT_IMPL_u64_f32_4(...) \
  /* v - 2^63 as f32x4, transforming the range of u64 to that of i64 */ \
  const __m128 voff = _mm_sub_ps(v.registr(), _mm_castsi128_ps(_mm_set1_epi32(0x5f000000))); \
  /* [i64(v[0] - 2^63), i64(v[1] - 2^63)] */ \
  const __m256i oi64 = convert(f32x4{voff}, type_tag<i64>).r; \
  /* [i64(v[0]), i64(v[1])] */ \
  const __m256i vi64 = convert(v, type_tag<i64>).r; \
  /* sign[i] is true if v[i] < 0 or v[i] in [2^63, 2^64) → we need the offset version */ \
  const __m256i sign = _mm256_shuffle_epi32(_mm256_srai_epi32(vi64, 31), 0b11110101); \
  /* mask out the element for which the offset version is required */ \
  const __m256i mi64 = _mm256_and_si256(oi64, sign); \
  /* due to modular arithmetic, it suffices to update using bitwise OR */ \
  return {.r = _mm256_or_si256(vi64, mi64)};
#define GREX_CVT_IMPL_u32_f32_8(...) \
  /* [i32(v[0]), …, i32(v[7])] */ \
  const __m256i vi32 = _mm256_cvttps_epi32(v.r); \
  /* v - 2^31 as f32x8, transforming the range of u32 to that of i32 */ \
  const __m256 voff = _mm256_sub_ps(v.r, _mm256_castsi256_ps(_mm256_set1_epi32(0x4f000000))); \
  /* [i32(v[0] - 2^31), …, i32(v[7] - 2^31)] */ \
  const __m256i oi32 = _mm256_cvttps_epi32(voff); \
  /* sign[i] is true if v[i] < 0 or v[i] in [2^31, 2^32) → we need the offset version */ \
  const __m256i sign = _mm256_srai_epi32(vi32, 31); \
  /* mask out the element for which the offset version is required */ \
  const __m256i mi32 = _mm256_and_si256(oi32, sign); \
  /* due to modular arithmetic, it suffices to update using bitwise OR */ \
  return {.r = _mm256_or_si256(vi32, mi32)};
#endif
#define GREX_CVT_IMPL_i32_f32_8 GREX_CVTT_INTRINSIC_EPU
#define GREX_CVT_IMPL_i16_f32_8 GREX_CVT_IMPL_F2SMALLI
#define GREX_CVT_IMPL_u16_f32_8 GREX_CVT_IMPL_F2SMALLI
#define GREX_CVT_IMPL_i8_f32_8 GREX_CVT_IMPL_F2SMALLI
#define GREX_CVT_IMPL_u8_f32_8 GREX_CVT_IMPL_F2SMALLI

#if GREX_X86_64_LEVEL >= 3
GREX_CVT_DEF_ALL(_mm256, 256)
#endif
#if GREX_X86_64_LEVEL == 3
GREX_CVT_SUPER(u, 16, u, 64, 8, _mm256, 256)
#endif
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_CONVERT_256_HPP
