// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_INTRINSICS_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_INTRINSICS_HPP

#include <immintrin.h>

#include "grex/backend/x86/instruction-sets.hpp"
#include "grex/base/defs.hpp"

// Wrap intrinsics that lead to conversion warnings in debug builds with libstdc++

#define GREX_DIAGCONV_PUSH() \
  _Pragma("GCC diagnostic push") _Pragma("GCC diagnostic ignored \"-Wsign-conversion\"") \
    _Pragma("GCC diagnostic ignored \"-Wconversion\"")
#define GREX_DIAGCONV_POP() _Pragma("GCC diagnostic pop")

namespace grex::backend {
namespace mm {
[[gnu::always_inline]] inline __m128i insert_epi8(__m128i a, int b, AnyIntTag auto imm8) {
  GREX_DIAGCONV_PUSH()
  return _mm_insert_epi8(a, b, imm8.value);
  GREX_DIAGCONV_POP()
}
[[gnu::always_inline]] inline __m128i insert_epi16(__m128i a, int b, AnyIntTag auto imm8) {
  GREX_DIAGCONV_PUSH()
  return _mm_insert_epi16(a, b, imm8.value);
  GREX_DIAGCONV_POP()
}
} // namespace mm

#if GREX_X86_64_LEVEL >= 4
namespace mm {
[[gnu::always_inline]] inline __m128 range_ps(__m128 a, __m128 b, AnyIntTag auto imm8) {
  GREX_DIAGCONV_PUSH()
  return _mm_range_ps(a, b, imm8.value);
  GREX_DIAGCONV_POP()
}
[[gnu::always_inline]] inline __m128d range_pd(__m128d a, __m128d b, AnyIntTag auto imm8) {
  GREX_DIAGCONV_PUSH()
  return _mm_range_pd(a, b, imm8.value);
  GREX_DIAGCONV_POP()
}

[[gnu::always_inline]] inline __m128 mmask_i32gather_ps(__m128 src, __mmask8 k, __m128i vindex,
                                                        const void* base_addr,
                                                        AnyIntTag auto scale) {
  GREX_DIAGCONV_PUSH()
  return _mm_mmask_i32gather_ps(src, k, vindex, base_addr, scale.value);
  GREX_DIAGCONV_POP()
}
[[gnu::always_inline]] inline __m128d mmask_i32gather_pd(__m128d src, __mmask8 k, __m128i vindex,
                                                         const void* base_addr,
                                                         AnyIntTag auto scale) {
  GREX_DIAGCONV_PUSH()
  return _mm_mmask_i32gather_pd(src, k, vindex, base_addr, scale.value);
  GREX_DIAGCONV_POP()
}
[[gnu::always_inline]] inline __m128i mmask_i32gather_epi32(__m128i src, __mmask8 k, __m128i vindex,
                                                            const void* base_addr,
                                                            AnyIntTag auto scale) {
  GREX_DIAGCONV_PUSH()
  return _mm_mmask_i32gather_epi32(src, k, vindex, base_addr, scale.value);
  GREX_DIAGCONV_POP()
}
[[gnu::always_inline]] inline __m128i mmask_i32gather_epi64(__m128i src, __mmask8 k, __m128i vindex,
                                                            const void* base_addr,
                                                            AnyIntTag auto scale) {
  GREX_DIAGCONV_PUSH()
  return _mm_mmask_i32gather_epi64(src, k, vindex, base_addr, scale.value);
  GREX_DIAGCONV_POP()
}

[[gnu::always_inline]] inline __m128 mmask_i64gather_ps(__m128 src, __mmask8 k, __m128i vindex,
                                                        const void* base_addr,
                                                        AnyIntTag auto scale) {
  GREX_DIAGCONV_PUSH()
  return _mm_mmask_i64gather_ps(src, k, vindex, base_addr, scale.value);
  GREX_DIAGCONV_POP()
}
[[gnu::always_inline]] inline __m128d mmask_i64gather_pd(__m128d src, __mmask8 k, __m128i vindex,
                                                         const void* base_addr,
                                                         AnyIntTag auto scale) {
  GREX_DIAGCONV_PUSH()
  return _mm_mmask_i64gather_pd(src, k, vindex, base_addr, scale.value);
  GREX_DIAGCONV_POP()
}
[[gnu::always_inline]] inline __m128i mmask_i64gather_epi32(__m128i src, __mmask8 k, __m128i vindex,
                                                            const void* base_addr,
                                                            AnyIntTag auto scale) {
  GREX_DIAGCONV_PUSH()
  return _mm_mmask_i64gather_epi32(src, k, vindex, base_addr, scale.value);
  GREX_DIAGCONV_POP()
}
[[gnu::always_inline]] inline __m128i mmask_i64gather_epi64(__m128i src, __mmask8 k, __m128i vindex,
                                                            const void* base_addr,
                                                            AnyIntTag auto scale) {
  GREX_DIAGCONV_PUSH()
  return _mm_mmask_i64gather_epi64(src, k, vindex, base_addr, scale.value);
  GREX_DIAGCONV_POP()
}
} // namespace mm

namespace mm256 {
[[gnu::always_inline]] inline __m256 range_ps(__m256 a, __m256 b, AnyIntTag auto imm8) {
  GREX_DIAGCONV_PUSH()
  return _mm256_range_ps(a, b, imm8.value);
  GREX_DIAGCONV_POP()
}
[[gnu::always_inline]] inline __m256d range_pd(__m256d a, __m256d b, AnyIntTag auto imm8) {
  GREX_DIAGCONV_PUSH()
  return _mm256_range_pd(a, b, imm8.value);
  GREX_DIAGCONV_POP()
}

[[gnu::always_inline]] inline __m256 mmask_i32gather_ps(__m256 src, __mmask8 k, __m256i vindex,
                                                        const void* base_addr,
                                                        AnyIntTag auto scale) {
  GREX_DIAGCONV_PUSH()
  return _mm256_mmask_i32gather_ps(src, k, vindex, base_addr, scale.value);
  GREX_DIAGCONV_POP()
}
[[gnu::always_inline]] inline __m256d mmask_i32gather_pd(__m256d src, __mmask8 k, __m128i vindex,
                                                         const void* base_addr,
                                                         AnyIntTag auto scale) {
  GREX_DIAGCONV_PUSH()
  return _mm256_mmask_i32gather_pd(src, k, vindex, base_addr, scale.value);
  GREX_DIAGCONV_POP()
}
[[gnu::always_inline]] inline __m256i mmask_i32gather_epi32(__m256i src, __mmask8 k, __m256i vindex,
                                                            const void* base_addr,
                                                            AnyIntTag auto scale) {
  GREX_DIAGCONV_PUSH()
  return _mm256_mmask_i32gather_epi32(src, k, vindex, base_addr, scale.value);
  GREX_DIAGCONV_POP()
}
[[gnu::always_inline]] inline __m256i mmask_i32gather_epi64(__m256i src, __mmask8 k, __m128i vindex,
                                                            const void* base_addr,
                                                            AnyIntTag auto scale) {
  GREX_DIAGCONV_PUSH()
  return _mm256_mmask_i32gather_epi64(src, k, vindex, base_addr, scale.value);
  GREX_DIAGCONV_POP()
}

[[gnu::always_inline]] inline __m128 mmask_i64gather_ps(__m128 src, __mmask8 k, __m256i vindex,
                                                        const void* base_addr,
                                                        AnyIntTag auto scale) {
  GREX_DIAGCONV_PUSH()
  return _mm256_mmask_i64gather_ps(src, k, vindex, base_addr, scale.value);
  GREX_DIAGCONV_POP()
}
[[gnu::always_inline]] inline __m256d mmask_i64gather_pd(__m256d src, __mmask8 k, __m256i vindex,
                                                         const void* base_addr,
                                                         AnyIntTag auto scale) {
  GREX_DIAGCONV_PUSH()
  return _mm256_mmask_i64gather_pd(src, k, vindex, base_addr, scale.value);
  GREX_DIAGCONV_POP()
}
[[gnu::always_inline]] inline __m128i mmask_i64gather_epi32(__m128i src, __mmask8 k, __m256i vindex,
                                                            const void* base_addr,
                                                            AnyIntTag auto scale) {
  GREX_DIAGCONV_PUSH()
  return _mm256_mmask_i64gather_epi32(src, k, vindex, base_addr, scale.value);
  GREX_DIAGCONV_POP()
}
[[gnu::always_inline]] inline __m256i mmask_i64gather_epi64(__m256i src, __mmask8 k, __m256i vindex,
                                                            const void* base_addr,
                                                            AnyIntTag auto scale) {
  GREX_DIAGCONV_PUSH()
  return _mm256_mmask_i64gather_epi64(src, k, vindex, base_addr, scale.value);
  GREX_DIAGCONV_POP()
}
} // namespace mm256

namespace mm512 {
[[gnu::always_inline]] inline __m512 range_ps(__m512 a, __m512 b, AnyIntTag auto imm8) {
  GREX_DIAGCONV_PUSH()
  return _mm512_range_ps(a, b, imm8.value);
  GREX_DIAGCONV_POP()
}
[[gnu::always_inline]] inline __m512d range_pd(__m512d a, __m512d b, AnyIntTag auto imm8) {
  GREX_DIAGCONV_PUSH()
  return _mm512_range_pd(a, b, imm8.value);
  GREX_DIAGCONV_POP()
}

[[gnu::always_inline]] inline __m512 i32gather_ps(__m512i vindex, const void* base_addr,
                                                  AnyIntTag auto scale) {
  GREX_DIAGCONV_PUSH()
  return _mm512_i32gather_ps(vindex, base_addr, scale.value);
  GREX_DIAGCONV_POP()
}
[[gnu::always_inline]] inline __m512d i32gather_pd(__m256i vindex, const void* base_addr,
                                                   AnyIntTag auto scale) {
  GREX_DIAGCONV_PUSH()
  return _mm512_i32gather_pd(vindex, base_addr, scale.value);
  GREX_DIAGCONV_POP()
}
[[gnu::always_inline]] inline __m512i i32gather_epi32(__m512i vindex, const void* base_addr,
                                                      AnyIntTag auto scale) {
  GREX_DIAGCONV_PUSH()
  return _mm512_i32gather_epi32(vindex, base_addr, scale.value);
  GREX_DIAGCONV_POP()
}
[[gnu::always_inline]] inline __m512i i32gather_epi64(__m256i vindex, const void* base_addr,
                                                      AnyIntTag auto scale) {
  GREX_DIAGCONV_PUSH()
  return _mm512_i32gather_epi64(vindex, base_addr, scale.value);
  GREX_DIAGCONV_POP()
}

[[gnu::always_inline]] inline __m512 mask_i32gather_ps(__m512 src, __mmask16 k, __m512i vindex,
                                                       const void* base_addr,
                                                       AnyIntTag auto scale) {
  GREX_DIAGCONV_PUSH()
  return _mm512_mask_i32gather_ps(src, k, vindex, base_addr, scale.value);
  GREX_DIAGCONV_POP()
}
[[gnu::always_inline]] inline __m512d mask_i32gather_pd(__m512d src, __mmask8 k, __m256i vindex,
                                                        const void* base_addr,
                                                        AnyIntTag auto scale) {
  GREX_DIAGCONV_PUSH()
  return _mm512_mask_i32gather_pd(src, k, vindex, base_addr, scale.value);
  GREX_DIAGCONV_POP()
}
[[gnu::always_inline]] inline __m512i mask_i32gather_epi32(__m512i src, __mmask16 k, __m512i vindex,
                                                           const void* base_addr,
                                                           AnyIntTag auto scale) {
  GREX_DIAGCONV_PUSH()
  return _mm512_mask_i32gather_epi32(src, k, vindex, base_addr, scale.value);
  GREX_DIAGCONV_POP()
}
[[gnu::always_inline]] inline __m512i mask_i32gather_epi64(__m512i src, __mmask8 k, __m256i vindex,
                                                           const void* base_addr,
                                                           AnyIntTag auto scale) {
  GREX_DIAGCONV_PUSH()
  return _mm512_mask_i32gather_epi64(src, k, vindex, base_addr, scale.value);
  GREX_DIAGCONV_POP()
}

[[gnu::always_inline]] inline __m256i i64gather_epi32(__m512i vindex, const void* base_addr,
                                                      AnyIntTag auto scale) {
  GREX_DIAGCONV_PUSH()
  return _mm512_i64gather_epi32(vindex, base_addr, scale.value);
  GREX_DIAGCONV_POP()
}
[[gnu::always_inline]] inline __m512i i64gather_epi64(__m512i vindex, const void* base_addr,
                                                      AnyIntTag auto scale) {
  GREX_DIAGCONV_PUSH()
  return _mm512_i64gather_epi64(vindex, base_addr, scale.value);
  GREX_DIAGCONV_POP()
}
[[gnu::always_inline]] inline __m512d i64gather_pd(__m512i vindex, const void* base_addr,
                                                   AnyIntTag auto scale) {
  GREX_DIAGCONV_PUSH()
  return _mm512_i64gather_pd(vindex, base_addr, scale.value);
  GREX_DIAGCONV_POP()
}
[[gnu::always_inline]] inline __m256 i64gather_ps(__m512i vindex, const void* base_addr,
                                                  AnyIntTag auto scale) {
  GREX_DIAGCONV_PUSH()
  return _mm512_i64gather_ps(vindex, base_addr, scale.value);
  GREX_DIAGCONV_POP()
}

[[gnu::always_inline]] inline __m256i mask_i64gather_epi32(__m256i src, __mmask8 k, __m512i vindex,
                                                           const void* base_addr,
                                                           AnyIntTag auto scale) {
  GREX_DIAGCONV_PUSH()
  return _mm512_mask_i64gather_epi32(src, k, vindex, base_addr, scale.value);
  GREX_DIAGCONV_POP()
}
[[gnu::always_inline]] inline __m512i mask_i64gather_epi64(__m512i src, __mmask8 k, __m512i vindex,
                                                           const void* base_addr,
                                                           AnyIntTag auto scale) {
  GREX_DIAGCONV_PUSH()
  return _mm512_mask_i64gather_epi64(src, k, vindex, base_addr, scale.value);
  GREX_DIAGCONV_POP()
}
[[gnu::always_inline]] inline __m512d mask_i64gather_pd(__m512d src, __mmask8 k, __m512i vindex,
                                                        const void* base_addr,
                                                        AnyIntTag auto scale) {
  GREX_DIAGCONV_PUSH()
  return _mm512_mask_i64gather_pd(src, k, vindex, base_addr, scale.value);
  GREX_DIAGCONV_POP()
}
[[gnu::always_inline]] inline __m256 mask_i64gather_ps(__m256 src, __mmask8 k, __m512i vindex,
                                                       const void* base_addr,
                                                       AnyIntTag auto scale) {
  GREX_DIAGCONV_PUSH()
  return _mm512_mask_i64gather_ps(src, k, vindex, base_addr, scale.value);
  GREX_DIAGCONV_POP()
}
} // namespace mm512
#endif
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_INTRINSICS_HPP
