// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_SPLIT_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_SPLIT_HPP

#include <cstddef>

#include <immintrin.h>

#include "grex/backend/choosers.hpp"
#include "grex/backend/defs.hpp"
#include "grex/backend/x86/base.hpp"
#include "grex/backend/x86/instruction-sets.hpp"
#include "grex/backend/x86/macros/for-each.hpp"
#include "grex/backend/x86/macros/intrinsics.hpp"
#include "grex/backend/x86/math.hpp"
#include "grex/backend/x86/types.hpp" // IWYU pragma: keep
#include "grex/base/defs.hpp"

namespace grex::backend {
#define GREX_SPLIT_LETTER_f f
#define GREX_SPLIT_LETTER_i i
#define GREX_SPLIT_LETTER_u i
#define GREX_SPLIT_LETTER(KIND) GREX_SPLIT_LETTER_##KIND

#define GREX_SPLIT_WRAP(KIND, BITS, SIZE, HALF, IMPL) \
  inline Vector<KIND##BITS, GREX_DIVIDE(SIZE, 2)> split(Vector<KIND##BITS, SIZE> v, \
                                                        IndexTag<HALF>) { \
    return {.r = IMPL}; \
  }

// The lower half can always be extracted using a cast
#define GREX_SPLIT_LOWER(KIND, BITS, SIZE, HALF, BITPREFIX, REGISTERBITS) \
  GREX_SPLIT_WRAP(KIND, BITS, SIZE, HALF, \
                  GREX_CAT(BITPREFIX##_cast, GREX_SIR_SUFFIX(KIND, BITS, REGISTERBITS), _, \
                           GREX_SIR_SUFFIX(KIND, BITS, GREX_DIVIDE(REGISTERBITS, 2)))(v.r))

// 128 bit: No splitting
#define GREX_SPLIT_128_0(...)
#define GREX_SPLIT_128_1(...)
// 256 bit
#define GREX_SPLIT_256_0 GREX_SPLIT_LOWER
#define GREX_SPLIT_256_1(KIND, BITS, SIZE, HALF, ...) \
  GREX_SPLIT_WRAP(KIND, BITS, SIZE, HALF, \
                  GREX_CAT(_mm256_extract, GREX_SPLIT_LETTER(KIND), 128_, \
                           GREX_SI_SUFFIX(KIND, BITS, 256))(v.r, 1))
// 512 bit
#define GREX_SPLIT_512_0 GREX_SPLIT_LOWER
#define GREX_SPLIT_512_1_f32(...) _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(v.r), 1))
#define GREX_SPLIT_512_1_f64(...) _mm512_extractf64x4_pd(v.r, 1)
#define GREX_SPLIT_512_1_f(KIND, BITS, ...) GREX_SPLIT_512_1_f##BITS(KIND, BITS, __VA_ARGS__)
#define GREX_SPLIT_512_1_i(...) _mm512_extracti64x4_epi64(v.r, 1)
#define GREX_SPLIT_512_1_u(...) _mm512_extracti64x4_epi64(v.r, 1)
#define GREX_SPLIT_512_1(KIND, BITS, SIZE, HALF, ...) \
  GREX_SPLIT_WRAP(KIND, BITS, SIZE, HALF, GREX_SPLIT_512_1_##KIND(KIND, BITS))

// Wrapper macros
#define GREX_SPLIT_HALF(KIND, BITS, SIZE, HALF, BITPREFIX, REGISTERBITS) \
  GREX_SPLIT_##REGISTERBITS##_##HALF(KIND, BITS, SIZE, HALF, BITPREFIX, REGISTERBITS)
#define GREX_SPLIT(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS) \
  GREX_SPLIT_HALF(KIND, BITS, SIZE, 0, BITPREFIX, REGISTERBITS) \
  GREX_SPLIT_HALF(KIND, BITS, SIZE, 1, BITPREFIX, REGISTERBITS)
#define GREX_SPLIT_ALL(REGISTERBITS, BITPREFIX) \
  GREX_FOREACH_TYPE(GREX_SPLIT, REGISTERBITS, BITPREFIX, REGISTERBITS)

GREX_FOREACH_X86_64_LEVEL(GREX_SPLIT_ALL)

#define GREX_SPLIT_f64x2_0(KIND, BITS, SIZE) v.registr()
#define GREX_SPLIT_f64x2_1(KIND, BITS, SIZE) \
  _mm_castsi128_ps(_mm_unpackhi_epi64(_mm_castps_si128(v.registr()), _mm_setzero_si128()))
#define GREX_SPLIT_i64x2_0(KIND, BITS, SIZE) v.registr()
#define GREX_SPLIT_i64x2_1(KIND, BITS, SIZE) _mm_unpackhi_epi64(v.registr(), _mm_setzero_si128())
#define GREX_SPLIT_i32x2_0(KIND, BITS, SIZE) v.registr()
#define GREX_SPLIT_i32x2_1(KIND, BITS, SIZE) _mm_shuffle_epi32(v.registr(), 1)
#define GREX_SPLIT_i16x2_0(KIND, BITS, SIZE) v.registr()
#define GREX_SPLIT_i16x2_1(KIND, BITS, SIZE) _mm_shufflelo_epi16(v.registr(), 1)
#define GREX_SPLIT_SUB(KIND, BITS, SIZE, HALF, IMPL) \
  inline VectorFor<KIND##BITS, GREX_DIVIDE(SIZE, 2)> split(VectorFor<KIND##BITS, SIZE> v, \
                                                           IndexTag<HALF>) { \
    return VectorFor<KIND##BITS, GREX_DIVIDE(SIZE, 2)>{IMPL##_##HALF(KIND, BITS, SIZE)}; \
  }
#define GREX_SPLIT_SUB_ALL(KIND, BITS, SIZE, IMPL) \
  GREX_SPLIT_SUB(KIND, BITS, SIZE, 0, IMPL) \
  GREX_SPLIT_SUB(KIND, BITS, SIZE, 1, IMPL)

// 64×2
GREX_SPLIT_SUB_ALL(f, 32, 4, GREX_SPLIT_f64x2)
GREX_SPLIT_SUB_ALL(i, 32, 4, GREX_SPLIT_i64x2)
GREX_SPLIT_SUB_ALL(u, 32, 4, GREX_SPLIT_i64x2)
GREX_SPLIT_SUB_ALL(i, 16, 8, GREX_SPLIT_i64x2)
GREX_SPLIT_SUB_ALL(u, 16, 8, GREX_SPLIT_i64x2)
GREX_SPLIT_SUB_ALL(i, 8, 16, GREX_SPLIT_i64x2)
GREX_SPLIT_SUB_ALL(u, 8, 16, GREX_SPLIT_i64x2)
// 32×2
GREX_SPLIT_SUB_ALL(i, 16, 4, GREX_SPLIT_i32x2)
GREX_SPLIT_SUB_ALL(u, 16, 4, GREX_SPLIT_i32x2)
GREX_SPLIT_SUB_ALL(i, 8, 8, GREX_SPLIT_i32x2)
GREX_SPLIT_SUB_ALL(u, 8, 8, GREX_SPLIT_i32x2)
// 16×2
GREX_SPLIT_SUB_ALL(i, 8, 4, GREX_SPLIT_i16x2)
GREX_SPLIT_SUB_ALL(u, 8, 4, GREX_SPLIT_i16x2)

// Split super-native vector
template<typename THalf>
inline THalf split(SuperVector<THalf> v, IndexTag<0> /*tag*/) {
  return v.lower;
}
template<typename THalf>
inline THalf split(SuperVector<THalf> v, IndexTag<1> /*tag*/) {
  return v.upper;
}

// Mask splitting
#if GREX_X86_64_LEVEL >= 4
// AVX-512: No-op for the lower half, bit shift for the upper
template<Vectorizable T, std::size_t tSize>
inline MaskFor<T, tSize / 2> split(Mask<T, tSize> m, IndexTag<0> /*tag*/) {
  using Out = MaskFor<T, tSize / 2>;
  using Register = Out::Register;
  return Out{Register(m.r)};
}
template<Vectorizable T, std::size_t tSize>
inline MaskFor<T, tSize / 2> split(Mask<T, tSize> m, IndexTag<1> /*tag*/) {
  using Out = MaskFor<T, tSize / 2>;
  using Register = Out::Register;
  return Out{Register(m.r >> (tSize / 2))};
}
#else
// Pre-AVX-512: Reinterpret as signed integer and split that way
template<Vectorizable T, std::size_t tSize, std::size_t tIdx>
inline MaskFor<T, tSize / 2> split(Mask<T, tSize> m, IndexTag<tIdx> tag) {
  const auto r = split(VectorFor<SignedInt<sizeof(T)>, tSize>{m.registr()}, tag).registr();
  return MaskFor<T, tSize / 2>{r};
}
template<Vectorizable T, std::size_t tPart, std::size_t tSize, std::size_t tIdx>
inline MaskFor<T, tPart / 2> split(SubMask<T, tPart, tSize> m, IndexTag<tIdx> tag) {
  const auto r = split(VectorFor<SignedInt<sizeof(T)>, tPart>{m.registr()}, tag).registr();
  return MaskFor<T, tPart / 2>{r};
}
#endif

// Split super-native mask
template<typename THalf>
inline THalf split(SuperMask<THalf> m, IndexTag<0> /*tag*/) {
  return m.lower;
}
template<typename THalf>
inline THalf split(SuperMask<THalf> m, IndexTag<1> /*tag*/) {
  return m.upper;
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_SPLIT_HPP
