// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_EXTRACT_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_EXTRACT_HPP

#include <cstddef>

#include <immintrin.h>

#include "grex/backend/base.hpp"
#include "grex/backend/macros/base.hpp"
#include "grex/backend/macros/for-each.hpp"
#include "grex/backend/macros/math.hpp"
#include "grex/backend/x86/instruction-sets.hpp"
#include "grex/backend/x86/macros/for-each.hpp"
#include "grex/backend/x86/macros/intrinsics.hpp"
#include "grex/backend/x86/operations/extract-single.hpp"
#include "grex/backend/x86/types.hpp"
#include "grex/base.hpp"

#if GREX_X86_64_LEVEL >= 2
#include "grex/backend/macros/cast.hpp"
#include "grex/backend/x86/operations/reinterpret.hpp"
#else
#include <array>

#include "grex/backend/x86/operations/store.hpp"
#endif

#if GREX_X86_64_LEVEL >= 3
#include "grex/backend/x86/operations/split.hpp"
#endif

#if GREX_X86_64_LEVEL >= 4
#include "grex/backend/macros/types.hpp"
#include "grex/backend/x86/operations/bit.hpp"
#endif

namespace grex::backend {
//==================================================================================================
// Extraction with a run-time index
//==================================================================================================

#if GREX_X86_64_LEVEL >= 2
//--------------------------------------------------------------------------------------------------
// Permuting the requested element to the front of the register
//--------------------------------------------------------------------------------------------------

/**
 * The control operand of a permutation with `tPartBits`-wide parts which moves the
 * `tValueBits`-wide element `index` to the front, packed into a single scalar: Part `p` of the
 * output is taken from part `index * parts + p` of the input, and since only the parts which make
 * up element 0 are of interest, the whole control fits into `tValueBits` bits.
 */
template<std::size_t tValueBits, std::size_t tPartBits>
requires(tPartBits <= tValueBits && tValueBits <= 64)
inline UnsignedInt<tValueBits / 8> extract_control(std::size_t index) {
  using Control = UnsignedInt<tValueBits / 8>;
  constexpr std::size_t parts = tValueBits / tPartBits;
  // A one in every part and the position of every part within the element, respectively.
  constexpr Control ones = static_apply<parts>(
    []<std::size_t... tParts>() { return Control(((Control{1} << (tParts * tPartBits)) | ...)); });
  constexpr Control offsets = static_apply<parts>([]<std::size_t... tParts>() {
    return Control(((Control(tParts) << (tParts * tPartBits)) | ...));
  });
  return Control(Control(index * parts) * ones + offsets);
}

/**
 * Permute the `tPartBits`-wide parts of `v` as prescribed by `control`, which is the packed control
 * described above, and return the lowest 128 bits of the result, of which only the parts belonging
 * to element 0 are meaningful.
 */
inline __m128i extract_permute(__m128i v, i64 control, IndexTag<8> /*part_bits*/) {
  return _mm_shuffle_epi8(v, _mm_cvtsi64_si128(control));
}
#if GREX_X86_64_LEVEL >= 3
inline __m128i extract_permute(__m128i v, i64 control, IndexTag<32> /*part_bits*/) {
  return _mm_castps_si128(_mm_permutevar_ps(_mm_castsi128_ps(v), _mm_cvtsi64_si128(control)));
}
// `vpermilpd` is the odd one out in that it takes the index of a part not from the lowest bits of
// the corresponding control part, but from its bit 1, which shifting the packed control accounts
// for. Being a within-128-bit permutation, it is only of use for a single 128-bit register.
inline __m128i extract_permute(__m128i v, i64 control, IndexTag<64> /*part_bits*/) {
  return _mm_castpd_si128(_mm_permutevar_pd(_mm_castsi128_pd(v), _mm_cvtsi64_si128(control << 1)));
}
inline __m128i extract_permute(__m256i v, i64 control, IndexTag<32> /*part_bits*/) {
  return _mm256_castsi256_si128(
    _mm256_permutevar8x32_epi32(v, _mm256_castsi128_si256(_mm_cvtsi64_si128(control))));
}
#endif
#if GREX_X86_64_LEVEL >= 4
inline __m128i extract_permute(__m128i v, i64 control, IndexTag<16> /*part_bits*/) {
  return _mm_permutexvar_epi16(_mm_cvtsi64_si128(control), v);
}
#define GREX_EXTRACT_PERMUTE(REGISTERBITS, PARTBITS) \
  inline __m128i extract_permute(__m##REGISTERBITS##i v, i64 control, \
                                 IndexTag<PARTBITS> /*part_bits*/) { \
    return _mm##REGISTERBITS##_castsi##REGISTERBITS##_si128( \
      _mm##REGISTERBITS##_permutexvar_epi##PARTBITS( \
        _mm##REGISTERBITS##_castsi128_si##REGISTERBITS(_mm_cvtsi64_si128(control)), v)); \
  }
GREX_EXTRACT_PERMUTE(256, 16)
GREX_EXTRACT_PERMUTE(256, 64)
GREX_EXTRACT_PERMUTE(512, 16)
GREX_EXTRACT_PERMUTE(512, 32)
GREX_EXTRACT_PERMUTE(512, 64)
#if GREX_HAS_AVX512VBMI
GREX_EXTRACT_PERMUTE(256, 8)
GREX_EXTRACT_PERMUTE(512, 8)
#endif
#endif

/**
 * The part width of the permutation which brings a `bits`-wide element of a `vector_bits`-wide
 * register to the front: The element width itself where a permutation of that width exists and 32
 * bits, i.e. `vpermd`, otherwise, in which case the element is isolated by a subsequent shift.
 * The choices mirror those which `shuffle` makes for the same element widths.
 */
consteval std::size_t extract_part_bits(std::size_t vector_bits, std::size_t bits) {
  if (vector_bits == 128) {
    // A byte shuffle is the only variable permutation below AVX and, being the one with the lowest
    // latency, remains the best choice for bytes even where `vpermb` is available.
    if (GREX_X86_64_LEVEL < 3 || bits == 8) {
      return 8;
    }
    // 16-bit parts require `vpermw`, so a byte shuffle has to fill in without AVX-512BW.
    if (bits == 16) {
      return (GREX_X86_64_LEVEL >= 4) ? 16 : 8;
    }
    // `vpermilps` and `vpermilpd` need no scaling of the index at all.
    return bits;
  }
  if (bits == 8) {
    return GREX_HAS_AVX512VBMI ? 8 : 32;
  }
  if (bits == 16 || bits == 64) {
    return (GREX_X86_64_LEVEL >= 4) ? bits : 32;
  }
  return 32;
}

/**
 * The width of the chunks which the permutation moves, i.e. of one or more parts: The element width
 * itself where the parts are no wider than the elements and the part width otherwise, in which case
 * a chunk covers multiple elements and the requested one has to be isolated by a shift.
 */
consteval std::size_t extract_chunk_bits(std::size_t vector_bits, std::size_t bits) {
  const std::size_t part_bits = extract_part_bits(vector_bits, bits);
  return (part_bits > bits) ? part_bits : bits;
}

/**
 * Move the `chunk_index`-th chunk of the unsigned integer vector `v` to the front of a 128-bit
 * register.
 */
template<AnyNativeVector TVec>
requires(UnsignedIntVector<TVec>)
inline __m128i extract_front(TVec v, std::size_t chunk_index) {
  static constexpr std::size_t bits = 8 * sizeof(ValueOf<TVec>);
  static constexpr std::size_t vector_bits = bits * size_of<TVec>;
  static constexpr std::size_t part_bits = extract_part_bits(vector_bits, bits);
  static constexpr std::size_t chunk_bits = extract_chunk_bits(vector_bits, bits);

  const i64 control = i64(extract_control<chunk_bits, part_bits>(chunk_index));
  return extract_permute(v.r, control, index_tag<part_bits>);
}

//--------------------------------------------------------------------------------------------------
// The operation itself
//--------------------------------------------------------------------------------------------------

/**
 * Extract the `index`-th element of `v` by permuting the chunk containing it to the front of the
 * register and reading it from there. Since only bit patterns are moved around, the permutation is
 * performed on the unsigned integer type of the same width.
 */
template<AnyNativeVector TVec>
inline ValueOf<TVec> extract(TVec v, std::size_t index) {
  using Value = ValueOf<TVec>;
  using Part = UnsignedInt<sizeof(Value)>;
  using Front = NativeVector<Part, 16 / sizeof(Part)>;
  static constexpr std::size_t bits = 8 * sizeof(Value);
  static constexpr std::size_t chunk_bits = extract_chunk_bits(bits * size_of<TVec>, bits);

  const std::size_t offset = index * bits;
  const __m128i front = extract_front(as<Part>(v), offset / chunk_bits);
  if constexpr (chunk_bits == bits) {
    return extract_single(as<Value>(Front{.r = front}));
  } else if constexpr (IntVectorizable<Value>) {
    // An integer ends up in a general-purpose register anyway, so isolating it there is cheaper
    // than a vector shift, which would require transferring the shift amount into a vector.
    using Chunk = UnsignedInt<chunk_bits / 8>;
    using ChunkVector = NativeVector<Chunk, 16 / sizeof(Chunk)>;
    return Value(extract_single(ChunkVector{.r = front}) >> (offset % chunk_bits));
  } else {
    // Binary16 lives in a vector register, which the shift must not leave.
    const __m128i shifted = _mm_srl_epi32(front, _mm_cvtsi32_si128(i32(offset % chunk_bits)));
    return extract_single(as<Value>(Front{.r = shifted}));
  }
}
#else
/** SSE2 provides no variable shuffle, so a round trip through memory is used as the fallback. */
template<AnyNativeVector TVec>
inline ValueOf<TVec> extract(TVec v, std::size_t index) {
  std::array<ValueOf<TVec>, size_of<TVec>> values{};
  store(values.data(), v);
  return values[index % size_of<TVec>];
}
#endif

//==================================================================================================
// Extraction with a compile-time index
//==================================================================================================

// i8x16: use _mm_extract_epi8 if available, otherwise extract via 16-bit lane
inline i8 extract(NativeVector<i8, 16> v, AnyIndexTag auto i) {
  static_assert(i < 16);
#if GREX_X86_64_LEVEL >= 2
  return i8(_mm_extract_epi8(v.r, i.value));
#else
  return i8(_mm_extract_epi16(v.r, i.value / 2) >> (8 * (i.value % 2)));
#endif
}

inline i16 extract(NativeVector<i16, 8> v, AnyIndexTag auto i) {
  static_assert(i < 8);
  return i16(_mm_extract_epi16(v.r, i.value));
}

// i32x4: use _mm_extract_epi32 if available, otherwise shuffle into lane 0
inline i32 extract(NativeVector<i32, 4> v, AnyIndexTag auto i) {
  static_assert(i < 4);
#if GREX_X86_64_LEVEL >= 2
  return _mm_extract_epi32(v.r, i.value);
#else
  return _mm_cvtsi128_si32(_mm_shuffle_epi32(v.r, i.value));
#endif
}

// i64x2: use _mm_extract_epi64 if available, otherwise unpack hi for index 1
inline i64 extract(NativeVector<i64, 2> v, AnyIndexTag auto i) {
  static_assert(i < 2);
#if GREX_X86_64_LEVEL >= 2
  return _mm_extract_epi64(v.r, i.value);
#else
  return _mm_cvtsi128_si64((i == 1) ? _mm_unpackhi_epi64(v.r, v.r) : v.r);
#endif
}

// f32x4: shuffle requested lane into lane 0 then cvtss
inline f32 extract(NativeVector<f32, 4> v, AnyIndexTag auto i) {
  static_assert(i < 4);
  const __m128i shuf = _mm_shuffle_epi32(_mm_castps_si128(v.r), i.value);
  return _mm_cvtss_f32(_mm_castsi128_ps(shuf));
}

// f64x2: for lane 1, unpackhi; then cvtsd
inline f64 extract(NativeVector<f64, 2> v, AnyIndexTag auto i) {
  static_assert(i < 2);
  const __m128d shuf = (i == 1) ? _mm_unpackhi_pd(v.r, v.r) : v.r;
  return _mm_cvtsd_f64(shuf);
}

// Binary16: shift down if `i > 0`, then `extract_single`.
inline f16 extract(f16x8 v, AnyIndexTag auto i) {
  static_assert(i < 8);
  if constexpr (i == 0) {
    return extract_single(v);
  } else if constexpr (i < 4) {
    return extract_single(f16x8{_mm_srli_epi64(v.r, i.value * 16)});
  } else {
    return extract_single(f16x8{_mm_bsrli_si128(v.r, i.value * 2)});
  }
}

#if GREX_X86_64_LEVEL >= 3
// AVX2: use _mm256_extract for integer vectors
#define GREX_STRACT_I256(KIND, BITS, SIZE) \
  inline KIND##BITS extract(NativeVector<KIND##BITS, SIZE> v, AnyIndexTag auto i) { \
    static_assert(i < SIZE); \
    return GREX_RETCAST(KIND, BITS, _mm256_extract_epi##BITS(v.r, i.value)); \
  }
GREX_FOREACH_INT_TYPE(GREX_STRACT_I256, 256)

// f32x8: just extract from low or high 128-bit half
inline f32 extract(f32x8 v, AnyIndexTag auto i) {
  static_assert(i < 8);
  if constexpr (i < 4) {
    return extract(get_low(v), i);
  } else {
    return extract(get_high(v), index_tag<i - 4>);
  }
}

// f64x4: low half via get_low, high half via 4×64 permutation+cvtsd
inline f64 extract(f64x4 v, AnyIndexTag auto i) {
  static_assert(i < 4);
  if constexpr (i < 2) {
    return extract(get_low(v), i);
  } else {
    return _mm256_cvtsd_f64(_mm256_permute4x64_pd(v.r, i.value));
  }
}

// Binary16x16: split based on the 128-bit lane `i` falls into.
inline f16 extract(f16x16 v, AnyIndexTag auto i) {
  static_assert(i < 16);
  if constexpr (i < 8) {
    return extract(get_low(v), i);
  } else {
    return extract(get_high(v), index_tag<i - 8>);
  }
}
#endif

#if GREX_X86_64_LEVEL >= 4
// Helpers to extract 128-bit lanes from 512-bit vectors
#define GREX_TRACT_512_INT(A, IMM8) _mm512_extracti32x4_epi32(A, IMM8)
#define GREX_TRACT_512_F32(A, IMM8) _mm512_extractf32x4_ps(A, IMM8)
#define GREX_TRACT_512_F64(A, IMM8) _mm512_extractf64x2_pd(A, IMM8)

#define GREX_TRACT_512_f(BITS, A, IMM8) GREX_TRACT_512_F##BITS(A, IMM8)
#define GREX_TRACT_512_i(BITS, A, IMM8) GREX_TRACT_512_INT(A, IMM8)
#define GREX_TRACT_512_u(BITS, A, IMM8) GREX_TRACT_512_INT(A, IMM8)
#define GREX_TRACT_512(KIND, BITS, A, IMM8) GREX_TRACT_512_##KIND(BITS, A, IMM8)

// 512-bit integer: split into 4 lanes of equal size and recurse
#define GREX_STRACT_512_INT(KIND, BITS, SIZE, RKIND) \
  if constexpr (lane_idx < 2) { \
    return extract(get_low(v), i); \
  } else { \
    const auto x = GREX_TRACT_512(RKIND, BITS, v.r, lane_idx); \
    return extract(NativeVector<KIND##BITS, lane_size>{x}, \
                   index_tag<i.value - lane_idx * lane_size>); \
  }

// 512-bit floating-point: rotate bits so desired element is at position 0, then extract_single
#define GREX_STRACT_512_FP(KIND, BITS, SIZE, RKIND) \
  if constexpr (lane_idx < 2) { \
    return extract(get_low(v), i); \
  } else { \
    const auto iv = GREX_KINDCAST(KIND, i, BITS, 512, v.r); \
    const auto ix = _mm512_alignr_epi##BITS(iv, iv, i.value); \
    const auto x = GREX_KINDCAST(i, KIND, BITS, 512, ix); \
    return extract_single(NativeVector<KIND##BITS, SIZE>{x}); \
  }

#define GREX_STRACT_512_f GREX_STRACT_512_FP
#define GREX_STRACT_512_i GREX_STRACT_512_INT
#define GREX_STRACT_512_u GREX_STRACT_512_INT

// Generic 512-bit extract: compute lane index, delegate to INT/FP variant
#define GREX_STRACT_512_I(KIND, BITS, SIZE, RKIND) \
  inline KIND##BITS extract(NativeVector<KIND##BITS, SIZE> v, AnyIndexTag auto i) { \
    static_assert(i < SIZE); \
    constexpr std::size_t lane_size = GREX_DIVIDE(SIZE, 4); \
    constexpr std::size_t lane_idx = i.value / lane_size; \
    GREX_CAT(GREX_STRACT_512_, RKIND)(KIND, BITS, SIZE, RKIND) \
  }
#define GREX_STRACT_512(KIND, BITS, SIZE) \
  GREX_STRACT_512_I(KIND, BITS, SIZE, GREX_REGKIND(KIND, BITS))

GREX_FOREACH_TYPE_EXT(GREX_STRACT_512, 512)
#endif

//==================================================================================================
// Mask extraction
//==================================================================================================

#if GREX_X86_64_LEVEL >= 4
// Compact masks: test bit i
#define GREX_EXTRACT_MASK_IMPL(KIND, BITS, SIZE, UMMASK) \
  inline bool extract(NativeMask<KIND##BITS, SIZE> v, std::size_t i) { \
    return bit_test(v.r, UMMASK(i)); \
  }
#define GREX_EXTRACT_MASK(KIND, BITS, SIZE) \
  GREX_EXTRACT_MASK_IMPL(KIND, BITS, SIZE, GREX_CAT(u, GREX_MAX(SIZE, 8)))
#else
// Broad masks: load mask as vector and test element != 0
#define GREX_EXTRACT_MASK(KIND, BITS, SIZE) \
  inline bool extract(NativeMask<KIND##BITS, SIZE> v, std::size_t i) { \
    return extract(NativeVector<u##BITS, SIZE>{v.r}, i) != 0; \
  }
#endif

#define GREX_EXTRACT_MASK_ALL(REGISTERBITS, BITPREFIX) \
  GREX_FOREACH_TYPE_EXT(GREX_EXTRACT_MASK, REGISTERBITS)

GREX_FOREACH_X86_64_LEVEL(GREX_EXTRACT_MASK_ALL)
} // namespace grex::backend

#include "grex/backend/shared/operations/extract.hpp" // IWYU pragma: export

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_EXTRACT_HPP
