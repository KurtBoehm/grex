#include <algorithm>
#include <array>
#include <bit>
#include <concepts>
#include <cstddef>
#include <cstring>
#include <random>
#include <vector>

#include <benchmark/benchmark.h>
#include <pcg_extras.hpp>
#include <pcg_random.hpp>

#include "grex/grex.hpp"

using namespace grex::primitives;
namespace be = grex::backend;

template<be::AnyVector TVec>
requires(be::AnyNativeVector<TVec> || be::AnySubNativeVector<TVec>)
GREX_ALWAYS_INLINE inline TVec load_part_grex(const be::ValueOf<TVec>* ptr, std::size_t size,
                                              grex::TypeTag<TVec> tag) {
  return be::load_part(ptr, size, tag);
}

template<be::AnySubNativeVector TVec>
GREX_ALWAYS_INLINE inline TVec load_part_xgrex(const be::ValueOf<TVec>* ptr, std::size_t size,
                                               grex::TypeTag<TVec> /*tag*/) {
  return TVec{be::load_part(ptr, size, grex::type_tag<typename TVec::Full>)};
}

alignas(16) inline constexpr i32 mask_table[5][4] = {
  {0, 0, 0, 0}, // 0 elements
  {-1, 0, 0, 0}, // 1 element
  {-1, -1, 0, 0}, // 2 elements
  {-1, -1, -1, 0}, // 3 elements
  {-1, -1, -1, -1}, // 4 elements
};

#if GREX_X86_64_LEVEL >= 3
GREX_ALWAYS_INLINE inline __m128d load_part_table(const f64* ptr, std::size_t size,
                                                  grex::TypeTag<be::f64x2> /*tag*/) {
  __m128i mask = _mm_load_si128(reinterpret_cast<const __m128i*>(mask_table[size]));
  return _mm_maskload_pd(ptr, mask);
}
GREX_ALWAYS_INLINE inline __m128 load_part_table(const f32* ptr, std::size_t size,
                                                 grex::TypeTag<be::f32x4> /*tag*/) {
  __m128i mask = _mm_load_si128(reinterpret_cast<const __m128i*>(mask_table[size]));
  return _mm_maskload_ps(ptr, mask);
}

GREX_ALWAYS_INLINE inline __m128i load_part_table(const i32* ptr, std::size_t size,
                                                  grex::TypeTag<be::i32x4> /*tag*/) {
  __m128i mask = _mm_load_si128(reinterpret_cast<const __m128i*>(mask_table[size]));
  return _mm_maskload_epi32(ptr, mask);
}
#endif

#if GREX_X86_64_LEVEL >= 2
GREX_ALWAYS_INLINE inline __m128d load_part_sse(const f64* ptr, std::size_t size,
                                                grex::TypeTag<be::f64x2> /*tag*/) {
  __m128d v = _mm_undefined_pd();
  switch (size) {
    case 2: return _mm_loadu_pd(ptr); // full vector
    case 1:
      v = __m128d(_mm_insert_epi64(__m128i(v), std::bit_cast<i64>(ptr[0]), 0));
      [[fallthrough]];
    case 0:
    default: return v;
  }
}

GREX_ALWAYS_INLINE inline __m128i load_part_sse(const i32* ptr, std::size_t size,
                                                grex::TypeTag<be::i32x4> /*tag*/) {
  __m128i v = _mm_undefined_si128();
  switch (size) {
    case 4: return _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr)); // full vector
    case 3:
      v = _mm_insert_epi32(v, ptr[2], 2); // element 2
      [[fallthrough]];
    case 2:
      v = _mm_insert_epi32(v, ptr[1], 1); // element 1
      [[fallthrough]];
    case 1:
      v = _mm_insert_epi32(v, ptr[0], 0); // element 0
      [[fallthrough]];
    case 0:
    default: return v;
  }
}

namespace shuffle_u8 {
using ShuffleRow = std::array<u8, 16>;
template<std::size_t tBlockBytes>
using ShuffleTable = std::array<ShuffleRow, 17 - tBlockBytes>;

// Generic compile-time shuffle-table generator for a given block size B.
//
// Conceptually we form AB = [ src[0 .. B-1], src[n-B .. n-1] ] (2*B bytes),
// sitting in the low bytes of a __m128i, then pshufb(AB, mask[n]) gives:
//
//   result[0 .. n-1] = src[0 .. n-1]
//   result[n .. 15]  = 0
//
// Valid n for a given B:   B <= n <= min(2*B, 16).
// Other rows are filled with 0x80 (zero everything if used by mistake).
template<std::size_t tBlockBytes>
consteval ShuffleTable<tBlockBytes> make_shuffle_table_block() {
  static_assert(tBlockBytes == 8 || tBlockBytes == 4 || tBlockBytes == 2);

  ShuffleTable<tBlockBytes> table{};

  for (std::size_t n = tBlockBytes; n <= 16; ++n) {
    auto& row = table[n - tBlockBytes];
    row.fill(0x80); // default: zero all bytes

    if (n > 2 * tBlockBytes) {
      continue;
    }

    for (std::size_t i = 0; i < 16; ++i) {
      u8 idx = 0x80; // default: zero this output byte

      if (i < tBlockBytes) {
        // First block: src[0 .. B-1] -> AB[0 .. B-1]
        idx = u8(i);
      } else if (i < n) {
        // Second block: src[n-B .. n-1] -> AB[B .. 2B-1]
        // For i in [B, n-1], map to AB index:
        //   idx = i + 2B - n
        idx = u8(i + 2 * tBlockBytes - n);
      }

      row[i] = idx;
    }
  }

  return table;
}

// Three tables, one per block size.
alignas(16) inline constexpr ShuffleTable<8> shuf_masks_8 = make_shuffle_table_block<8>();
alignas(16) inline constexpr ShuffleTable<4> shuf_masks_4 = make_shuffle_table_block<4>();
alignas(16) inline constexpr ShuffleTable<2> shuf_masks_2 = make_shuffle_table_block<2>();
} // namespace shuffle_u8

// Load up to 16 bytes from src without reading past src+len, zero-padding.
//
// Returns a __m128i where:
//   bytes [0 .. len-1] = src[0 .. len-1],
//   bytes [len .. 15]  = 0.
//
// Requires: 0 <= len <= 16, SSSE3 for _mm_shuffle_epi8.
__m128i load_part_overlap(const u8* src, std::size_t len, grex::TypeTag<be::u8x16> /*tag*/) {
  if (len == 0) [[unlikely]] {
    return _mm_setzero_si128();
  }
  if (len >= 16) [[unlikely]] {
    return _mm_loadu_si128(reinterpret_cast<const __m128i*>(src));
  }

  // 8-byte block path: len ∈ [8,16]
  if (len >= 8) {
    __m128i lo = _mm_loadu_si64(src);
    __m128i hi = _mm_loadu_si64(src + len - 8);
    // AB = [src[0..7], src[len-8..len-1]]
    __m128i ab = _mm_unpacklo_epi64(lo, hi);

    const shuffle_u8::ShuffleRow& row = shuffle_u8::shuf_masks_8[len - 8];
    __m128i mask = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row.data()));
    return _mm_shuffle_epi8(ab, mask);
  }

  // 4-byte block path: len ∈ [4,7]
  if (len >= 4) {
    __m128i lo = _mm_loadu_si32(src);
    __m128i hi = _mm_loadu_si32(src + (len - 4));
    // AB = [src[0..3], src[len-4..len-1]] in bytes [0..7]
    __m128i ab = _mm_unpacklo_epi32(lo, hi);

    const shuffle_u8::ShuffleRow& row = shuffle_u8::shuf_masks_4[len - 4];
    __m128i mask = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row.data()));

    return _mm_shuffle_epi8(ab, mask);
  }

  // 2-byte block path: len ∈ [2,3]
  if (len >= 2) {
    __m128i lo = _mm_loadu_si16(src);
    __m128i hi = _mm_loadu_si16(src + (len - 2));
    // AB = [src[0..1], src[len-2..len-1]] in bytes [0..3]
    __m128i ab = _mm_unpacklo_epi16(lo, hi);

    const shuffle_u8::ShuffleRow& row = shuffle_u8::shuf_masks_2[len - 2];
    __m128i mask = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row.data()));

    return _mm_shuffle_epi8(ab, mask);
  }

  // len == 1
  return _mm_cvtsi32_si128(static_cast<unsigned char>(*src));
}
__m128i load_part_overlap(const u16* src, std::size_t len, grex::TypeTag<be::u16x8> /*tag*/) {
  if (len == 0) [[unlikely]] {
    return _mm_setzero_si128();
  }
  if (len >= 8) [[unlikely]] {
    return _mm_loadu_si128(reinterpret_cast<const __m128i*>(src));
  }

  // 8-byte block path: len ∈ [4,8]
  if (len >= 4) {
    __m128i lo = _mm_loadu_si64(src);
    __m128i hi = _mm_loadu_si64(src + len - 4);
    // AB = [src[0..7], src[len-8..len-1]]
    __m128i ab = _mm_unpacklo_epi64(lo, hi);

    const shuffle_u8::ShuffleRow& row = shuffle_u8::shuf_masks_8[2 * len - 8];
    __m128i mask = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row.data()));
    return _mm_shuffle_epi8(ab, mask);
  }

  // 4-byte block path: len ∈ [2,3]
  if (len >= 2) {
    __m128i lo = _mm_loadu_si32(src);
    __m128i hi = _mm_loadu_si32(src + (len - 2));
    // AB = [src[0..3], src[len-4..len-1]] in bytes [0..7]
    __m128i ab = _mm_unpacklo_epi32(lo, hi);

    const shuffle_u8::ShuffleRow& row = shuffle_u8::shuf_masks_4[2 * len - 4];
    __m128i mask = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row.data()));

    return _mm_shuffle_epi8(ab, mask);
  }

  // len == 1
  return _mm_loadu_si16(src);
}
__m128i load_part_overlap(const i32* src, std::size_t len, grex::TypeTag<be::i32x4> /*tag*/) {
  if (len == 0) [[unlikely]] {
    return _mm_setzero_si128();
  }
  if (len >= 4) [[unlikely]] {
    return _mm_loadu_si128(reinterpret_cast<const __m128i*>(src));
  }

  // 8-byte block path: len ∈ [2,3]
  if (len >= 2) {
    __m128i lo = _mm_loadu_si64(src);
    __m128i hi = _mm_loadu_si64(src + len - 2);
    // AB = [src[0..7], src[len-8..len-1]]
    __m128i ab = _mm_unpacklo_epi64(lo, hi);

    const shuffle_u8::ShuffleRow& row = shuffle_u8::shuf_masks_8[4 * len - 8];
    __m128i mask = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row.data()));
    return _mm_shuffle_epi8(ab, mask);
  }

  // len == 1
  return _mm_loadu_si32(src);
}
#endif

#define DIST_full std::uniform_int_distribution<u64> uniform_dist(0, Vec::size);
#define DIST_redu std::uniform_int_distribution<u64> uniform_dist(1, Vec::size - 1);
#define DIST_N(i) std::uniform_int_distribution<u64> uniform_dist(i, i);

template<typename T>
auto value_distribution() {
  if constexpr (std::floating_point<T>) {
    return std::uniform_real_distribution<T>{};
  } else {
    return std::uniform_int_distribution<T>{};
  }
}

#define BM_OP_I(VALUE, SIZE, SUFFIX, DISTNAME, DIST) \
  void bm_load_part_##SUFFIX##_##VALUE##x##SIZE##_##DISTNAME(benchmark::State& state) { \
    using Vec = be::VectorFor<VALUE, SIZE>; \
    pcg_extras::seed_seq_from<std::random_device> seed_source; \
    pcg32 rng(seed_source); \
    GREX_CAT(DIST_, DIST); \
    std::vector<VALUE> src_vec(1UZ << 28UZ); \
    std::generate(src_vec.begin(), src_vec.end(), \
                  [&] { return value_distribution<VALUE>()(rng); }); \
    std::vector<u64> nvec(1UZ << 28UZ); \
    std::generate(nvec.begin(), nvec.end(), [&] { return uniform_dist(rng); }); \
    std::size_t i = 0; \
    std::size_t j = 0; \
    for (auto _ : state) { \
      if (i >= src_vec.size()) { \
        i = 0; \
      } \
      if (j >= nvec.size()) { \
        j = 0; \
      } \
      auto v = load_part_##SUFFIX(src_vec.data() + i, nvec[j], grex::type_tag<Vec>); \
      benchmark::DoNotOptimize(v); \
      i += SIZE; \
      ++j; \
    } \
  } \
  BENCHMARK(bm_load_part_##SUFFIX##_##VALUE##x##SIZE##_##DISTNAME);

#define BM_OP_grex BM_OP_I
#define BM_OP_xgrex BM_OP_I
#if GREX_X86_64_LEVEL >= 2
#define BM_OP_sse BM_OP_I
#define BM_OP_overlap BM_OP_I
#else
#define BM_OP_sse(...)
#define BM_OP_overlap(...)
#endif
#if GREX_X86_64_LEVEL >= 3
#define BM_OP_table BM_OP_I
#else
#define BM_OP_table(...)
#endif

#define BM_OP(VALUE, SIZE, SUFFIX, DISTNAME, DIST) \
  GREX_CAT(BM_OP_, SUFFIX)(VALUE, SIZE, SUFFIX, DISTNAME, DIST)
#define BM_OP_WRAP(OPSIZE, OPINDEX, VALUE, SIZE, DISTNAME, DIST, ...) \
  BM_OP(VALUE, SIZE, GREX_AT(OPINDEX, __VA_ARGS__), DISTNAME, DIST)
#define BM_OPS(KIND, BITS, SIZE, DISTNAME, DIST, ...) \
  GREX_NREPEAT(GREX_VARIADIC_SIZE(__VA_ARGS__), BM_OP_WRAP, KIND##BITS, SIZE, DISTNAME, DIST, \
               __VA_ARGS__)

#define BM_OPSN(SIZE, PART, KIND, BITS, ...) BM_OPS(KIND, BITS, SIZE, PART, N(PART), __VA_ARGS__)
#define BM_OPS_WRAP(KIND, BITS, SIZE, ...) \
  BM_OPS(KIND, BITS, SIZE, full, full, __VA_ARGS__) \
  BM_OPS(KIND, BITS, SIZE, redu, redu, __VA_ARGS__) \
  GREX_REPEAT(SIZE, BM_OPSN, KIND, BITS, __VA_ARGS__) \
  BM_OPS(KIND, BITS, SIZE, SIZE, N(SIZE), __VA_ARGS__)

// NOLINTBEGIN
// f64
BM_OPS_WRAP(f, 64, 2, grex, sse, table)
// f32
BM_OPS_WRAP(f, 32, 4, grex, table)
BM_OPS_WRAP(f, 32, 2, grex, xgrex)
// i32
BM_OPS_WRAP(i, 32, 4, grex, sse, overlap, table)
BM_OPS_WRAP(i, 32, 2, grex, xgrex)
// u16
BM_OPS_WRAP(u, 16, 8, grex, overlap)
BM_OPS_WRAP(u, 16, 4, grex, xgrex)
BM_OPS_WRAP(u, 16, 2, grex, xgrex)
// u8
BM_OPS_WRAP(u, 8, 16, grex, overlap)
BM_OPS_WRAP(u, 8, 8, grex, xgrex)
BM_OPS_WRAP(u, 8, 4, grex, xgrex)
BM_OPS_WRAP(u, 8, 2, grex, xgrex)
// NOLINTEND

BENCHMARK_MAIN();
