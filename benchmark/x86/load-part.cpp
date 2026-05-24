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
#endif

#if GREX_X86_64_LEVEL == 2
be::VectorFor<u16, 16> load_part_overlap(const u16* src, std::size_t len,
                                         grex::TypeTag<be::VectorFor<u16, 16>> /*tag*/) {
  if (len == 0) [[unlikely]] {
    return {.lower = {.r = _mm_setzero_si128()}, .upper = {.r = _mm_setzero_si128()}};
  }
  if (len >= 16) [[unlikely]] {
    return {
      .lower = {.r = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src))},
      .upper = {.r = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 8))},
    };
  }

  // 16-byte block path: len ∈ [8,16]
  if (len >= 8) {
    __m128i lo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src));
    __m128i hi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + (len - 8)));

    const auto& row = be::shld::idxs16[2 * len - 16];
    __m128i mask = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row.data()));
    return {.lower = {.r = lo}, .upper = {.r = _mm_shuffle_epi8(hi, mask)}};
  }

  return {
    .lower = be::load_part(src, len, grex::type_tag<be::u16x8>),
    .upper = {.r = _mm_setzero_si128()},
  };
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
#else
#define BM_OP_sse(...)
#endif
#if GREX_X86_64_LEVEL == 2
#define BM_OP_overlap BM_OP_I
#else
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
BM_OPS_WRAP(i, 32, 4, grex, sse, table)
BM_OPS_WRAP(i, 32, 2, grex, xgrex)
// u16
BM_OPS_WRAP(u, 16, 16, grex, overlap)
BM_OPS_WRAP(u, 16, 8, grex)
BM_OPS_WRAP(u, 16, 4, grex, xgrex)
BM_OPS_WRAP(u, 16, 2, grex, xgrex)
// u8
BM_OPS_WRAP(u, 8, 16, grex)
BM_OPS_WRAP(u, 8, 8, grex, xgrex)
BM_OPS_WRAP(u, 8, 4, grex, xgrex)
BM_OPS_WRAP(u, 8, 2, grex, xgrex)
// NOLINTEND

BENCHMARK_MAIN();
