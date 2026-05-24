#include <algorithm>
#include <bit>
#include <concepts>
#include <cstddef>
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
GREX_ALWAYS_INLINE inline __m128i load_part_sse(const f32* ptr, std::size_t size,
                                                grex::TypeTag<be::f32x4> /*tag*/) {
  return load_part_sse(reinterpret_cast<const i32*>(ptr), size, grex::type_tag<be::i32x4>);
}
#endif

#define DIST_full std::uniform_int_distribution<u64> uniform_dist(0, Vec::size);
#define DIST_redu std::uniform_int_distribution<u64> uniform_dist(1, Vec::size - 1);
#define DISTN(i) std::uniform_int_distribution<u64> uniform_dist(i, i);

template<typename T>
auto value_distribution() {
  if constexpr (std::floating_point<T>) {
    return std::uniform_real_distribution<T>{};
  } else {
    return std::uniform_int_distribution<T>{};
  }
}

#define BM_OP(VALUE, SIZE, SUFFIX, DISTNAME, DIST) \
  void bm_load_part_##SUFFIX##_##VALUE##x##SIZE##_##DISTNAME(benchmark::State& state) { \
    using Vec = be::NativeVector<VALUE, SIZE>; \
    pcg_extras::seed_seq_from<std::random_device> seed_source; \
    pcg32 rng(seed_source); \
    DIST; \
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

#if GREX_X86_64_LEVEL >= 3
#define BM_OPS(VALUE, SIZE, DISTNAME, DIST) \
  BM_OP(VALUE, SIZE, grex, DISTNAME, DIST) \
  BM_OP(VALUE, SIZE, table, DISTNAME, DIST) \
  BM_OP(VALUE, SIZE, sse, DISTNAME, DIST)
#elif GREX_X86_64_LEVEL >= 2
#define BM_OPS(VALUE, SIZE, DISTNAME, DIST) \
  BM_OP(VALUE, SIZE, grex, DISTNAME, DIST) \
  BM_OP(VALUE, SIZE, sse, DISTNAME, DIST)
#else
#define BM_OPS(VALUE, SIZE, DISTNAME, DIST) BM_OP(VALUE, SIZE, grex, DISTNAME, DIST)
#endif

#define BM_OPSN(SIZE, PART, VALUE) BM_OPS(VALUE, SIZE, PART, DISTN(PART))
#define BM_OPS_WRAP(VALUE, SIZE) \
  BM_OPS(VALUE, SIZE, full, DIST_full) \
  BM_OPS(VALUE, SIZE, redu, DIST_redu) \
  GREX_REPEAT(SIZE, BM_OPSN, VALUE) \
  BM_OPS(VALUE, SIZE, SIZE, DISTN(SIZE))

// NOLINTBEGIN
BM_OPS_WRAP(f64, 2)
BM_OPS_WRAP(f32, 4)
// BM_OPS_WRAP(u16, 8)
// BM_OPS_WRAP(u8, 16)
BM_OPS_WRAP(i32, 4)
// NOLINTEND

BENCHMARK_MAIN();
