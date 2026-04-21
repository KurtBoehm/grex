#include <cstddef>
#include <memory>
#include <random>
#include <span>

#include <benchmark/benchmark.h>
#include <pcg_extras.hpp>
#include <pcg_random.hpp>

#include "grex/grex.hpp"

namespace grex::backend {
template<std::size_t tSrc, typename TDst, std::size_t tSize>
requires(tSrc < sizeof(TDst) && (sizeof(TDst) * tSize) == 64)
inline NativeVector<TDst, tSize> ldmb(const u8* ptr, IndexTag<tSrc> /*src*/,
                                      TypeTag<NativeVector<TDst, tSize>> /*dst*/) {
  // The comments are based on M == 5; M == 6 and 7 are analogous.

  static constexpr std::size_t src_bytes = tSrc;
  static constexpr std::size_t dst_bytes = sizeof(TDst);
  // Total padding across all elements; always a multiple of 8.
  // offset = (dst_bytes - src_bytes) * tSize = (8 - 5) * 8 = 24
  static constexpr std::size_t offset = (dst_bytes - src_bytes) * tSize;

  // Load with an offset of half the total padding so that each half of the elements ends up in the
  // correct 256-bit half of the 512-bit register.
  // ........|....0000|01111122|22233333|44444555|55666667|7777....|........
  __m512i out = load(ptr - offset / 2, type_tag<NativeVector<u8, dst_bytes * tSize>>).r;

  // First, permute 32-bit chunks into the correct 128-bit lanes.
  //
  // - The two middle 128-bit lanes stay where they are.
  // - The outer lanes are moved by offset / 2 bytes to the very beginning/end.
  //
  // After this, lanes 0 and 2 (even indices) are bottom-aligned, while lanes 1 and 3
  // are top-aligned within their 128-bit lanes.
  // 00000111|11222223|01111122|22233333|44444555|55666667|45555566|66677777
  const __m512i idxs32 = static_apply<16>([]<std::size_t... tIdxs>() {
    return set(type_tag<NativeVector<i32, 16>>,
               i32{(tIdxs < 4) ? (tIdxs + offset / 8)
                               : ((tIdxs >= 12) ? (tIdxs - offset / 8) : tIdxs)}...)
      .r;
  });
  out = _mm512_permutexvar_epi32(idxs32, out);

  // Then perform a byte-wise shuffle within each 128-bit lane.
  //
  // For each output byte index i in [0, 64), we:
  // 1. Check if it is inside the padding; if so, use -1 (zero).
  // 2. Otherwise compute the source byte index as a sum of:
  //    (a) index within the element (`i % dst_bytes`, bounded by src_bytes),
  //    (b) element offset within the 128-bit lane,
  //    (c) additional offset for top-aligned lanes (odd lane indices).
  //
  // In the example (M == 5, N == 8, tSize == 8), these components look like:
  // (a) 01234...|01234...|01234...|01234...|01234...|01234...|01234...|01234...
  // (b) 00000...|55555...|00000...|55555...|00000...|55555...|00000...|55555...
  // (c) 00000...|00000...|66666...|66666...|00000...|00000...|66666...|66666...
  const __m512i idxs8 = static_apply<64>([]<std::size_t... tIdxs>() {
    return set(type_tag<NativeVector<i8, 64>>,
               ((tIdxs % dst_bytes < src_bytes)
                  ? i8{tIdxs % dst_bytes + ((tIdxs % 16) / dst_bytes) * src_bytes +
                       ((tIdxs / 16) % 2) * (offset / 4)}
                  : i8{-1})...)
      .r;
  });
  return {.r = _mm512_shuffle_epi8(out, idxs8)};
}

#if GREX_HAS_AVX512VBMI
template<std::size_t tSrc, typename TDst, std::size_t tSize>
requires(tSrc < sizeof(TDst) && (sizeof(TDst) * tSize) == 64)
inline NativeVector<TDst, tSize> ldmb_vpermb(const u8* ptr, IndexTag<tSrc> /*src*/,
                                             TypeTag<NativeVector<TDst, tSize>> /*dst*/) {
  // The comments are based on M == 5; M == 6 and 7 are analogous.

  static constexpr std::size_t src_bytes = tSrc;
  static constexpr std::size_t dst_bytes = sizeof(TDst);

  // Load a full 512-bit vector.
  // 00000111|11222223|33334444|45555566|66677777|........|........|........
  __m512i src = load(ptr, type_tag<NativeVector<u8, dst_bytes * tSize>>).r;

  // Mask out the padding bits.
  constexpr __mmask64 k = static_apply<64>(
    []<std::size_t... tI>() { return (... | (__mmask64(tI % dst_bytes < src_bytes) << tI)); });

  // Perform a byte-wise shuffle across 128-bit lanes.
  const __m512i idxs8 = static_apply<64>([]<std::size_t... tIdxs>() {
    return set(type_tag<NativeVector<i8, 64>>,
               ((tIdxs % dst_bytes < src_bytes)
                  ? i8{tIdxs % dst_bytes + (tIdxs / dst_bytes) * src_bytes}
                  : i8{-1})...)
      .r;
  });
  return {.r = _mm512_maskz_permutexvar_epi8(k, idxs8, src)};
}

template<std::size_t tSrc, typename TDst, std::size_t tSize>
requires(tSrc < sizeof(TDst) && (sizeof(TDst) * tSize) == 64)
inline NativeVector<TDst, tSize> ldmb_vpermt2b(const u8* ptr, IndexTag<tSrc> /*src*/,
                                               TypeTag<NativeVector<TDst, tSize>> /*dst*/) {
  // The comments are based on M == 5; M == 6 and 7 are analogous.

  static constexpr std::size_t src_bytes = tSrc;
  static constexpr std::size_t dst_bytes = sizeof(TDst);

  // Load a full 512-bit vector.
  // 00000111|11222223|33334444|45555566|66677777|........|........|........
  __m512i src = load(ptr, type_tag<NativeVector<u8, dst_bytes * tSize>>).r;

  // Perform a byte-wise shuffle across 128-bit lanes.
  const __m512i idxs8 = static_apply<64>([]<std::size_t... tIdxs>() {
    return set(type_tag<NativeVector<i8, 64>>,
               ((tIdxs % dst_bytes < src_bytes)
                  ? i8{tIdxs % dst_bytes + (tIdxs / dst_bytes) * src_bytes}
                  : i8{-1})...)
      .r;
  });
  return {.r = _mm512_permutex2var_epi8(src, idxs8, _mm512_setzero_si512())};
}
#endif
} // namespace grex::backend

using namespace grex::primitives;
namespace be = grex::backend;

namespace {
inline constexpr std::size_t buffer_size = 1UZ << 20UZ;
inline constexpr std::size_t vector_bytes = 64;

std::unique_ptr<u8[]> make_buffer(std::size_t size) {
  const auto psize = size + 2 * vector_bytes;

  std::unique_ptr<u8[]> buf = std::make_unique_for_overwrite<u8[]>(psize);
  pcg_extras::seed_seq_from<std::random_device> seed_source;
  pcg32 rng(seed_source);

  std::uniform_int_distribution<u8> dist(0, 255);
  for (auto& b : std::span{buf.get(), psize}) {
    b = dist(rng);
  }
  return buf;
}

template<std::size_t tSrc, typename TDst, std::size_t tSize>
void bm_ldmb_base(benchmark::State& state, auto op) {
  auto buffer = make_buffer(buffer_size); // 1 MiB
  const u8* base = buffer.get() + vector_bytes;

  std::size_t pos = 0;
  be::NativeVector<TDst, tSize> acc{};

  for (auto _ : state) {
    const u8* ptr = base + pos;
    acc = op(ptr, grex::index_tag<tSrc>, grex::type_tag<be::NativeVector<TDst, tSize>>);
    pos += vector_bytes;
    if (pos + vector_bytes >= buffer_size) {
      pos = 0;
    }
    benchmark::DoNotOptimize(acc);
  }
  benchmark::DoNotOptimize(acc);
}

#define BM_LDMB_NOVBMI(SRC, DST, SIZE) \
  void bm_ldmb_novbmi_##SRC##_##DST##_##SIZE(benchmark::State& state) { \
    bm_ldmb_base<SRC, DST, SIZE>(state, [](const u8* ptr, grex::AnyIndexTag auto src, auto tag) { \
      return be::ldmb(ptr, src, tag); \
    }); \
  } \
  BENCHMARK(bm_ldmb_novbmi_##SRC##_##DST##_##SIZE)

#define BM_LDMB_VPERMB(SRC, DST, SIZE) \
  void bm_ldmb_vpermb_##SRC##_##DST##_##SIZE(benchmark::State& state) { \
    bm_ldmb_base<SRC, DST, SIZE>(state, [](const u8* ptr, grex::AnyIndexTag auto src, auto tag) { \
      return be::ldmb_vpermb(ptr, src, tag); \
    }); \
  } \
  BENCHMARK(bm_ldmb_vpermb_##SRC##_##DST##_##SIZE); \
  void bm_ldmb_vpermt2b_##SRC##_##DST##_##SIZE(benchmark::State& state) { \
    bm_ldmb_base<SRC, DST, SIZE>(state, [](const u8* ptr, grex::AnyIndexTag auto src, auto tag) { \
      return be::ldmb_vpermt2b(ptr, src, tag); \
    }); \
  } \
  BENCHMARK(bm_ldmb_vpermt2b_##SRC##_##DST##_##SIZE)

#if GREX_HAS_AVX512VBMI
#define BM_LDBM(SRC, DST, SIZE) \
  BM_LDMB_NOVBMI(SRC, DST, SIZE); \
  BM_LDMB_VPERMB(SRC, DST, SIZE);
#else
#define BM_LDBM(SRC, DST, SIZE) BM_LDMB_NOVBMI(SRC, DST, SIZE);
#endif

BM_LDBM(3, u32, 16) // NOLINT
BM_LDBM(5, u64, 8) // NOLINT
BM_LDBM(6, u64, 8) // NOLINT
BM_LDBM(7, u64, 8) // NOLINT
} // namespace

BENCHMARK_MAIN();
