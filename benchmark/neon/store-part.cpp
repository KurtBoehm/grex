#include <cstddef>
#include <random>

#include <benchmark/benchmark.h>
#include <pcg_extras.hpp>
#include <pcg_random.hpp>
#include <vector>

#include "grex/grex.hpp"

using namespace grex::primitives;
namespace be = grex::backend;

template<be::AnyVector TVec>
requires(be::AnyNativeVector<TVec> || be::AnySubNativeVector<TVec>)
[[gnu::noinline]] void store_part_ifs(typename TVec::Value* dst, TVec src, std::size_t size) {
  using Value = TVec::Value;
  constexpr std::size_t bytes = sizeof(Value) * TVec::size;

  if (__builtin_constant_p(size)) {
    bool matched = false;
    grex::static_apply<TVec::size>([&]<std::size_t... tI>() {
      matched =
        (((tI == size) ? (be::store_part(dst, src, grex::index_tag<tI>), true) : false) || ...);
    });
    if (!matched) {
      be::store(dst, src);
    }
    return;
  }

  if (size >= TVec::size) [[unlikely]] {
    be::store(dst, src);
    return;
  }

  if constexpr (bytes > 8) {
    if ((size & (8U / sizeof(Value))) != 0 && bytes > 8) {
      be::store_first<8>(dst, src);
      src = TVec{be::as<Value>(vdupq_laneq_u64(be::as<u64>(src.registr()), 1))};
    }
  }
  if constexpr (sizeof(Value) <= 4 && bytes > 4) {
    if ((size & (4U / sizeof(Value))) != 0) {
      constexpr std::size_t f = 8 / sizeof(Value);
      be::store_first<4>(dst + (size / f * f), src);
      src = TVec{be::as<Value>(vdupq_laneq_u32(be::as<u32>(src.registr()), 1))};
    }
  }
  if constexpr (sizeof(Value) <= 2 && bytes > 2) {
    if ((size & (2U / sizeof(Value))) != 0) {
      constexpr std::size_t f = 4 / sizeof(Value);
      be::store_first<2>(dst + (size / f * f), src);
      src = TVec{be::as<Value>(vdupq_laneq_u16(be::as<u16>(src.registr()), 1))};
    }
  }
  if constexpr (sizeof(Value) == 1) {
    if ((size & (1U / sizeof(Value))) != 0) {
      constexpr std::size_t f = 2 / sizeof(Value);
      be::store_first<1>(dst + (size / f * f), src);
    }
  }
}

#define BM_PARTSTORE_ATTR_0
#define BM_PARTSTORE_ATTR_1 [[unlikely]]
#define BM_PARTSTORE_ATTR(INDEX, REF) GREX_CAT(BM_PARTSTORE_ATTR_, GREX_EQUALS(INDEX, REF))

#define BM_PARTSTORE_CASE(SIZE, INDEX, KIND, BITS) \
  BM_PARTSTORE_ATTR(INDEX, 0) case INDEX: { \
    be::store_part(dst, src, grex::index_tag<INDEX>); \
    return; \
  }
#define BM_PARTSTORE(KIND, BITS, SIZE) \
  [[gnu::noinline]] void store_part_swi(KIND##BITS* dst, be::Vector<KIND##BITS, SIZE> src, \
                                        std::size_t size) { \
    switch (size) { \
      GREX_REPEAT(SIZE, BM_PARTSTORE_CASE, KIND, BITS) \
      [[unlikely]] BM_PARTSTORE_CASE(SIZE, SIZE, KIND, BITS) default : std::unreachable(); \
    } \
  }
GREX_FOREACH_TYPE(BM_PARTSTORE, 128)

#define BM_SUBPARTSTORE(KIND, BITS, PART, SIZE) \
  [[gnu::noinline]] void store_part_switch( \
    KIND##BITS* dst, be::SubVector<KIND##BITS, PART, SIZE> src, std::size_t size) { \
    switch (size) { \
      GREX_REPEAT(PART, BM_PARTSTORE_CASE, KIND, BITS) \
      [[unlikely]] BM_PARTSTORE_CASE(PART, PART, KIND, BITS) default : std::unreachable(); \
    } \
  }
GREX_FOREACH_SUB(BM_SUBPARTSTORE)

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
  static void bm_store_part_##SUFFIX##_##VALUE##x##SIZE##_##DISTNAME(benchmark::State& state) { \
    using Vec = be::Vector<VALUE, SIZE>; \
    pcg_extras::seed_seq_from<std::random_device> seed_source; \
    pcg32 rng(seed_source); \
    DIST; \
    std::vector<VALUE> src_vec(1UZ << 28UZ); \
    std::generate(src_vec.begin(), src_vec.end(), \
                  [&] { return value_distribution<VALUE>()(rng); }); \
    std::array<VALUE, SIZE> arr{}; \
    std::size_t i = 0; \
    for (auto _ : state) { \
      if (i >= src_vec.size()) { \
        i = 0; \
      } \
      store_part_##SUFFIX(arr.data(), be::load(src_vec.data() + i, grex::type_tag<Vec>), \
                          uniform_dist(rng)); \
      i += SIZE; \
    } \
  } \
  BENCHMARK(bm_store_part_##SUFFIX##_##VALUE##x##SIZE##_##DISTNAME);

#define BM_OPS(VALUE, SIZE, DISTNAME, DIST) \
  BM_OP(VALUE, SIZE, ifs, DISTNAME, DIST) \
  BM_OP(VALUE, SIZE, swi, DISTNAME, DIST)
#define BM_OPSN(SIZE, PART, VALUE) BM_OPS(VALUE, SIZE, PART, DISTN(PART))
#define BM_OPS_WRAP(VALUE, SIZE) \
  BM_OPS(VALUE, SIZE, full, DIST_full) \
  BM_OPS(VALUE, SIZE, redu, DIST_redu) \
  GREX_REPEAT(SIZE, BM_OPSN, VALUE) \
  BM_OPS(VALUE, SIZE, SIZE, DISTN(SIZE))

// NOLINTBEGIN
BM_OPS_WRAP(f64, 2)
BM_OPS_WRAP(f32, 4)
BM_OPS_WRAP(u16, 8)
BM_OPS_WRAP(u8, 16)
// NOLINTEND

BENCHMARK_MAIN();
