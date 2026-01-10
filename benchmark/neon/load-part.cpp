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
GREX_ALWAYS_INLINE inline TVec load_part_ifs(const typename TVec::Value* ptr, std::size_t size,
                                             grex::TypeTag<TVec> tag) {
  using Value = TVec::Value;
  constexpr std::size_t bytes = sizeof(Value) * TVec::size;

  if (__builtin_constant_p(size)) {
    auto result = zeros(tag);
    bool matched = false;
    grex::static_apply<TVec::size>([&]<std::size_t... tI>() {
      matched =
        (((tI == size) ? (result = be::load_part(ptr, grex::index_tag<tI>, tag), true) : false) ||
         ...);
    });
    return matched ? result : load(ptr, tag);
  }

  if (size >= TVec::size) [[unlikely]] {
    return be::load(ptr, tag);
  }
  auto out = be::zeros(tag).registr();
  if constexpr (sizeof(Value) == 1) {
    if ((size & (1U / sizeof(Value))) != 0) {
      constexpr std::size_t f = 2 / sizeof(Value);
      out = be::load_first<1>(ptr + (size / f * f)).r;
    }
  }
  if constexpr (sizeof(Value) <= 2 && bytes > 2) {
    if ((size & (2U / sizeof(Value))) != 0) {
      constexpr std::size_t f = 4 / sizeof(Value);
      const auto lo = be::load_first<2>(ptr + (size / f * f)).r;
      out = be::as<Value>(vzip1q_u16(be::as<u16>(lo), be::as<u16>(out)));
    }
  }
  if constexpr (sizeof(Value) <= 4 && bytes > 4) {
    if ((size & (4U / sizeof(Value))) != 0) {
      constexpr std::size_t f = 8 / sizeof(Value);
      const auto lo = be::load_first<4>(ptr + (size / f * f)).r;
      out = be::as<Value>(vzip1q_u32(be::as<u32>(lo), be::as<u32>(out)));
    }
  }
  if ((size & (8U / sizeof(Value))) != 0 && bytes > 8) {
    const auto lo = be::load_first<8>(ptr).r;
    out = be::as<Value>(vzip1q_u64(be::as<u64>(lo), be::as<u64>(out)));
  }
  return TVec{out};
}

#define BM_PARTLOAD_ATTR_0
#define BM_PARTLOAD_ATTR_1 [[unlikely]]
#define BM_PARTLOAD_ATTR(INDEX, REF) GREX_CAT(BM_PARTLOAD_ATTR_, GREX_EQUALS(INDEX, REF))

#define BM_PARTLOAD_CASE(SIZE, INDEX, KIND, BITS) \
  BM_PARTLOAD_ATTR(INDEX, 0) case INDEX: \
  return be::load_part(ptr, grex::index_tag<INDEX>, grex::type_tag<be::Vector<KIND##BITS, SIZE>>);
#define GREX_PARTLOAD(KIND, BITS, SIZE) \
  inline be::Vector<KIND##BITS, SIZE> load_part_swi(const KIND##BITS* ptr, std::size_t size, \
                                                    grex::TypeTag<be::Vector<KIND##BITS, SIZE>>) { \
    switch (size) { \
      GREX_REPEAT(SIZE, BM_PARTLOAD_CASE, KIND, BITS) \
      [[unlikely]] BM_PARTLOAD_CASE(SIZE, SIZE, KIND, BITS) default : std::unreachable(); \
    } \
  }
GREX_FOREACH_TYPE(GREX_PARTLOAD, 128)

#define GREX_SUBPARTLOAD_CASE(PART, INDEX, KIND, BITS, SIZE) \
  BM_PARTLOAD_ATTR(INDEX, 0) case INDEX: \
  return Dst{ \
    be::load_part(ptr, grex::index_tag<INDEX>, grex::type_tag<be::Vector<KIND##BITS, SIZE>>)};
#define GREX_SUBPARTLOAD(KIND, BITS, PART, SIZE) \
  inline be::SubVector<KIND##BITS, PART, SIZE> load_part_swi( \
    const KIND##BITS* ptr, std::size_t size, \
    grex::TypeTag<be::SubVector<KIND##BITS, PART, SIZE>>) { \
    using Dst = be::SubVector<KIND##BITS, PART, SIZE>; \
    switch (size) { \
      GREX_REPEAT(PART, GREX_SUBPARTLOAD_CASE, KIND, BITS, SIZE) \
      [[unlikely]] GREX_SUBPARTLOAD_CASE(PART, PART, KIND, BITS, SIZE) default \
          : std::unreachable(); \
    } \
  }
GREX_FOREACH_SUB(GREX_SUBPARTLOAD)

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
  static void bm_load_part_##SUFFIX##_##VALUE##x##SIZE##_##DISTNAME(benchmark::State& state) { \
    using Vec = be::Vector<VALUE, SIZE>; \
    pcg_extras::seed_seq_from<std::random_device> seed_source; \
    pcg32 rng(seed_source); \
    DIST; \
    std::vector<VALUE> src_vec(1UZ << 28UZ); \
    std::generate(src_vec.begin(), src_vec.end(), \
                  [&] { return value_distribution<VALUE>()(rng); }); \
    std::size_t i = 0; \
    for (auto _ : state) { \
      if (i >= src_vec.size()) { \
        i = 0; \
      } \
      load_part_##SUFFIX(src_vec.data() + i, uniform_dist(rng), grex::type_tag<Vec>); \
      i += SIZE; \
    } \
  } \
  BENCHMARK(bm_load_part_##SUFFIX##_##VALUE##x##SIZE##_##DISTNAME);

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
