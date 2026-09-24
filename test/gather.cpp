// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <random>
#include <span>
#include <type_traits>

#include <fmt/base.h>
#include <fmt/color.h>
#include <pcg_extras.hpp>

#include "grex/grex.hpp"

#include "defs.hpp"

#if !GREX_BACKEND_SCALAR
#include <bit>
#endif

namespace {
namespace test = grex::test;
inline constexpr std::size_t repetitions = 4096;
template<typename T>
using Distribution =
  std::conditional_t<grex::FloatVectorizable<T>, std::uniform_real_distribution<T>,
                     std::uniform_int_distribution<T>>;

#if !GREX_BACKEND_SCALAR
template<grex::Vectorizable V>
void run_simd(test::Rng& rng, grex::TypeTag<V> /*tag*/) {
  fmt::print(fmt::fg(fmt::terminal_color::magenta) | fmt::emphasis::bold, "value: {}\n",
             test::type_name<V>());
  constexpr std::size_t data_size = 3 * (1UZ << 32UZ) / sizeof(V);

  const auto data = std::make_unique<V[]>(data_size);
  auto vdist = test::make_distribution<V>(); // NOLINT(*-const-correctness)
#pragma omp parallel for default(none) shared(data) private(vdist, rng) schedule(guided)
  for (std::size_t i = 0; i < data_size; ++i) {
    data[i] = vdist(rng);
  }
  const std::span<const V, data_size> sdata{data.get(), data_size};

  const auto outer = [&]<grex::Vectorizable Index>(grex::TypeTag<Index> /*tag*/) {
    fmt::print(fmt::fg(fmt::terminal_color::blue) | fmt::emphasis::bold, "index: {}\n",
               test::type_name<Index>());

    const auto imax = std::size_t(std::numeric_limits<Index>::max());
    std::uniform_int_distribution<Index> idist{0, std::min(data_size - 1, imax)};
    auto ival = [&] { return idist(rng); };
    std::uniform_int_distribution<int> mdist{0, 1};
    auto mval = [&](std::size_t /*dummy*/) { return static_cast<bool>(mdist(rng)); };

    auto op = [&]<std::size_t N>(grex::IndexTag<N> /*tag*/) {
      std::uniform_int_distribution<std::size_t> pdist{0, N};

      for (std::size_t i = 0; i < repetitions; ++i) {
        grex::static_apply<N>([&]<std::size_t... I> {
          auto idxs = test::VectorChecker<Index, N>::random(ival);
          // gather
          {
            const test::VectorChecker<V, N> gathered{
              grex::gather(sdata, idxs.vec),
              {sdata[std::size_t(idxs.ref[I])]...},
            };
            gathered.check("gather", {.verbose = false});
          }
          {
            const test::VectorChecker<V, N> gathered{
              grex::gather(sdata, idxs.vec, grex::typed_full_tag<V, N>),
              {sdata[std::size_t(idxs.ref[I])]...},
            };
            gathered.check("gather tagged", {.verbose = false});
          }
          // mask_gather
          {
            const test::MaskChecker<V, N> m{mval(I)...};
            const test::VectorChecker<V, N> gathered{
              grex::mask_gather(sdata, m.mask, idxs.vec),
              {(m.ref[I] ? sdata[std::size_t(idxs.ref[I])] : V{})...},
            };
            gathered.check("mask_gather", {.verbose = false});
          }
          {
            const std::size_t part = pdist(rng);
            const test::VectorChecker<V, N> gathered{
              grex::gather(sdata, idxs.vec, grex::part_tag<N>(part)),
              {((I < part) ? sdata[std::size_t(idxs.ref[I])] : V{})...},
            };
            gathered.check("gather part tagged", {.verbose = false});
          }
          {
            const test::MaskChecker<V, N> m{mval(I)...};
            const test::VectorChecker<V, N> gathered{
              grex::gather(sdata, idxs.vec, grex::typed_masked_tag(m.mask)),
              {(m.ref[I] ? sdata[std::size_t(idxs.ref[I])] : V{})...},
            };
            gathered.check("gather masked tagged", {.verbose = false});
          }
        });
      }
    };

    constexpr std::size_t size = std::min(grex::max_native_size<V>, grex::max_native_size<Index>);
    grex::static_apply<1, std::bit_width(size) + 2>(
      [&]<std::size_t... Ns> { (..., op(grex::index_tag<1ULL << Ns>)); });
  };
  test::for_each_integral(outer);
}
#endif
template<grex::Vectorizable V>
void run_scalar(test::Rng& rng, grex::TypeTag<V> /*tag*/) {
  fmt::print(fmt::fg(fmt::terminal_color::magenta) | fmt::emphasis::bold, "value: {}\n",
             test::type_name<V>());
  constexpr std::size_t data_size = 3 * (1UZ << 32UZ) / sizeof(V);

  const auto data = std::make_unique<V[]>(data_size);
  auto vdist = test::make_distribution<V>(); // NOLINT(*-const-correctness)
#pragma omp parallel for default(none) shared(data) private(vdist, rng) schedule(guided)
  for (std::size_t i = 0; i < data_size; ++i) {
    data[i] = vdist(rng);
  }
  const std::span<const V, data_size> sdata{data.get(), data_size};

  const auto outer = [&]<grex::Vectorizable Index>(grex::TypeTag<Index> /*tag*/) {
    fmt::print(fmt::fg(fmt::terminal_color::blue) | fmt::emphasis::bold, "index: {}\n",
               test::type_name<Index>());

    const auto imax = std::size_t(std::numeric_limits<Index>::max());
    std::uniform_int_distribution<Index> idist{0, std::min(data_size - 1, imax)};
    std::uniform_int_distribution<int> bdist{0, 1};

    for (std::size_t i = 0; i < repetitions; ++i) {
      const Index idx = idist(rng);
      // gather
      {
        const V a = grex::gather(sdata, idx, grex::scalar_tag);
        const V b = sdata[std::size_t(idx)];
        test::check("gather scalar", a, b, {.verbose = false});
      }
      // mask_gather
      {
        const bool m = static_cast<bool>(bdist(rng));
        const V a = grex::mask_gather(sdata, m, idx, grex::scalar_tag);
        const V b = m ? sdata[std::size_t(idx)] : V{};
        test::check("mask_gather scalar", a, b, {.verbose = false});
      }
    }
  };
  test::for_each_integral(outer);
}
} // namespace

int main() {
  pcg_extras::seed_seq_from<std::random_device> seed_source{};
  test::Rng rng{seed_source};
  test::for_each_type([&](auto tag) {
#if !GREX_BACKEND_SCALAR
    run_simd(rng, tag);
#endif
    run_scalar(rng, tag);
  });
}
