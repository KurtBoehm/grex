// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <bit>
#include <concepts>
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

namespace test = grex::test;
inline constexpr std::size_t repetitions = 4096;
template<typename T>
using Distribution = std::conditional_t<std::floating_point<T>, std::uniform_real_distribution<T>,
                                        std::uniform_int_distribution<T>>;

template<grex::Vectorizable TValue>
void gather(test::Rng& rng, grex::TypeTag<TValue> /*tag*/) {
  fmt::print(fmt::fg(fmt::terminal_color::magenta) | fmt::text_style(fmt::emphasis::bold),
             "value: {}\n", test::type_name<TValue>());
  constexpr std::size_t data_size = 3 * (std::size_t(1) << 32) / sizeof(TValue);

  const auto data = std::make_unique<TValue[]>(data_size);
  auto vdist = test::make_distribution<TValue>();
#pragma omp parallel for default(shared) private(vdist, rng) schedule(guided)
  for (std::size_t i = 0; i < data_size; ++i) {
    data[i] = vdist(rng);
  }
  const std::span<const TValue> sdata{data.get(), data_size};

  auto outer = [&]<grex::Vectorizable TIndex>(grex::TypeTag<TIndex> /*tag*/) {
    fmt::print(fmt::fg(fmt::terminal_color::blue) | fmt::text_style(fmt::emphasis::bold),
               "index: {}\n", test::type_name<TIndex>());

    const auto imax = std::size_t(std::numeric_limits<TIndex>::max());
    std::uniform_int_distribution<TIndex> idist{0, std::min(data_size - 1, imax)};
    auto ival = [&](std::size_t /*dummy*/) { return idist(rng); };
    std::uniform_int_distribution<int> mdist{0, 1};
    auto mval = [&](std::size_t /*dummy*/) { return bool(mdist(rng)); };

    auto op = [&]<std::size_t tSize>(grex::IndexTag<tSize> /*tag*/) {
      std::uniform_int_distribution<std::size_t> pdist{0, tSize};

      for (std::size_t i = 0; i < repetitions; ++i) {
        grex::static_apply<tSize>([&]<std::size_t... tIdxs> {
          test::VectorChecker<TIndex, tSize> idxs{ival(tIdxs)...};
          // gather
          {
            test::VectorChecker<TValue, tSize> gathered{
              grex::gather(sdata, idxs.vec),
              {sdata[std::size_t(idxs.ref[tIdxs])]...},
            };
            gathered.check("gather", false);
          }
          {
            test::VectorChecker<TValue, tSize> gathered{
              grex::gather(sdata, idxs.vec, grex::typed_full_tag<TValue, tSize>),
              {sdata[std::size_t(idxs.ref[tIdxs])]...},
            };
            gathered.check("gather tagged", false);
          }
          // mask_gather
          {
            test::MaskChecker<TValue, tSize> m{mval(tIdxs)...};
            test::VectorChecker<TValue, tSize> gathered{
              grex::mask_gather(sdata, m.mask, idxs.vec),
              {(m.ref[tIdxs] ? sdata[std::size_t(idxs.ref[tIdxs])] : TValue{})...},
            };
            gathered.check("mask_gather", false);
          }
          {
            const std::size_t part = pdist(rng);
            const test::VectorChecker<TValue, tSize> gathered{
              grex::gather(sdata, idxs.vec, grex::part_tag<tSize>(part)),
              {((tIdxs < part) ? sdata[std::size_t(idxs.ref[tIdxs])] : TValue{})...},
            };
            gathered.check("gather part tagged", false);
          }
          {
            test::MaskChecker<TValue, tSize> m{mval(tIdxs)...};
            test::VectorChecker<TValue, tSize> gathered{
              grex::gather(sdata, idxs.vec, grex::typed_masked_tag(m.mask)),
              {(m.ref[tIdxs] ? sdata[std::size_t(idxs.ref[tIdxs])] : TValue{})...},
            };
            gathered.check("gather masked tagged", false);
          }
        });
      }
    };

    constexpr std::size_t size =
      std::min(grex::native_sizes<TValue>.back(), grex::native_sizes<TIndex>.back());
    grex::static_apply<1, std::bit_width(size) + 2>(
      [&]<std::size_t... tSizes> { (..., op(grex::index_tag<1ULL << tSizes>)); });
  };
  test::for_each_integral(outer);
}

int main() {
  pcg_extras::seed_seq_from<std::random_device> seed_source{};
  test::Rng rng{seed_source};
  test::for_each_type([&](auto vtag) { gather(rng, vtag); });
}
