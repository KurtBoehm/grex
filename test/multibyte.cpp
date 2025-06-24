// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <array>
#include <bit>
#include <cstddef>
#include <limits>
#include <random>

#include <fmt/base.h>
#include <fmt/color.h>
#include <pcg_extras.hpp>
#include <thesauros/containers/multi-byte-integers.hpp>
#include <thesauros/ranges/iota.hpp>
#include <thesauros/utility/byte-integer.hpp>

#include "grex/grex.hpp"

#include "defs.hpp"

namespace test = grex::test;

inline constexpr std::size_t mbi_size = 1UL << 15UL;
inline constexpr std::size_t repetitions = 4096;

template<std::size_t tSrc>
void run(test::Rng& rng, grex::IndexTag<tSrc> /*tag*/) {
  static constexpr std::size_t src_bytes = tSrc;
  static constexpr std::size_t dst_bytes = std::bit_ceil(src_bytes);
  using Dst = grex::UnsignedInt<dst_bytes>;
  static constexpr auto sizes = grex::native_sizes<Dst>;
  static constexpr auto padding = grex::register_bits.back() / 8;
  fmt::print(fmt::fg(fmt::terminal_color::magenta) | fmt::text_style(fmt::emphasis::bold),
             "{} → {}, {}\n", src_bytes, dst_bytes, test::type_name<Dst>());

  std::uniform_int_distribution<Dst> dist{
    Dst{},
    (src_bytes == dst_bytes) ? std::numeric_limits<Dst>::max() : Dst(Dst{1} << (8 * src_bytes)),
  };
  thes::MultiByteIntegers<thes::ByteInteger<src_bytes>, padding> mbi(mbi_size);
  for (const auto i : thes::range(mbi_size)) {
    mbi[i] = dist(rng);
  }

  auto op = [&]<std::size_t tSize>(grex::IndexTag<tSize> /*tag*/) {
    fmt::print(fmt::fg(fmt::terminal_color::blue), "{}×{}\n", test::type_name<Dst>(), tSize);
    std::uniform_int_distribution<std::size_t> idist{0, mbi_size - tSize};

    grex::static_apply<tSize>([&]<std::size_t... tIdxs>() {
      for (std::size_t r = 0; r < repetitions; ++r) {
        const std::size_t i = idist(rng);
        test::VectorChecker<Dst, tSize> checker{
          grex::Vector<Dst, tSize>::load_multibyte((mbi.begin() + i).raw(),
                                                   grex::index_tag<src_bytes>),
          std::array{std::as_const(mbi)[i + tIdxs]...},
        };
        checker.check(false);
      }
    });
  };
  grex::static_apply<1, std::bit_width(sizes.back()) + 1>(
    [&]<std::size_t... tIdxs>() { (..., op(grex::index_tag<1U << tIdxs>)); });
}

int main() {
  pcg_extras::seed_seq_from<std::random_device> seed_source{};
  test::Rng rng{seed_source};
  grex::static_apply<8>(
    [&]<std::size_t... tIdxs>() { (..., run(rng, grex::index_tag<tIdxs + 1>)); });
}
