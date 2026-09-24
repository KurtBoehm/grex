// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <bit>
#include <cstddef>
#include <limits>
#include <random>

#include <fmt/base.h>
#include <fmt/color.h>
#include <pcg_extras.hpp>
#include <thesauros/containers/multi-byte-integers.hpp>
#include <thesauros/ranges/indices.hpp>
#include <thesauros/utility/byte-integer.hpp>

#include "grex/grex.hpp"

#include "defs.hpp"

#if !GREX_BACKEND_SCALAR
#include <array>
#endif

namespace {
namespace test = grex::test;

inline constexpr std::size_t mbi_size = 1UL << 15UL;
inline constexpr std::size_t repetitions = 4096;

#if !GREX_BACKEND_SCALAR
template<std::size_t Src>
void run_simd(test::Rng& rng, grex::IndexTag<Src> /*tag*/) {
  static constexpr std::size_t src_bytes = Src;
  static constexpr std::size_t dst_bytes = std::bit_ceil(src_bytes);
  using Dst = grex::UnsignedInt<dst_bytes>;
  static constexpr auto sizes = grex::native_sizes<Dst>;
  static constexpr auto padding = grex::register_bits.back() / 8;
  fmt::print(fmt::fg(fmt::terminal_color::magenta) | fmt::emphasis::bold, "{} → {}, {}\n",
             src_bytes, dst_bytes, test::type_name<Dst>());

  std::uniform_int_distribution<Dst> dist{
    Dst{},
    (src_bytes == dst_bytes) ? std::numeric_limits<Dst>::max() : Dst(Dst{1} << (8 * src_bytes)),
  };
  thes::MultiByteIntegers<thes::ByteInteger<src_bytes>, padding> mbi(mbi_size);
  for (const auto i : thes::views::indices(mbi_size)) {
    mbi[i] = dist(rng);
  }

  auto op = [&]<std::size_t N>(grex::IndexTag<N> /*tag*/) {
    fmt::print(fmt::fg(fmt::terminal_color::blue), "{}×{}\n", test::type_name<Dst>(), N);
    std::uniform_int_distribution<std::ptrdiff_t> idist{
      0,
      static_cast<std::ptrdiff_t>(mbi_size - N),
    };

    grex::static_apply<N>([&]<std::size_t... I> {
      for (std::size_t r = 0; r < repetitions; ++r) {
        const auto it = std::as_const(mbi).begin() + idist(rng);

        {
          const test::VectorChecker<Dst, N> checker{
            grex::Vector<Dst, N>::load_multibyte(it),
            std::array{it[I]...},
          };
          checker.check("load_multibyte vector/thesauros", {.verbose = false});
        }
        {
          const test::VectorChecker<Dst, N> checker{
            grex::Vector<Dst, N>::load_multibyte(it),
            std::array{grex::load_multibyte(it + I, grex::scalar_tag)...},
          };
          checker.check("load_multibyte vector/tagged scalar", {.verbose = false});
        }
        {
          const test::VectorChecker<Dst, N> checker{
            grex::load_multibyte(it, grex::full_tag<N>),
            std::array{grex::load_multibyte(it + I, grex::scalar_tag)...},
          };
          checker.check("load_multibyte tagged vector/tagged scalar", {.verbose = false});
        }
      }
    });
  };
  grex::static_apply<1, std::bit_width(sizes.back()) + 1>(
    [&]<std::size_t... I> { (..., op(grex::index_tag<1U << I>)); });
}
#endif
template<std::size_t Src>
void run_scalar(test::Rng& rng, grex::IndexTag<Src> /*tag*/) {
  static constexpr std::size_t src_bytes = Src;
  static constexpr std::size_t dst_bytes = std::bit_ceil(src_bytes);
  using Dst = grex::UnsignedInt<dst_bytes>;
  fmt::print(fmt::fg(fmt::terminal_color::magenta) | fmt::emphasis::bold, "{} → {}, {}\n",
             src_bytes, dst_bytes, test::type_name<Dst>());

  std::uniform_int_distribution<Dst> dist{
    Dst{},
    (src_bytes == dst_bytes) ? std::numeric_limits<Dst>::max() : Dst(Dst{1} << (8 * src_bytes)),
  };
  thes::MultiByteIntegers<thes::ByteInteger<src_bytes>, std::bit_ceil(src_bytes)> mbi(mbi_size);
  for (const auto i : thes::views::indices(mbi_size)) {
    mbi[i] = dist(rng);
  }

  fmt::print(fmt::fg(fmt::terminal_color::blue), "{}\n", test::type_name<Dst>());
  std::uniform_int_distribution<std::ptrdiff_t> idist{0, static_cast<std::ptrdiff_t>(mbi_size) - 1};

  for (std::size_t r = 0; r < repetitions; ++r) {
    const auto it = std::as_const(mbi).begin() + idist(rng);
    const auto a = grex::load_multibyte(it, grex::scalar_tag);
    const auto b = *it;
    const auto c = it[0];
    test::check("load_multibyte", a, b, {.verbose = false});
    test::check("load_multibyte", a, c, {.verbose = false});
  }
}
} // namespace

int main() {
  pcg_extras::seed_seq_from<std::random_device> seed_source{};
  test::Rng rng{seed_source};
  grex::static_apply<8>([&]<std::size_t... I> {
#if !GREX_BACKEND_SCALAR
    (..., run_simd(rng, grex::index_tag<I + 1>));
#endif
    (..., run_scalar(rng, grex::index_tag<I + 1>));
  });
}
