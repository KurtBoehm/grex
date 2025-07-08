#include <array>
#include <cstddef>

#include <fmt/base.h>
#include <fmt/format.h>

#include "grex/grex.hpp"

#include "defs.hpp"

namespace test = grex::test;

int main() {
  using enum grex::IterDirection;

  // transform
  {
    // scalar
    test::check("transform scalar",
                grex::transform([](auto i) { return int{i}; }, grex::scalar_tag), 0);
    // vectorized
    auto op = [](grex::AnyIndexTag auto size) {
      fmt::print("{}\n", size.value);
      // full
      {
        test::VectorChecker<int, size> checker{
          grex::transform([](auto i) { return int{i}; }, grex::full_tag<size>),
          grex::static_apply<size>(
            []<std::size_t... tIdxs>() { return std::array{int{tIdxs}...}; }),
        };
        checker.check("transform full");
      }
      // part
      for (std::size_t i = 0; i <= size; ++i) {
        test::VectorChecker<int, size> checker{
          grex::transform([](auto i) { return int{i + 1}; }, grex::part_tag<size>(i)),
          grex::static_apply<size>([&]<std::size_t... tIdxs>() {
            return std::array{((tIdxs < i) ? int{tIdxs + 1} : 0)...};
          }),
        };
        checker.check("transform part", false);
      }
    };
    grex::static_apply<1, 8>(
      [&]<std::size_t... tIdxs>() { (..., op(grex::index_tag<std::size_t{1} << tIdxs>)); });
  }
  // for_each
  {
    // scalar
    {
      auto op = [](grex::AnyValueTag<grex::IterDirection> auto dir) {
        std::array<int, 1> dst{0};
        grex::for_each([&](auto i) { dst.at(i) = int(i + 1); }, dir,
                       grex::typed_scalar_tag<std::size_t>);
        test::check(fmt::format("for_each scalar {}", dir.value), dst, std::array{1});
      };
      op(grex::auto_tag<forward>);
      op(grex::auto_tag<backward>);
    } // vectorized
    auto opo = [](grex::AnyIndexTag auto size, grex::AnyValueTag<grex::IterDirection> auto dir) {
      // full
      {
        std::array<int, size> dst{};
        std::size_t i = 0;
        grex::for_each([&](auto j) { dst.at(i++) = int(j + 1); }, dir,
                       grex::typed_full_tag<std::size_t, size>);
        const auto ref = grex::static_apply<size>([&]<std::size_t... tIdxs>() {
          return std::array{int{(dir == forward) ? (tIdxs + 1) : (size - tIdxs)}...};
        });
        test::check(fmt::format("for_each full {}", dir.value), dst, ref);
      }
      // part
      for (std::size_t part = 0; part <= size; ++part) {
        std::array<int, size> dst{};
        std::size_t i = 0;
        grex::for_each([&](auto j) { dst.at(i++) = int(j + 1); }, dir, grex::part_tag<size>(part));
        const auto ref = grex::static_apply<size>([&]<std::size_t... tIdxs>() {
          return std::array{
            ((tIdxs < part) ? int((dir == forward) ? (tIdxs + 1) : (part - tIdxs)) : 0)...};
        });
        test::check(fmt::format("for_each part {}", dir.value), dst, ref, false);
      }
    };
    auto op = [&](grex::AnyIndexTag auto size) {
      fmt::print("{}\n", size.value);
      opo(size, grex::auto_tag<forward>);
      opo(size, grex::auto_tag<backward>);
    };
    grex::static_apply<1, 8>(
      [&]<std::size_t... tIdxs>() { (..., op(grex::index_tag<std::size_t{1} << tIdxs>)); });
  }
}
