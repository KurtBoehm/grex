#include <array>
#include <cstddef>

#include <fmt/base.h>

#include "grex/grex.hpp"

#include "defs.hpp"

namespace test = grex::test;

int main() {
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
      std::array<int, 1> dst{0};
      grex::for_each([&](auto i) { dst.at(i) = int(i + 1); }, grex::typed_scalar_tag<std::size_t>);
      test::check("for_each scalar", dst, std::array{1});
    }
    // vectorized
    auto opo = [](grex::AnyIndexTag auto size, grex::AnyValueTag<grex::IterDirection> auto dir) {
      fmt::print("{}\n", size.value);
      // full
      {
        std::array<int, size> dst{};
        grex::for_each([&](auto i) { dst.at(i) = int(i + 1); }, dir,
                       grex::typed_full_tag<std::size_t, size>);
        const auto ref = grex::static_apply<size>(
          []<std::size_t... tIdxs>() { return std::array{int{tIdxs + 1}...}; });
        test::check("for_each full", dst, ref);
      }
      // part
      for (std::size_t i = 0; i <= size; ++i) {
        std::array<int, size> dst{};
        grex::for_each([&](auto i) { dst.at(i) = int(i + 1); }, dir, grex::part_tag<size>(i));
        const auto ref = grex::static_apply<size>([&]<std::size_t... tIdxs>() {
          return std::array{((tIdxs < i) ? int{tIdxs + 1} : 0)...};
        });
        test::check("for_each part", dst, ref, false);
      }
    };
    auto op = [&](grex::AnyIndexTag auto size) {
      opo(size, grex::auto_tag<grex::IterDirection::FORWARD>);
      opo(size, grex::auto_tag<grex::IterDirection::BACKWARD>);
    };
    grex::static_apply<1, 8>(
      [&]<std::size_t... tIdxs>() { (..., op(grex::index_tag<std::size_t{1} << tIdxs>)); });
  }
}
