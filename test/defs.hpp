// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef TEST_DEFS_HPP
#define TEST_DEFS_HPP

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <functional>

#include "thesauros/format.hpp"
#include "thesauros/static-ranges.hpp"
#include "thesauros/types.hpp"

#include "grex/base/defs.hpp"
#include "grex/types.hpp"

namespace grex::test {
template<Vectorizable T, std::size_t tSize>
struct VectorChecker {
  grex::Vector<T, tSize> vec{};
  std::array<T, tSize> ref{};

  VectorChecker() = default;

  explicit VectorChecker(T value) : vec{value} {
    std::ranges::fill(ref, value);
  }
  template<typename... Ts>
  requires(sizeof...(Ts) == tSize)
  explicit VectorChecker(Ts... values) : vec{values...}, ref{values...} {}

  static VectorChecker indices() {
    return VectorChecker{
      grex::Vector<T, tSize>::indices(),
      thes::star::index_transform<tSize>([](auto i) { return T(i); }) | thes::star::to_array,
    };
  }

  void check() const {
    const bool same = thes::star::transform([](T v0, T v1) { return v0 == v1; }, vec, ref) |
                      thes::star::left_reduce(std::logical_and{});
    if (same) {
      fmt::print(thes::fg_green, "{} == {}\n", vec, ref);
    } else {
      fmt::print(thes::fg_red, "{} != {}\n", vec, ref);
      std::exit(EXIT_FAILURE);
    }
  }

private:
  VectorChecker(grex::Vector<T, tSize> v, std::array<T, tSize> a) : vec{v}, ref{a} {}
};

template<Vectorizable T, std::size_t tSize>
struct MaskChecker {
  grex::Mask<T, tSize> vec{};
  std::array<bool, tSize> ref{};

  MaskChecker() = default;

  explicit MaskChecker(bool value) : vec{value} {
    std::ranges::fill(ref, value);
  }
  template<typename... Ts>
  requires(sizeof...(Ts) == tSize)
  explicit MaskChecker(Ts... values) : vec{values...}, ref{values...} {}

  static MaskChecker ones() {
    return MaskChecker{
      grex::Mask<T, tSize>::ones(),
      thes::star::constant<tSize>(true) | thes::star::to_array,
    };
  }

  void check() const {
    const bool same = thes::star::transform([](T v0, T v1) { return v0 == v1; }, vec, ref) |
                      thes::star::left_reduce(std::logical_and{});
    if (same) {
      fmt::print(thes::fg_green, "{} == {}\n", vec, ref);
    } else {
      fmt::print(thes::fg_red, "{} != {}\n", vec, ref);
      std::exit(EXIT_FAILURE);
    }
  }

private:
  MaskChecker(grex::Mask<T, tSize> v, std::array<bool, tSize> a) : vec{v}, ref{a} {}
};

inline void run_types_sizes(auto f) {
  auto inner = [&]<typename T, std::size_t tSize>(thes::TypeTag<T> t, thes::IndexTag<tSize> s) {
    fmt::print(thes::fg_blue, "{} × {}\n", thes::type_name<T>(), tSize);
    f(t, s);
  };
  auto outer = [&]<typename T>(thes::TypeTag<T> t) {
    thes::star::iota<1, 7> |
      thes::star::for_each([&](auto i) { inner(t, thes::index_tag<1U << i>); });
  };
  outer(thes::type_tag<f32>);
  outer(thes::type_tag<f64>);
  outer(thes::type_tag<i8>);
  outer(thes::type_tag<i16>);
  outer(thes::type_tag<i32>);
  outer(thes::type_tag<i64>);
  outer(thes::type_tag<u8>);
  outer(thes::type_tag<u16>);
  outer(thes::type_tag<u32>);
  outer(thes::type_tag<u64>);
}
} // namespace grex::test

#endif // TEST_DEFS_HPP
