// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <array>
#include <limits>

#include "thesauros/format.hpp"
#include "thesauros/types.hpp"

#include "grex/grex.hpp"

#define CONDITION (GREX_X86_64_LEVEL >= 3)

int main() {
  using namespace grex::literals;

#if CONDITION
  using IVec = grex::Vector<grex::i64, 4>;
  using IMask = grex::Mask<grex::i64, 4>;
  const IVec i1{4, 3, 2, -1};
  const IVec i2{-2, 3, 4, 5};
  const IMask m1{true, true, false, true};
  const IMask m2{true, false, false, true};
  std::array<grex::f64, 4> f1d{4.0, 3.0, 2.0, -1.0};
  const auto f1 = grex::Vector<grex::f64, 4>::load(f1d.data());
  const grex::Vector<grex::f64, 4> f2{-2.0, 3.0, 4.0, 5.0};
  const grex::Vector<grex::f64, 4> f3{1.0, -1.0, 1.0, -1.0};
  const grex::Vector<grex::f64, 4> f4{1.0, std::numeric_limits<grex::f64>::infinity(),
                                      std::numeric_limits<grex::f64>::quiet_NaN(), -1.0};
  const grex::Mask<grex::f64, 4> fm1{true, true, false, true};
  const grex::Mask<grex::f64, 4> fm2{true, false, false, true};
#else
  using IVec = grex::Vector<grex::i64, 2>;
  using IMask = grex::Mask<grex::i64, 2>;
  IVec i1{4, 3};
  IVec i2{-2, 3};
  IMask m1{true, true};
  IMask m2{true, false};
  grex::Vector<grex::f64, 2> f1{4.0, 3.0};
  grex::Vector<grex::f64, 2> f2{-2.0, 3.0};
  grex::Vector<grex::f64, 2> f3{1.0, -1.0};
  grex::Vector<grex::f64, 2> f4{1.0, std::numeric_limits<grex::f64>::infinity()};
  grex::Mask<grex::f64, 2> fm1{true, false};
#endif

  fmt::print("indices: {} {}\n", IVec::indices(), IVec::indices(5));
  fmt::print("-{} = {}, -{} = {}\n", i1, -i1, f1, -f1);
  fmt::print("{} + {} = {}\n", i1, i2, i1 + i2);
  fmt::print("{} - {} = {}\n", i1, i2, i1 - i2);
  fmt::print("{} * {} = {}\n", i1, i2, i1 * i2);
  fmt::print("{} / {} = {}\n", f1, f2, f1 / f2);
  fmt::print("~{} = {}\n", i1, ~i1);
  fmt::print("{} & {} = {}\n", i1, i2, i1 & i2);
  fmt::print("{} | {} = {}\n", i1, i2, i1 | i2);
  fmt::print("{} ^ {} = {}\n", i1, i2, i1 ^ i2);
  fmt::print("{}.insert(1, 2) = {}\n", i1, i1.insert(1, 2));

  fmt::print("fmadd({}, {}, {}) = {}\n", f1, f2, f3, grex::fmadd(f1, f2, f3));
  fmt::print("fmsub({}, {}, {}) = {}\n", f1, f2, f3, grex::fmsub(f1, f2, f3));
  fmt::print("fnmadd({}, {}, {}) = {}\n", f1, f2, f3, grex::fnmadd(f1, f2, f3));
  fmt::print("fnmsub({}, {}, {}) = {}\n", f1, f2, f3, grex::fnmsub(f1, f2, f3));
  fmt::print("abs({}) = {}\n", i2, grex::abs(i2));
  fmt::print("horizontal_sum({}) = {}\n", i2, grex::horizontal_add(i2));
  fmt::print("horizontal_sum({}) = {}\n", f1, grex::horizontal_add(f1));
  fmt::print("horizontal_min({}) = {}\n", i2, grex::horizontal_min(i2));
  fmt::print("horizontal_min({}) = {}\n", f1, grex::horizontal_min(f1));
  fmt::print("horizontal_max({}) = {}\n", i2, grex::horizontal_max(i2));
  fmt::print("horizontal_max({}) = {}\n", f1, grex::horizontal_max(f1));
  fmt::print("mask_add({}, {}, {}) = {}\n", m1, i1, i2, grex::mask_add(m1, i1, i2));
  fmt::print("mask_subtract({}, {}, {}) = {}\n", m1, i1, i2, grex::mask_subtract(m1, i1, i2));
  fmt::print("mask_multiply({}, {}, {}) = {}\n", m1, i1, i2, grex::mask_multiply(m1, i1, i2));
  fmt::print("mask_divide({}, {}, {}) = {}\n", m1, f1, f2, grex::mask_divide(fm1, f1, f2));
  fmt::print("min({}, {}) = {}\n", i1, i2, grex::min(i1, i2));
  fmt::print("max({}, {}) = {}\n", i1, i2, grex::max(i1, i2));
  fmt::print("blend_zero: {}\n", grex::blend_zero(m1, i1));
  fmt::print("blend: {}\n", grex::blend(m1, i1, i2));

  fmt::print("cutoff({}, {}) = {}\n", i1, IVec::size / 2, i1.cutoff(IVec::size / 2));
  fmt::print("!{} = {}\n", m1, !m1);
  fmt::print("{} == {}: {}\n", i1, i2, i1 == i2);
  fmt::print("{} != {}: {}\n", i1, i2, i1 != i2);
  fmt::print("{} < {}: {}\n", i2, i1, i2 < i1);
  fmt::print("{} > {}: {}\n", i2, i1, i2 > i1);
  fmt::print("{} >= {}: {}\n", i2, i1, i2 >= i1);
  fmt::print("{} <= {}: {}\n", i2, i1, i2 <= i1);

  fmt::print("cutoff_mask({}) = {}\n", IMask::size / 2, IMask::cutoff_mask(IMask::size / 2));
  fmt::print("{} == {}: {}\n", m1, m2, m1 == m2);
  fmt::print("{} != {}: {}\n", m1, m2, m1 != m2);
  fmt::print("{} && {}: {}\n", m1, m2, m1 && m2);
  fmt::print("{} || {}: {}\n", m1, m2, m1 || m2);
  fmt::print("horizontal_and({}) = {}\n", m1, grex::horizontal_and(m1));
  fmt::print("horizontal_and({}) = {}\n", m1 || !m2, grex::horizontal_and(m1 || !m2));
  fmt::print("is_finite({}) = {}\n", f4, grex::is_finite(f4));

#if !CONDITION
  fmt::print("shuffle({}, [1, 0]) = {}", i1,
             grex::shuffle(i1, thes::auto_tag<std::array{1_sh, 0_sh}>));
#endif
}
