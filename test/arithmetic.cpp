// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <array>

#include "thesauros/format.hpp"
#include "thesauros/types.hpp"

#include "grex/grex.hpp"

#define CONDITION (GREX_X86_64_LEVEL >= 3)

int main() {
  using namespace grex::literals;

#if CONDITION
  using IVec = grex::Vector<grex::i64, 4>;
  using IMask = grex::Mask<grex::i64, 4>;
  IVec i1{4, 3, 2, -1};
  IVec i2{-2, 3, 4, 5};
  IMask m1{true, true, false, true};
  IMask m2{true, false, false, true};
  grex::Vector<grex::f64, 4> f1{4.0, 3.0, 2.0, -1.0};
  grex::Vector<grex::f64, 4> f2{-2.0, 3.0, 4.0, 5.0};
  grex::Vector<grex::f64, 4> f3{1.0, -1.0, 1.0, -1.0};
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

#if !CONDITION
  fmt::print("shuffle({}, [1, 0]) = {}", i1,
             grex::shuffle(i1, thes::auto_tag<std::array{1_sh, 0_sh}>));
#endif
}
