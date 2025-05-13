// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "thesauros/format.hpp"

#include "grex/grex.hpp"

int main() {
#if GREX_X86_64_LEVEL >= 3
  grex::Vector<grex::i64, 4> i1{4, 3, 2, -1};
  grex::Vector<grex::i64, 4> i2{-2, 3, 4, 5};
  grex::Mask<grex::i64, 4> m{true, true, false, true};
  grex::Vector<grex::f64, 4> f1{4.0, 3.0, 2.0, -1.0};
  grex::Vector<grex::f64, 4> f2{-2.0, 3.0, 4.0, 5.0};
  grex::Vector<grex::f64, 4> f3{1.0, -1.0, 1.0, -1.0};
#else
  grex::Vector<grex::i64, 2> i1{4, 3};
  grex::Vector<grex::i64, 2> i2{-2, 3};
  grex::Mask<grex::i64, 2> m{true, false};
#endif

  fmt::print("-{} = {}, -{} = {}\n", i1, -i1, f1, -f1);
  fmt::print("{} + {} = {}\n", i1, i2, i1 + i2);
  fmt::print("{} - {} = {}\n", i1, i2, i1 - i2);
  fmt::print("{} * {} = {}\n", i1, i2, i1 * i2);
  fmt::print("{} / {} = {}\n", f1, f2, f1 / f2);
  fmt::print("~{} = {}\n", i1, ~i1);

  fmt::print("fmadd({}, {}, {}) = {}\n", f1, f2, f3, grex::fmadd(f1, f2, f3));
  fmt::print("fmsub({}, {}, {}) = {}\n", f1, f2, f3, grex::fmsub(f1, f2, f3));
  fmt::print("fnmadd({}, {}, {}) = {}\n", f1, f2, f3, grex::fnmadd(f1, f2, f3));
  fmt::print("fnmsub({}, {}, {}) = {}\n", f1, f2, f3, grex::fnmsub(f1, f2, f3));

  fmt::print("abs({}) = {}\n", i2, grex::abs(i2));
  fmt::print("~{} = {}, !{} = {}\n", m, ~m, m, !m);
  fmt::print("min({}, {}) = {}\n", i1, i2, grex::min(i1, i2));
  fmt::print("max({}, {}) = {}\n", i1, i2, grex::max(i1, i2));
  fmt::print("blend_zero: {}\n", grex::blend_zero(m, i1));
  fmt::print("blend: {}\n", grex::blend(m, i1, i2));
  fmt::print("{} == {}: {}\n", i1, i2, i1 == i2);
  fmt::print("{} != {}: {}\n", i1, i2, i1 != i2);
  fmt::print("{} < {}: {}\n", i2, i1, i2 < i1);
  fmt::print("{} > {}: {}\n", i2, i1, i2 > i1);
  fmt::print("{} >= {}: {}\n", i2, i1, i2 >= i1);
  fmt::print("{} <= {}: {}\n", i2, i1, i2 <= i1);
}
