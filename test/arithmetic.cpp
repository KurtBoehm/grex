// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "thesauros/format.hpp"

#include "grex/grex.hpp"

int main() {
#if GREX_X86_64_LEVEL >= 3
  grex::Vector<grex::i64, 4> v1{4, 3, 2, 1};
  grex::Vector<grex::i64, 4> v2{2, 3, 4, 5};
  grex::Mask<grex::i64, 4> m{true, true, false, true};
#else
  grex::Vector<grex::i64, 2> v1{4, 3};
  grex::Vector<grex::i64, 2> v2{2, 3};
  grex::Mask<grex::i64, 2> m{true, false};
#endif

  fmt::print("{} + {} = {}, {} - {} = {}\n", v1, v2, v1 + v2, v1, v2, v1 - v2);
  fmt::print("~{} = {}\n", v1, ~v1);

  fmt::print("{}, neg: {} {}\n", m, ~m, !m);
  fmt::print("min({}, {}) = {}\n", v1, v2, grex::min(v1, v2));
  fmt::print("max({}, {}) = {}\n", v1, v2, grex::max(v1, v2));
  fmt::print("blend_zero: {}\n", grex::blend_zero(m, v1));
  fmt::print("blend: {}\n", grex::blend(m, v1, v2));
  fmt::print("{} == {}: {}\n", v1, v2, v1 == v2);
  fmt::print("{} != {}: {}\n", v1, v2, v1 != v2);
  fmt::print("{} < {}: {}\n", v2, v1, v2 < v1);
  fmt::print("{} > {}: {}\n", v2, v1, v2 > v1);
  fmt::print("{} >= {}: {}\n", v2, v1, v2 >= v1);
  fmt::print("{} <= {}: {}\n", v2, v1, v2 <= v1);
}
