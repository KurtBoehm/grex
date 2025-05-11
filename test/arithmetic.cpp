// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "thesauros/format.hpp"

#include "grex/grex.hpp"

int main() {
  grex::Vector<grex::i64, 4> v1{4, 3, 2, 1};
  grex::Vector<grex::i64, 4> v2{6, 7, 8, 9};
  fmt::print("{} + {} = {}, {} - {} = {}\n", v1, v2, v1 + v2, v1, v2, v1 - v2);
  grex::Mask<grex::i64, 4> m{true, true, false, true};
  fmt::print("{}\n", m);
  fmt::print("blend_zero: {}\n", grex::blend_zero(m, v1));
  fmt::print("blend: {}\n", grex::blend(m, v1, v2));
}
