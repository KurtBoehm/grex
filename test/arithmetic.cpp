// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "grex/grex.hpp"

int main() {
  grex::Vector<grex::u64, 4> v{};
  v = v + v;
  v = v - v;
}
