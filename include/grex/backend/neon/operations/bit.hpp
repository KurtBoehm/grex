// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_NEON_OPERATIONS_BIT_HPP
#define INCLUDE_GREX_BACKEND_NEON_OPERATIONS_BIT_HPP

#include <cstddef>

#include "grex/base.hpp"

namespace grex::backend {
template<UnsignedIntVectorizable T>
inline bool bit_test(T a, T b) {
  return ((a >> b) & 1) != 0;
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_BIT_HPP
