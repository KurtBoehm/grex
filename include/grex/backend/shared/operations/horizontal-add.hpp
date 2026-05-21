// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_HORIZONTAL_ADD_HPP
#define INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_HORIZONTAL_ADD_HPP

#include "grex/backend/active/operations/arithmetic.hpp"
#include "grex/backend/base.hpp"

namespace grex::backend {
// Super-native: Compute the horizontal sum of the sum of the two halves
template<AnyVector THalf>
inline THalf::Value horizontal_add(SuperVector<THalf> v) {
  return horizontal_add(add(v.lower, v.upper));
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_HORIZONTAL_ADD_HPP
