// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_HORIZONTAL_MINMAX_HPP
#define INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_HORIZONTAL_MINMAX_HPP

#include "grex/backend/active/operations/minmax.hpp"
#include "grex/backend/base.hpp"

namespace grex::backend {
template<typename THalf>
inline THalf::Value horizontal_min(SuperVector<THalf> v) {
  return horizontal_min(min(v.lower, v.upper));
}
template<typename THalf>
inline THalf::Value horizontal_max(SuperVector<THalf> v) {
  return horizontal_max(max(v.lower, v.upper));
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_HORIZONTAL_MINMAX_HPP
