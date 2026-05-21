// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_HORIZONTAL_AND_HPP
#define INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_HORIZONTAL_AND_HPP

#include "grex/backend/base.hpp"

namespace grex::backend {
template<AnyMask THalf>
inline bool horizontal_and(SuperMask<THalf> m) {
  return horizontal_and(m.lower) && horizontal_and(m.upper);
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_HORIZONTAL_AND_HPP
