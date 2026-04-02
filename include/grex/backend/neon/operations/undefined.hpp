// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_NEON_OPERATIONS_UNDEFINED_HPP
#define INCLUDE_GREX_BACKEND_NEON_OPERATIONS_UNDEFINED_HPP

#include <type_traits>

namespace grex::backend {
template<typename T>
requires(std::is_trivial_v<T>)
inline T make_undefined() {
  T undefined;
  // This prevents warnings about undefined values
  asm("" : "=w"(undefined)); // NOLINT
  return undefined;
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_UNDEFINED_HPP
