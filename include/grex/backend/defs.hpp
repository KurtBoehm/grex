// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_DEFS_HPP
#define INCLUDE_GREX_BACKEND_DEFS_HPP

#include <cstddef>

#include "grex/base/defs.hpp"

namespace grex::backend {
template<Vectorizable T, std::size_t tSize>
struct Vector;
template<Vectorizable T, std::size_t tSize>
struct Mask;
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_DEFS_HPP
