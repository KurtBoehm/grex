// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_SCALAR_SIZES_HPP
#define INCLUDE_GREX_BACKEND_SCALAR_SIZES_HPP

#include <cstddef>

#include "grex/base.hpp"

namespace grex::backend {
template<Vectorizable T>
inline constexpr std::size_t min_native_size = 1;
template<Vectorizable T>
inline constexpr std::size_t max_native_size = 1;
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_SCALAR_SIZES_HPP
