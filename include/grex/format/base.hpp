// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_FORMAT_BASE_HPP
#define INCLUDE_GREX_FORMAT_BASE_HPP

#include "grex/backend/defs.hpp" // IWYU pragma: keep
#include "grex/base.hpp"

#if !GREX_BACKEND_SCALAR
#include <array>
#include <concepts>
#include <cstddef>
#include <type_traits>

#include "grex/types.hpp"

namespace grex::format_impl {
/**
 * The type that the lanes of a vector are formatted as.
 *
 * Binary16 lanes are formatted as binary32, which is exact: A native `_Float16` is not necessarily
 * formattable, since it is not one of the standard floating-point types, and `std::formatter`
 * must not be specialized for it, as it is not a program-defined type.
 */
template<Vectorizable T>
using Formatted = std::conditional_t<std::same_as<T, f16>, f32, T>;

/** The lanes of `v` as an array of formattable values. */
template<Vectorizable T, std::size_t N>
inline std::array<Formatted<T>, N> formatted_array(Vector<T, N> v) {
  if constexpr (std::same_as<T, f16>) {
    return v.convert(type_tag<f32>).as_array();
  } else {
    return v.as_array();
  }
}
} // namespace grex::format_impl
#endif

#endif // INCLUDE_GREX_FORMAT_BASE_HPP
