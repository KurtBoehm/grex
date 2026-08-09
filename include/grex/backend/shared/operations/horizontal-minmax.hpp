// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_HORIZONTAL_MINMAX_HPP
#define INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_HORIZONTAL_MINMAX_HPP

#include "grex/backend/active/operations/minmax.hpp"
#include "grex/backend/base.hpp"
#include "grex/backend/defs.hpp"

#if !GREX_F16_NATIVE_ARITHMETIC
#include "grex/backend/active/operations/f16.hpp"
#include "grex/f16.hpp"
#endif

namespace grex::backend {
template<typename THalf>
inline THalf::Value horizontal_min(SuperVector<THalf> v) {
  return horizontal_min(min(v.lower, v.upper));
}
template<typename THalf>
inline THalf::Value horizontal_max(SuperVector<THalf> v) {
  return horizontal_max(max(v.lower, v.upper));
}

#if !GREX_F16_NATIVE_ARITHMETIC
// Binary16 without hardware support: round-trip through binary32.
template<Float16Vector TVec>
requires(!AnySuperNativeVector<TVec>)
inline f16 horizontal_min(TVec v) {
  return grex::f32_to_f16(horizontal_min(f16_to_f32(v)));
}
template<Float16Vector TVec>
requires(!AnySuperNativeVector<TVec>)
inline f16 horizontal_max(TVec v) {
  return grex::f32_to_f16(horizontal_max(f16_to_f32(v)));
}
#endif
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_HORIZONTAL_MINMAX_HPP
