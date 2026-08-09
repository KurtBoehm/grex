// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_HORIZONTAL_ADD_HPP
#define INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_HORIZONTAL_ADD_HPP

#include "grex/backend/active/operations/arithmetic.hpp"
#include "grex/backend/base.hpp"
#include "grex/backend/defs.hpp"

#if !GREX_F16_NATIVE_ARITHMETIC
#include "grex/backend/active/operations/f16.hpp"
#include "grex/f16.hpp"
#endif

namespace grex::backend {
// Super-native: Compute the horizontal sum of the sum of the two halves
template<AnyVector THalf>
inline THalf::Value horizontal_add(SuperVector<THalf> v) {
  return horizontal_add(add(v.lower, v.upper));
}

#if !GREX_F16_NATIVE_ARITHMETIC
// Binary16 without hardware support: round-trip through binary32.
template<Float16Vector TVec>
requires(!AnySuperNativeVector<TVec>)
inline f16 horizontal_add(TVec v) {
  return grex::f32_to_f16(horizontal_add(f16_to_f32(v)));
}
#endif
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_HORIZONTAL_ADD_HPP
