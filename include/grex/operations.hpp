// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_OPERATIONS_HPP
#define INCLUDE_GREX_OPERATIONS_HPP

#include "grex/backend/active/operations.hpp"
#include "grex/backend/defs.hpp"
#include "grex/base/defs.hpp"

namespace grex {
#define GREX_MATH_FMA(NAME) \
  template<FpVectorizable T> \
  inline T NAME(T a, T b, T c) { \
    return backend::NAME(backend::Scalar{a}, backend::Scalar{b}, backend::Scalar{c}).value; \
  }
GREX_MATH_FMA(fmadd)
GREX_MATH_FMA(fmsub)
GREX_MATH_FMA(fnmadd)
GREX_MATH_FMA(fnmsub)
#undef GREX_MATH_FMA

template<FpVectorizable T>
inline T sqrt(T a) {
  return backend::sqrt(backend::Scalar{a}).value;
}

template<Vectorizable T>
inline T abs(T a) {
  return backend::abs(backend::Scalar{a}).value;
}
template<Vectorizable T>
inline T min(T a, T b) {
  return backend::min(backend::Scalar{a}, backend::Scalar{b}).value;
}
template<Vectorizable T>
inline T max(T a, T b) {
  return backend::max(backend::Scalar{a}, backend::Scalar{b}).value;
}

#define GREX_MATH_MASKARITH(NAME) \
  template<Vectorizable T> \
  inline T NAME(bool mask, T a, T b) { \
    return backend::NAME(mask, backend::Scalar{a}, backend::Scalar{b}).value; \
  }
GREX_MATH_MASKARITH(mask_add)
GREX_MATH_MASKARITH(mask_subtract)
GREX_MATH_MASKARITH(mask_multiply)
GREX_MATH_MASKARITH(mask_divide)
#undef GREX_MATH_MASKARITH

template<Vectorizable T>
inline T blend_zero(bool selector, T v1) {
  return backend::blend_zero(selector, backend::Scalar{v1}).value;
}
template<Vectorizable T>
inline T blend(bool selector, T v0, T v1) {
  return backend::blend(selector, backend::Scalar{v0}, backend::Scalar{v1}).value;
}

template<Vectorizable T>
inline bool is_finite(T a) {
  return backend::is_finite(backend::Scalar{a});
}
} // namespace grex

#endif // INCLUDE_GREX_OPERATIONS_HPP
