// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_SCALAR_OPERATIONS_HPP
#define INCLUDE_GREX_BACKEND_SCALAR_OPERATIONS_HPP

#include <cmath>

#include "grex/backend/base.hpp"
#include "grex/base.hpp"

namespace grex::backend {
template<FloatVectorizable T>
inline Scalar<T> make_finite(Scalar<T> v) {
  return is_finite(v) ? v : Scalar{T{}};
}

template<FloatVectorizable T>
inline Scalar<T> sqrt(Scalar<T> v) {
  return {.value = std::sqrt(v.value)};
}

inline constexpr bool has_fma = false;

template<FloatVectorizable T>
inline Scalar<T> fmadd(Scalar<T> a, Scalar<T> b, Scalar<T> c) {
  return {.value = (a.value * b.value) + c.value};
}
template<FloatVectorizable T>
inline Scalar<T> fmsub(Scalar<T> a, Scalar<T> b, Scalar<T> c) {
  return {.value = (a.value * b.value) - c.value};
}
template<FloatVectorizable T>
inline Scalar<T> fnmadd(Scalar<T> a, Scalar<T> b, Scalar<T> c) {
  return {.value = c.value - (a.value * b.value)};
}
template<FloatVectorizable T>
inline Scalar<T> fnmsub(Scalar<T> a, Scalar<T> b, Scalar<T> c) {
  return {.value = -(a.value * b.value + c.value)};
}

template<UnsignedIntVectorizable T>
inline bool bit_test(T a, T b) {
  return ((a >> b) & 1) != 0;
}

template<IntVectorizable TDst, IntVectorizable TSrc>
inline TDst expand_any(TSrc src) {
  return TDst(src);
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_SCALAR_OPERATIONS_HPP
