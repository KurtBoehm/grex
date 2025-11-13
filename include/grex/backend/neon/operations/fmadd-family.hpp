// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_NEON_OPERATIONS_FMADD_FAMILY_HPP
#define INCLUDE_GREX_BACKEND_NEON_OPERATIONS_FMADD_FAMILY_HPP

#include <cmath>

#include <arm_neon.h>

#include "grex/backend/neon/macros/types.hpp"
#include "grex/backend/neon/operations/arithmetic.hpp"
#include "grex/backend/neon/types.hpp"
#include "grex/base/defs.hpp"

namespace grex::backend {
inline constexpr bool has_fma = true;

#define GREX_FMADDF(KIND, BITS, SIZE) \
  inline Vector<KIND##BITS, SIZE> fmadd(Vector<KIND##BITS, SIZE> a, Vector<KIND##BITS, SIZE> b, \
                                        Vector<KIND##BITS, SIZE> c) { \
    return {.r = GREX_ISUFFIXED(vfmaq, KIND, BITS)(c.r, a.r, b.r)}; \
  } \
  inline Vector<KIND##BITS, SIZE> fnmadd(Vector<KIND##BITS, SIZE> a, Vector<KIND##BITS, SIZE> b, \
                                         Vector<KIND##BITS, SIZE> c) { \
    return {.r = GREX_ISUFFIXED(vfmsq, KIND, BITS)(c.r, a.r, b.r)}; \
  }

GREX_FOREACH_FP_TYPE(GREX_FMADDF, 128)

template<FloatVectorizable T, std::size_t tSize>
inline Vector<T, tSize> fmsub(Vector<T, tSize> a, Vector<T, tSize> b, Vector<T, tSize> c) {
  return fmadd(a, b, negate(c));
}
template<FloatVectorizable T, std::size_t tSize>
inline Vector<T, tSize> fnmsub(Vector<T, tSize> a, Vector<T, tSize> b, Vector<T, tSize> c) {
  return negate(fmadd(a, b, c));
}

GREX_SUBVECTOR_TERNARY(fmadd)
GREX_SUBVECTOR_TERNARY(fmsub)
GREX_SUBVECTOR_TERNARY(fnmadd)
GREX_SUBVECTOR_TERNARY(fnmsub)

GREX_SUPERVECTOR_TERNARY(fmadd)
GREX_SUPERVECTOR_TERNARY(fmsub)
GREX_SUPERVECTOR_TERNARY(fnmadd)
GREX_SUPERVECTOR_TERNARY(fnmsub)

template<typename T>
inline Scalar<T> fmadd(Scalar<T> a, Scalar<T> b, Scalar<T> c) {
  return {.value = std::fma(a.value, b.value, c.value)};
}
template<typename T>
inline Scalar<T> fmsub(Scalar<T> a, Scalar<T> b, Scalar<T> c) {
  return {.value = std::fma(a.value, b.value, -c.value)};
}
template<typename T>
inline Scalar<T> fnmadd(Scalar<T> a, Scalar<T> b, Scalar<T> c) {
  return {.value = std::fma(-a.value, b.value, c.value)};
}
template<typename T>
inline Scalar<T> fnmsub(Scalar<T> a, Scalar<T> b, Scalar<T> c) {
  return {.value = std::fma(-a.value, b.value, -c.value)};
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_FMADD_FAMILY_HPP
