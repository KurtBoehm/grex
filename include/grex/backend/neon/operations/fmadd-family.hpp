// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_NEON_OPERATIONS_FMADD_FAMILY_HPP
#define INCLUDE_GREX_BACKEND_NEON_OPERATIONS_FMADD_FAMILY_HPP

#include <cmath>
#include <cstddef>

#include <arm_neon.h>

#include "grex/backend/base.hpp"
#include "grex/backend/defs.hpp" // IWYU pragma: keep
#include "grex/backend/macros/for-each.hpp"
#include "grex/backend/macros/types.hpp"
#include "grex/backend/neon/macros/types.hpp"
#include "grex/backend/neon/operations/arithmetic.hpp"
#include "grex/backend/neon/types.hpp"
#include "grex/base.hpp"

namespace grex::backend {
inline constexpr bool has_fma = true;

#define GREX_FMADDF(KIND, BITS, SIZE) \
  inline NativeVector<KIND##BITS, SIZE> fmadd(NativeVector<KIND##BITS, SIZE> a, \
                                              NativeVector<KIND##BITS, SIZE> b, \
                                              NativeVector<KIND##BITS, SIZE> c) { \
    return {.r = GREX_ISUFFIXED(vfmaq, KIND, BITS)(c.r, a.r, b.r)}; \
  } \
  inline NativeVector<KIND##BITS, SIZE> fnmadd(NativeVector<KIND##BITS, SIZE> a, \
                                               NativeVector<KIND##BITS, SIZE> b, \
                                               NativeVector<KIND##BITS, SIZE> c) { \
    return {.r = GREX_ISUFFIXED(vfmsq, KIND, BITS)(c.r, a.r, b.r)}; \
  }

GREX_FOREACH_FP_TYPE(GREX_FMADDF, 128)

template<FloatVectorizable T, std::size_t tSize>
inline NativeVector<T, tSize> fmsub(NativeVector<T, tSize> a, NativeVector<T, tSize> b,
                                    NativeVector<T, tSize> c) {
  return fmadd(a, b, negate(c));
}
template<FloatVectorizable T, std::size_t tSize>
inline NativeVector<T, tSize> fnmsub(NativeVector<T, tSize> a, NativeVector<T, tSize> b,
                                     NativeVector<T, tSize> c) {
  return negate(fmadd(a, b, c));
}

GREX_NNVECTOR_TERNARY(fmadd)
GREX_NNVECTOR_TERNARY(fmsub)
GREX_NNVECTOR_TERNARY(fnmadd)
GREX_NNVECTOR_TERNARY(fnmsub)

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
