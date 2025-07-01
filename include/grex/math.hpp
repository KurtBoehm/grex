// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_MATH_HPP
#define INCLUDE_GREX_MATH_HPP

#include "grex/backend/active/operations.hpp"
#include "grex/backend/active/types.hpp"
#include "grex/base/defs.hpp"

namespace grex {
#define GREX_MATH_FMA(NAME) \
  template<FpVectorizable T> \
  inline T NAME(T a, T b, T c) { \
    return backend::NAME(backend::Scalar{a}, backend::Scalar{b}, backend::Scalar{c}); \
  }
GREX_MATH_FMA(fmadd)
GREX_MATH_FMA(fmsub)
GREX_MATH_FMA(fnmadd)
GREX_MATH_FMA(fnmsub)
#undef GREX_MATH_FMA

template<FpVectorizable T>
inline T sqrt(T a) {
  return backend::sqrt(backend::Scalar{a});
}
} // namespace grex

#endif // INCLUDE_GREX_MATH_HPP
