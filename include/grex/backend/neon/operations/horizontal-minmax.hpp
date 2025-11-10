// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_NEON_OPERATIONS_HORIZONTAL_MINMAX_HPP
#define INCLUDE_GREX_BACKEND_NEON_OPERATIONS_HORIZONTAL_MINMAX_HPP

#include <algorithm>

#include <arm_neon.h>

#include "grex/backend/macros/base.hpp"
#include "grex/backend/neon/macros/types.hpp"
#include "grex/backend/neon/types.hpp"

namespace grex::backend {
#define GREX_HMINMAX_IMPL_BASE(NAME, KIND, BITS) \
  return GREX_CAT(v##NAME##vq_, GREX_ISUFFIX(KIND, BITS))(v.r);
#define GREX_HMINMAX_IMPL_INT8 GREX_HMINMAX_IMPL_BASE
#define GREX_HMINMAX_IMPL_INT16 GREX_HMINMAX_IMPL_BASE
#define GREX_HMINMAX_IMPL_INT32 GREX_HMINMAX_IMPL_BASE
#define GREX_HMINMAX_IMPL_INT64(NAME, KIND, BITS) \
  const auto a = GREX_CAT(vgetq_lane_, GREX_ISUFFIX(KIND, BITS))(v.r, 0); \
  const auto b = GREX_CAT(vgetq_lane_, GREX_ISUFFIX(KIND, BITS))(v.r, 1); \
  return std::NAME(a, b);

#define GREX_HMINMAX_IMPL_f(NAME, KIND, BITS) return v##NAME##nmvq_f##BITS(v.r);
#define GREX_HMINMAX_IMPL_i(NAME, KIND, BITS) GREX_HMINMAX_IMPL_INT##BITS(NAME, KIND, BITS)
#define GREX_HMINMAX_IMPL_u(NAME, KIND, BITS) GREX_HMINMAX_IMPL_INT##BITS(NAME, KIND, BITS)

#define GREX_HMINMAX(KIND, BITS, SIZE) \
  inline KIND##BITS horizontal_min(Vector<KIND##BITS, SIZE> v) { \
    GREX_HMINMAX_IMPL_##KIND(min, KIND, BITS) \
  } \
  inline KIND##BITS horizontal_max(Vector<KIND##BITS, SIZE> v) { \
    GREX_HMINMAX_IMPL_##KIND(max, KIND, BITS) \
  }

GREX_FOREACH_TYPE(GREX_HMINMAX, 128)
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_HORIZONTAL_MINMAX_HPP
