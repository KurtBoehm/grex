// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_SQRT_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_SQRT_HPP

#include <immintrin.h>

#include "grex/backend/defs.hpp"
#include "grex/backend/x86/helpers.hpp"
#include "grex/backend/x86/operations/expand-scalar.hpp"
#include "grex/backend/x86/types.hpp"
#include "grex/base/defs.hpp"

namespace grex::backend {
#define GREX_SQRT_INSTRINSIC(KIND, BITS, BITPREFIX) \
  {.r = GREX_CAT(BITPREFIX##_sqrt_, GREX_EPI_SUFFIX(KIND, BITS))(v.r)}

#define GREX_SQRT(KIND, BITS, SIZE, BITPREFIX) \
  inline Vector<KIND##BITS, SIZE> sqrt(Vector<KIND##BITS, SIZE> v) { \
    return GREX_SQRT_INSTRINSIC(KIND, BITS, BITPREFIX); \
  }
#define GREX_SQRT_ALL(REGISTERBITS, BITPREFIX) \
  GREX_FOREACH_FP_TYPE(GREX_SQRT, REGISTERBITS, BITPREFIX)

GREX_FOREACH_X86_64_LEVEL(GREX_SQRT_ALL)
GREX_SUBVECTOR_UNARY(sqrt)
GREX_SUPERVECTOR_UNARY(sqrt)

// scalar implementations
inline Scalar<f32> sqrt(Scalar<f32> v) {
  return {.value = _mm_cvtss_f32(_mm_sqrt_ss(expand_any(v, index_tag<4>).r))};
}
inline Scalar<f64> sqrt(Scalar<f64> v) {
  const __m128d vec = expand_any(v, index_tag<2>).r;
  return {.value = _mm_cvtsd_f64(_mm_sqrt_sd(vec, vec))};
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_SQRT_HPP
