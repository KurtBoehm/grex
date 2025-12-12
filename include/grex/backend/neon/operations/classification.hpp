// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_NEON_OPERATIONS_CLASSIFICATION_HPP
#define INCLUDE_GREX_BACKEND_NEON_OPERATIONS_CLASSIFICATION_HPP

#include <cstddef>

#include <arm_neon.h>

#include "grex/backend/defs.hpp" // IWYU pragma: keep
#include "grex/backend/macros/for-each.hpp"
#include "grex/backend/neon/operations/blend.hpp"
#include "grex/backend/neon/operations/expand.hpp"
#include "grex/backend/neon/operations/extract-single.hpp"
#include "grex/backend/neon/types.hpp"
#include "grex/base.hpp"

namespace grex::backend {
#define GREX_ISFIN(KIND, BITS, SIZE, ...) \
  inline Mask<KIND##BITS, SIZE> is_finite(Vector<KIND##BITS, SIZE> v) { \
    /* the largest finite value */ \
    const auto maxvec = vdupq_n_f##BITS(std::numeric_limits<f##BITS>::max()); \
    /* compare the absolute value with the largest finite value */ \
    return {.r = vcleq_f##BITS(vabsq_f##BITS(v.r), maxvec)}; \
  }
GREX_FOREACH_FP_TYPE(GREX_ISFIN, 128)

template<FloatVectorizable T, std::size_t tPart, std::size_t tSize>
inline SubMask<T, tPart, tSize> is_finite(SubVector<T, tPart, tSize> v) {
  return SubMask<T, tPart, tSize>{is_finite(v.full)};
}
template<typename THalf>
inline auto is_finite(SuperVector<THalf> v) {
  return SuperMask{.lower = is_finite(v.lower), .upper = is_finite(v.upper)};
}

template<AnyVector TVec>
inline TVec make_finite(TVec v) {
  return blend_zero(is_finite(v), v);
}
template<FloatVectorizable T>
inline Scalar<T> make_finite(Scalar<T> v) {
  const auto vec = expand_any(v, index_tag<16 / sizeof(T)>);
  return extract_single(blend_zero(is_finite(vec), vec));
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_CLASSIFICATION_HPP
