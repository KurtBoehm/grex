// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.


#ifndef INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_CLASSIFICATION_HPP
#define INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_CLASSIFICATION_HPP

#include <cstddef>

#include "grex/backend/active/operations/blend.hpp"
#include "grex/backend/active/operations/expand.hpp"
#include "grex/backend/active/operations/extract-single.hpp"
#include "grex/backend/base.hpp"
#include "grex/base.hpp"

namespace grex::backend {
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

#endif // INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_CLASSIFICATION_HPP
