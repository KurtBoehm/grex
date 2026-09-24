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
template<FloatVectorizable T, std::size_t N>
inline SubMask<T, N> is_finite(SubVector<T, N> v) {
  return SubMask<T, N>{is_finite(v.full)};
}
template<typename Half>
inline auto is_finite(SuperVector<Half> v) {
  return SuperMask{.lower = is_finite(v.lower), .upper = is_finite(v.upper)};
}

template<AnyVector Vec>
inline Vec make_finite(Vec v) {
  return blend_zero(is_finite(v), v);
}
template<FloatVectorizable T>
inline T make_finite(T v) {
  const auto vec = expand_any(v, index_tag<16 / sizeof(T)>);
  return extract_single(blend_zero(is_finite(vec), vec));
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_CLASSIFICATION_HPP
