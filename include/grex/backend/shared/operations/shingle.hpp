// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_SHINGLE_HPP
#define INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_SHINGLE_HPP

#include "grex/backend/active/operations/extract-single.hpp"
#include "grex/backend/active/operations/extract.hpp"
#include "grex/backend/base.hpp"
#include "grex/base.hpp"

namespace grex::backend {
// Super-native vectors: carry over the last element of the lower part to the upper part.
template<typename Half>
inline SuperVector<Half> shingle_up(SuperVector<Half> v) {
  return {
    .lower = shingle_up(v.lower),
    .upper = shingle_up(extract(v.lower, index_tag<size_of<Half> - 1>), v.upper),
  };
}
template<typename Half>
inline SuperVector<Half> shingle_up(ValueOf<Half> front, SuperVector<Half> v) {
  return {
    .lower = shingle_up(front, v.lower),
    .upper = shingle_up(extract(v.lower, index_tag<size_of<Half> - 1>), v.upper),
  };
}

// Super-native vectors: carry over the first element of the upper part to the lower part.
template<typename Half>
inline SuperVector<Half> shingle_down(SuperVector<Half> v) {
  return {
    .lower = shingle_down(v.lower, extract_single(v.upper)),
    .upper = shingle_down(v.upper),
  };
}
template<typename Half>
inline SuperVector<Half> shingle_down(SuperVector<Half> v, ValueOf<Half> back) {
  return {
    .lower = shingle_down(v.lower, extract_single(v.upper)),
    .upper = shingle_down(v.upper, back),
  };
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_SHINGLE_HPP
