// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_SHINGLE_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_SHINGLE_HPP

#include <cstddef>

#include "grex/backend/defs.hpp"

namespace grex::backend {
// super-native vectors: carry over the last element from the lower part to the upper part
template<typename THalf>
inline SuperVector<THalf> shingle_up(SuperVector<THalf> v) {
  return {
    .lower = shingle_up(v.lower),
    .upper = shingle_up(Scalar{extract(v.lower, THalf::size - 1)}, v.upper),
  };
}
template<typename THalf>
inline SuperVector<THalf> shingle_up(Scalar<typename THalf::Value> front, SuperVector<THalf> v) {
  return {
    .lower = shingle_up(front, v.lower),
    .upper = shingle_up(Scalar{extract(v.lower, THalf::size - 1)}, v.upper),
  };
}
template<typename THalf>
inline SuperVector<THalf> shingle_down(SuperVector<THalf> v) {
  return {
    .lower = shingle_down(v.lower, Scalar{extract(v.upper, 0)}),
    .upper = shingle_down(v.upper),
  };
}
template<typename THalf>
inline SuperVector<THalf> shingle_down(SuperVector<THalf> v, Scalar<typename THalf::Value> back) {
  return {
    .lower = shingle_down(v.lower, Scalar{extract(v.upper, 0)}),
    .upper = shingle_down(v.upper, back),
  };
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_SHINGLE_HPP
