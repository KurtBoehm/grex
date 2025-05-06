// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_TYPES_HPP
#define INCLUDE_GREX_TYPES_HPP

#include <cstddef>

#include "grex/backend.hpp"
#include "grex/base.hpp"

namespace grex {
template<Vectorizable T, std::size_t tSize>
struct Vector {
  Vector() = default;

  friend Vector operator+(Vector a, Vector b) {
    return Vector(backend::add(a.vec_, b.vec_));
  }
  friend Vector operator-(Vector a, Vector b) {
    return Vector(backend::subtract(a.vec_, b.vec_));
  }

private:
  using BackendVec = backend::Vector<T, tSize>;
  explicit Vector(BackendVec v) : vec_(v) {}

  BackendVec vec_;
};
} // namespace grex

#endif // INCLUDE_GREX_TYPES_HPP
