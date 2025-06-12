// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_DEFS_HPP
#define INCLUDE_GREX_BACKEND_DEFS_HPP

#include <cstddef>

#include "grex/base/defs.hpp"

namespace grex::backend {
template<Vectorizable T, std::size_t tSize>
struct Vector;
template<Vectorizable T, std::size_t tPart, std::size_t tSize>
struct SubVector {
  using Full = Vector<T, tSize>;
  using Register = Full::Register;
  using Value = T;
  static constexpr std::size_t size = tPart;

  Full full;

  explicit SubVector(Full v) : full{v} {}
  explicit SubVector(Register r) : full{.r = r} {}

  Register registr() const {
    return full.r;
  }
};
template<typename THalf>
struct SuperVector {
  using Value = THalf::Value;
  static constexpr std::size_t size = 2 * THalf::size;

  THalf lower;
  THalf upper;
};
template<Vectorizable T, std::size_t tSize>
using VectorPair = SuperVector<Vector<T, tSize>>;

template<Vectorizable T, std::size_t tSize>
struct Mask;
template<Vectorizable T, std::size_t tPart, std::size_t tSize>
struct SubMask {
  using Full = Mask<T, tSize>;
  using Register = Full::Register;
  static constexpr std::size_t size = tPart;

  Full full;

  explicit SubMask(Full m) : full{m} {}
  explicit SubMask(Register r) : full{.r = r} {}

  Register registr() const {
    return full.r;
  }
};
template<typename THalf>
struct SuperMask {
  static constexpr std::size_t size = 2 * THalf::size;

  THalf lower;
  THalf upper;
};
template<Vectorizable T, std::size_t tSize>
using MaskPair = SuperMask<Mask<T, tSize>>;
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_DEFS_HPP
