// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_GLUE_CHOOSERS_HPP
#define INCLUDE_GREX_GLUE_CHOOSERS_HPP

#include <cstddef>

#include "grex/backend.hpp"
#include "grex/base/defs.hpp"

namespace grex::glue {
template<Vectorizable T, std::size_t tSize,
         bool tIsSub = (tSize < backend::native_sizes<T>.front()),
         bool tIsSuper = (tSize > backend::native_sizes<T>.back())>
struct VectorTrait;
template<Vectorizable T, std::size_t tSize>
struct VectorTrait<T, tSize, false, false> {
  using Type = backend::Vector<T, tSize>;
};
template<Vectorizable T, std::size_t tSize>
struct VectorTrait<T, tSize, true, false> {
  using Type = backend::SubVector<T, tSize, backend::native_sizes<T>.front()>;
};
template<Vectorizable T, std::size_t tSize>
struct VectorTrait<T, tSize, false, true> {
  using Half = VectorTrait<T, tSize / 2>::Type;
  using Type = backend::SuperVector<Half>;
};
template<Vectorizable T, std::size_t tSize>
using Vector = VectorTrait<T, tSize>::Type;

template<Vectorizable T, std::size_t tSize,
         bool tIsSub = (tSize < backend::native_sizes<T>.front()),
         bool tIsSuper = (tSize > backend::native_sizes<T>.back())>
struct MaskTrait;
template<Vectorizable T, std::size_t tSize>
struct MaskTrait<T, tSize, false, false> {
  using Type = backend::Mask<T, tSize>;
};
template<Vectorizable T, std::size_t tSize>
struct MaskTrait<T, tSize, true, false> {
  using Type = backend::SubMask<T, tSize, backend::native_sizes<T>.front()>;
};
template<Vectorizable T, std::size_t tSize>
struct MaskTrait<T, tSize, false, true> {
  using Half = MaskTrait<T, tSize / 2>::Type;
  using Type = backend::SuperMask<Half>;
};
template<Vectorizable T, std::size_t tSize>
using Mask = MaskTrait<T, tSize>::Type;
} // namespace grex::glue

#endif // INCLUDE_GREX_GLUE_CHOOSERS_HPP
