// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_CHOOSERS_HPP
#define INCLUDE_GREX_BACKEND_CHOOSERS_HPP

#if !GREX_BACKEND_SCALAR
#include <cstddef>

#include "grex/backend/active/sizes.hpp"
#include "grex/backend/defs.hpp"
#include "grex/base/defs.hpp"

namespace grex::backend {
template<Vectorizable T, std::size_t tSize, bool tIsSub = (tSize < backend::min_native_size<T>),
         bool tIsSuper = (tSize > backend::max_native_size<T>)>
struct VectorTrait;
template<Vectorizable T, std::size_t tSize>
struct VectorTrait<T, tSize, false, false> {
  using Type = backend::Vector<T, tSize>;
};
template<Vectorizable T, std::size_t tSize>
struct VectorTrait<T, tSize, true, false> {
  using Type = backend::SubVector<T, tSize, backend::min_native_size<T>>;
};
template<Vectorizable T, std::size_t tSize>
struct VectorTrait<T, tSize, false, true> {
  using Half = VectorTrait<T, tSize / 2>::Type;
  using Type = backend::SuperVector<Half>;
};
template<Vectorizable T, std::size_t tSize>
using VectorFor = VectorTrait<T, tSize>::Type;

template<Vectorizable T, std::size_t tSize, bool tIsSub = (tSize < backend::min_native_size<T>),
         bool tIsSuper = (tSize > backend::max_native_size<T>)>
struct MaskTrait;
template<Vectorizable T, std::size_t tSize>
struct MaskTrait<T, tSize, false, false> {
  using Type = backend::Mask<T, tSize>;
};
template<Vectorizable T, std::size_t tSize>
struct MaskTrait<T, tSize, true, false> {
  using Type = backend::SubMask<T, tSize, backend::min_native_size<T>>;
};
template<Vectorizable T, std::size_t tSize>
struct MaskTrait<T, tSize, false, true> {
  using Half = MaskTrait<T, tSize / 2>::Type;
  using Type = backend::SuperMask<Half>;
};
template<Vectorizable T, std::size_t tSize>
using MaskFor = MaskTrait<T, tSize>::Type;
} // namespace grex::backend
#endif

#endif // INCLUDE_GREX_BACKEND_CHOOSERS_HPP
