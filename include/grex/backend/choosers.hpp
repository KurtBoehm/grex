// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_CHOOSERS_HPP
#define INCLUDE_GREX_BACKEND_CHOOSERS_HPP

#include "grex/backend/defs.hpp" // IWYU pragma: keep

#if !GREX_BACKEND_SCALAR
#include <cstddef>

#include "grex/backend/active/sizes.hpp"
#include "grex/backend/base.hpp"
#include "grex/base.hpp"

namespace grex::backend {
template<Vectorizable T, std::size_t tSize, bool tIsSub = (tSize < min_native_size<T>),
         bool tIsSuper = (tSize > max_native_size<T>)>
struct VectorTrait;
template<Vectorizable T, std::size_t tSize>
struct VectorTrait<T, tSize, false, false> {
  using Type = NativeVector<T, tSize>;
};
template<Vectorizable T, std::size_t tSize>
struct VectorTrait<T, tSize, true, false> {
  using Type = SubVector<T, tSize, min_native_size<T>>;
};
template<Vectorizable T, std::size_t tSize>
struct VectorTrait<T, tSize, false, true> {
  using Half = VectorTrait<T, tSize / 2>::Type;
  using Type = SuperVector<Half>;
};
template<Vectorizable T, std::size_t tSize>
using VectorFor = VectorTrait<T, tSize>::Type;

template<Vectorizable T, std::size_t tSize, bool tIsSub = (tSize < min_native_size<T>),
         bool tIsSuper = (tSize > max_native_size<T>)>
struct MaskTrait;
template<Vectorizable T, std::size_t tSize>
struct MaskTrait<T, tSize, false, false> {
  using Type = NativeMask<T, tSize>;
};
template<Vectorizable T, std::size_t tSize>
struct MaskTrait<T, tSize, true, false> {
  using Type = SubMask<T, tSize, min_native_size<T>>;
};
template<Vectorizable T, std::size_t tSize>
struct MaskTrait<T, tSize, false, true> {
  using Half = MaskTrait<T, tSize / 2>::Type;
  using Type = SuperMask<Half>;
};
template<Vectorizable T, std::size_t tSize>
using MaskFor = MaskTrait<T, tSize>::Type;
} // namespace grex::backend
#endif

#endif // INCLUDE_GREX_BACKEND_CHOOSERS_HPP
