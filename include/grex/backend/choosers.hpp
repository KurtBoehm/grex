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
template<Vectorizable T, std::size_t N, bool IsSub = (N < min_native_size<T>),
         bool IsSuper = (N > max_native_size<T>)>
struct VectorTrait;
template<Vectorizable T, std::size_t N>
struct VectorTrait<T, N, false, false> {
  using Type = NativeVector<T, N>;
};
template<Vectorizable T, std::size_t N>
struct VectorTrait<T, N, true, false> {
  using Type = SubVector<T, N>;
};
template<Vectorizable T, std::size_t N>
struct VectorTrait<T, N, false, true> {
  using Half = VectorTrait<T, N / 2>::Type;
  using Type = SuperVector<Half>;
};
template<Vectorizable T, std::size_t N>
using VectorFor = VectorTrait<T, N>::Type;

template<Vectorizable T, std::size_t N, bool IsSub = (N < min_native_size<T>),
         bool IsSuper = (N > max_native_size<T>)>
struct MaskTrait;
template<Vectorizable T, std::size_t N>
struct MaskTrait<T, N, false, false> {
  using Type = NativeMask<T, N>;
};
template<Vectorizable T, std::size_t N>
struct MaskTrait<T, N, true, false> {
  using Type = SubMask<T, N>;
};
template<Vectorizable T, std::size_t N>
struct MaskTrait<T, N, false, true> {
  using Half = MaskTrait<T, N / 2>::Type;
  using Type = SuperMask<Half>;
};
template<Vectorizable T, std::size_t N>
using MaskFor = MaskTrait<T, N>::Type;
} // namespace grex::backend
#endif

#endif // INCLUDE_GREX_BACKEND_CHOOSERS_HPP
