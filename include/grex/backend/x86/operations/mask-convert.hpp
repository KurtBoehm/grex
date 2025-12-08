// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_MASK_CONVERT_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_MASK_CONVERT_HPP

#include "grex/backend/x86/instruction-sets.hpp"

// reinterpret-casts between signed integer vectors and masks → only defined for levels below 4
#if GREX_X86_64_LEVEL < 4
#include <cstddef>

#include "grex/backend/base.hpp"
#include "grex/base.hpp"

namespace grex::backend {
// Convert a mask to signed integers
template<Vectorizable T, std::size_t tSize>
inline Vector<SignedInt<sizeof(T)>, tSize> mask2vector(Mask<T, tSize> m) {
  return {.r = m.r};
}

// Convert (signed) integers to a mask
template<SignedIntVectorizable T, std::size_t tSize, Vectorizable TDst>
requires(sizeof(T) == sizeof(TDst))
inline Mask<TDst, tSize> vector2mask(Vector<T, tSize> m, TypeTag<TDst> /*tag*/) {
  return {.r = m.r};
}
} // namespace grex::backend

#include "grex/backend/shared/operations/mask-convert.hpp" // IWYU pragma: export
#endif

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_MASK_CONVERT_HPP
