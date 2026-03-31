// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_NEON_OPERATIONS_ARITHMETIC_MASK_HPP
#define INCLUDE_GREX_BACKEND_NEON_OPERATIONS_ARITHMETIC_MASK_HPP

#include <cstddef>

#include "grex/backend/base.hpp"
#include "grex/backend/defs.hpp" // IWYU pragma: keep
#include "grex/backend/neon/operations/arithmetic.hpp"
#include "grex/backend/neon/operations/blend.hpp"
#include "grex/backend/neon/types.hpp" // IWYU pragma: keep
#include "grex/base.hpp"

namespace grex::backend {
template<Vectorizable T, std::size_t tSize>
inline NativeVector<T, tSize> mask_add(NativeMask<T, tSize> m, NativeVector<T, tSize> a,
                                       NativeVector<T, tSize> b) {
  return blend(m, a, add(a, b));
}
template<Vectorizable T, std::size_t tSize>
inline NativeVector<T, tSize> mask_subtract(NativeMask<T, tSize> m, NativeVector<T, tSize> a,
                                            NativeVector<T, tSize> b) {
  return blend(m, a, subtract(a, b));
}
template<Vectorizable T, std::size_t tSize>
inline NativeVector<T, tSize> mask_multiply(NativeMask<T, tSize> m, NativeVector<T, tSize> a,
                                            NativeVector<T, tSize> b) {
  return blend(m, a, multiply(a, b));
}
template<FloatVectorizable T, std::size_t tSize>
inline NativeVector<T, tSize> mask_divide(NativeMask<T, tSize> m, NativeVector<T, tSize> a,
                                          NativeVector<T, tSize> b) {
  return blend(m, a, divide(a, b));
}
} // namespace grex::backend

#include "grex/backend/shared/operations/arithmetic-mask.hpp" // IWYU pragma: export

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_ARITHMETIC_MASK_HPP
