// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_NEON_OPERATIONS_SPLIT_HPP
#define INCLUDE_GREX_BACKEND_NEON_OPERATIONS_SPLIT_HPP

#include <cstddef>

#include <arm_neon.h>

#include "grex/backend/choosers.hpp"
#include "grex/backend/defs.hpp" // IWYU pragma: keep
#include "grex/backend/macros/for-each.hpp"
#include "grex/backend/macros/math.hpp"
#include "grex/backend/neon/macros/types.hpp"
#include "grex/backend/neon/types.hpp" // IWYU pragma: keep
#include "grex/base.hpp"

namespace grex::backend {
// native vectors
#define GREX_SPLIT(KIND, BITS, PART, SIZE) \
  inline VectorFor<KIND##BITS, PART> get_low(VectorFor<KIND##BITS, GREX_MULTIPLY(PART, 2)> v) { \
    return VectorFor<KIND##BITS, PART>{v.registr()}; \
  } \
  inline VectorFor<KIND##BITS, PART> get_high(VectorFor<KIND##BITS, GREX_MULTIPLY(PART, 2)> v) { \
    const auto r = v.registr(); \
    const auto out = GREX_ISUFFIXED(vextq, KIND, BITS)(r, r, PART); \
    return VectorFor<KIND##BITS, PART>{out}; \
  }
GREX_FOREACH_SUB(GREX_SPLIT)

// super-native vectors
template<typename THalf>
inline THalf get_low(SuperVector<THalf> v) {
  return v.lower;
}
template<typename THalf>
inline THalf get_high(SuperVector<THalf> v) {
  return v.upper;
}

// native masks
template<Vectorizable T, std::size_t tSize>
inline MaskFor<T, tSize / 2> get_low(Mask<T, tSize> m) {
  return MaskFor<T, tSize / 2>{get_low(Vector<UnsignedInt<sizeof(T)>, tSize>{m.r}).registr()};
}
template<Vectorizable T, std::size_t tSize>
inline MaskFor<T, tSize / 2> get_high(Mask<T, tSize> m) {
  return MaskFor<T, tSize / 2>{get_high(Vector<UnsignedInt<sizeof(T)>, tSize>{m.r}).registr()};
}

// sub-native masks
template<Vectorizable T, std::size_t tPart, std::size_t tSize>
inline MaskFor<T, tPart / 2> get_low(SubMask<T, tPart, tSize> m) {
  return MaskFor<T, tPart / 2>{
    get_low(SubVector<UnsignedInt<sizeof(T)>, tPart, tSize>{m.registr()}).registr()};
}
template<Vectorizable T, std::size_t tPart, std::size_t tSize>
inline MaskFor<T, tPart / 2> get_high(SubMask<T, tPart, tSize> m) {
  return MaskFor<T, tPart / 2>{
    get_high(SubVector<UnsignedInt<sizeof(T)>, tPart, tSize>{m.registr()}).registr()};
}

// super-native masks
template<typename THalf>
inline THalf get_low(SuperMask<THalf> v) {
  return v.lower;
}
template<typename THalf>
inline THalf get_high(SuperMask<THalf> v) {
  return v.upper;
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_SPLIT_HPP
