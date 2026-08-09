// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_SCALAR_OPERATIONS_HPP
#define INCLUDE_GREX_BACKEND_SCALAR_OPERATIONS_HPP

#include <concepts>

#include "grex/backend/base.hpp"
#include "grex/base.hpp"

namespace grex::backend {
template<FloatVectorizable T>
inline T make_finite(T v) {
  return is_finite(v) ? v : T{};
}

template<std::same_as<f64> T>
inline T sqrt(T v) {
  return __builtin_sqrt(v);
}
template<std::same_as<f32> T>
inline T sqrt(T v) {
  return __builtin_sqrtf(v);
}
// Binary16: `__builtin_sqrtf16` does not apply to the emulated type and, for the native one, turns
// into a call to `sqrtf16`, which the target libm need not provide — and the scalar back-end is
// used precisely on the architectures least likely to have it. The binary32 round trip is therefore
// used unconditionally, which is still correctly rounded.
template<std::same_as<f16> T>
inline T sqrt(T v) {
  return f32_to_f16(__builtin_sqrtf(f16_to_f32(v)));
}

inline constexpr bool has_fma = false;

template<FloatVectorizable T>
inline T fused(T a, T b, T c, MultiplyAdd /*tag*/) {
  return (a * b) + c;
}
template<FloatVectorizable T>
inline T fused(T a, T b, T c, MultiplySubtract /*tag*/) {
  return (a * b) - c;
}
template<FloatVectorizable T>
inline T fused(T a, T b, T c, NegatedMultiplyAdd /*tag*/) {
  return c - (a * b);
}
template<FloatVectorizable T>
inline T fused(T a, T b, T c, NegatedMultiplySubtract /*tag*/) {
  return -(a * b + c);
}

template<UnsignedIntVectorizable T>
inline bool bit_test(T a, T b) {
  return ((a >> b) & 1) != 0;
}

template<IntVectorizable TDst, IntVectorizable TSrc>
inline TDst expand_any(TSrc src) {
  return TDst(src);
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_SCALAR_OPERATIONS_HPP
