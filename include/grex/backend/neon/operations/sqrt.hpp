// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_NEON_OPERATIONS_SQRT_HPP
#define INCLUDE_GREX_BACKEND_NEON_OPERATIONS_SQRT_HPP

#include <concepts>

#include <arm_neon.h>

#include "grex/backend/base.hpp"
#include "grex/backend/defs.hpp"
#include "grex/backend/macros/for-each.hpp"
#include "grex/backend/macros/types.hpp"
#include "grex/backend/neon/macros/types.hpp"
#include "grex/backend/neon/operations/f16.hpp"
#include "grex/backend/neon/types.hpp"
#include "grex/base.hpp"
#include "grex/f16.hpp"

// With FP16, native binary16 operation is defined together with the other versions, otherwise the
// round-trip through binary32 is used.

namespace grex::backend {
#define GREX_VSQRT(KIND, BITS, SIZE) \
  inline NativeVector<KIND##BITS, SIZE> sqrt(NativeVector<KIND##BITS, SIZE> v) { \
    const auto ret = GREX_ISUFFIXED(vsqrtq, KIND, BITS)(from_stored<KIND##BITS>(v.r)); \
    return {.r = to_stored<KIND##BITS>(ret)}; \
  }
GREX_FOREACH_FP_TYPE_OPT_EXT(GREX_VSQRT, 128)
GREX_NNVECTOR_UNARY(sqrt)
#undef GREX_VSQRT

#if !GREX_F16_NATIVE_ARITHMETIC
inline f16x8 sqrt(f16x8 v) {
  return f32_to_f16(sqrt(f16_to_f32(v)));
}
#endif

// Scalar implementations.

#if GREX_GCC
#define GREX_FSQRT_GCC(TYPE, RPREFIX) \
  if (__builtin_constant_p(v) == 0) { \
    T r{}; \
    asm("fsqrt %" RPREFIX "0, %" RPREFIX "1" : "=w"(r) : "w"(v)); /*NOLINT(*-no-assembler)*/ \
    return r; \
  }
#else
#define GREX_FSQRT_GCC(TYPE, RPREFIX)
#endif

#define GREX_FSQRT(TYPE, RPREFIX, BSUFFIX) \
  template<std::same_as<TYPE> T> \
  inline T sqrt(T v) { \
    GREX_FSQRT_GCC(TYPE, RPREFIX) \
    return __builtin_sqrt##BSUFFIX(v); \
  }

GREX_FSQRT(f32, "s", f)
GREX_FSQRT(f64, "d", )

// Binary16: Both ways of taking the square root directly require the FP16 extension, and neither
// fails loudly without it: `fsqrt` in its half-precision form assembles to an instruction the
// target does not have, and `__builtin_sqrtf16` turns into a call to `sqrtf16`, which the target
// libm need not provide. The binary32 round trip is therefore used instead, which is still
// correctly rounded. It must follow the binary32 definition above, which it calls.
#if GREX_F16_NATIVE_ARITHMETIC
GREX_FSQRT(f16, "h", f16)
#else
template<std::same_as<f16> T>
inline T sqrt(T v) {
  return grex::f32_to_f16(sqrt(grex::f16_to_f32(v)));
}
#endif
#undef GREX_FSQRT
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_SQRT_HPP
