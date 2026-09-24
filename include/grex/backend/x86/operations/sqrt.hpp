// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_SQRT_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_SQRT_HPP

#include <concepts>

#include <immintrin.h>

#include "grex/backend/base.hpp"
#include "grex/backend/defs.hpp"
#include "grex/backend/macros/base.hpp"
#include "grex/backend/macros/for-each.hpp"
#include "grex/backend/macros/types.hpp"
#include "grex/backend/x86/macros/for-each.hpp"
#include "grex/backend/x86/macros/intrinsics.hpp"
#include "grex/backend/x86/operations/expand.hpp"
#include "grex/backend/x86/operations/f16.hpp"
#include "grex/backend/x86/types.hpp"
#include "grex/base.hpp"

namespace grex::backend {
#define GREX_SQRT(KIND, BITS, SIZE, BITPREFIX) \
  inline NativeVector<KIND##BITS, SIZE> sqrt(NativeVector<KIND##BITS, SIZE> v) { \
    const auto vr = from_stored<KIND##BITS>(v.r); \
    const auto r = GREX_CAT(BITPREFIX##_sqrt_, GREX_EPI_SUFFIX(KIND, BITS))(vr); \
    return {.r = to_stored<KIND##BITS>(r)}; \
  }
#define GREX_SQRT_ALL(REGISTERBITS, BITPREFIX) \
  GREX_FOREACH_FP_TYPE_OPT_EXT(GREX_SQRT, REGISTERBITS, BITPREFIX)

GREX_FOREACH_X86_64_LEVEL(GREX_SQRT_ALL)
GREX_NNVECTOR_UNARY(sqrt)

// Binary16 without AVX512-FP16: round-trip through binary32, exactly as the Neon
// back-end does. `SubVector`/`SuperVector` are covered by `GREX_NNVECTOR_UNARY` above already.
#if !GREX_F16_NATIVE_ARITHMETIC
template<Float16Vector Vec>
inline Vec sqrt(Vec v) {
  return f32_to_f16(sqrt(f16_to_f32(v)));
}
#endif

// Scalar implementations, using the corresponding scalar instruction directly.
#define GREX_SQRTS_OP_16(V) _mm_sqrt_sh(V, V)
#define GREX_SQRTS_OP_32(V) _mm_sqrt_ss(V)
#define GREX_SQRTS_OP_64(V) _mm_sqrt_sd(V, V)
#define GREX_SQRTS_OP(BITS, V) GREX_SQRTS_OP_##BITS(V)

#define GREX_SQRTS(KIND, BITS, SIZE) \
  template<std::same_as<KIND##BITS> T> \
  inline T sqrt(T v) { \
    const auto vv = from_stored<KIND##BITS>(expand_any(v, index_tag<SIZE>).r); \
    return GREX_CAT(_mm_cvts, GREX_FP_LETTER(BITS), _, \
                    GREX_CVTS_VALSUFFIX(BITS))(GREX_SQRTS_OP(BITS, vv)); \
  }
GREX_FOREACH_FP_TYPE_OPT_EXT(GREX_SQRTS, 128)
#undef GREX_SQRTS_OP_16
#undef GREX_SQRTS_OP_32
#undef GREX_SQRTS_OP_64
#undef GREX_SQRTS_OP
#undef GREX_SQRTS

#if !GREX_F16_NATIVE_ARITHMETIC
inline f16 sqrt(f16 v) {
  return grex::f32_to_f16(sqrt(grex::f16_to_f32(v)));
}
#endif
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_SQRT_HPP
