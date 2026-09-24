// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_FMADD_FAMILY_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_FMADD_FAMILY_HPP

#include <concepts>
#include <cstddef>

#include <immintrin.h>

#include "grex/backend/base.hpp"
#include "grex/backend/defs.hpp"
#include "grex/backend/macros/for-each.hpp"
#include "grex/backend/x86/instruction-sets.hpp"
#include "grex/backend/x86/macros/for-each.hpp"
#include "grex/backend/x86/operations/f16.hpp"
#include "grex/backend/x86/types.hpp"
#include "grex/base.hpp"

#if GREX_X86_64_LEVEL >= 3
#include "grex/backend/macros/base.hpp"
#include "grex/backend/x86/macros/intrinsics.hpp"
#include "grex/backend/x86/operations/expand.hpp"
#else
#include "grex/backend/x86/operations/arithmetic.hpp"
#endif

namespace grex::backend {
#if GREX_X86_64_LEVEL >= 3
#define GREX_FMADDF_CALL(NAME, KIND, BITS, BITPREFIX) \
  {.r = to_stored<KIND##BITS>(GREX_CAT(BITPREFIX##_##NAME##_, GREX_EPI_SUFFIX(KIND, BITS))( \
     from_stored<KIND##BITS>(a.r), from_stored<KIND##BITS>(b.r), from_stored<KIND##BITS>(c.r)))}
#define GREX_FMADDS_CALL(NAME, KIND, BITS, SIZE) \
  const auto va = from_stored<KIND##BITS>(expand_any(a, index_tag<SIZE>).r); \
  const auto vb = from_stored<KIND##BITS>(expand_any(b, index_tag<SIZE>).r); \
  const auto vc = from_stored<KIND##BITS>(expand_any(c, index_tag<SIZE>).r); \
  const auto vout = GREX_CAT(_mm_##NAME##_s, GREX_FP_LETTER(BITS))(va, vb, vc); \
  return GREX_CAT(_mm_cvts, GREX_FP_LETTER(BITS), _, GREX_CVTS_VALSUFFIX(BITS))(vout);

inline constexpr bool has_fma = true;
#else
#define GREX_FMADDF_CALL_fmadd add(multiply(a, b), c)
#define GREX_FMADDF_CALL_fmsub subtract(multiply(a, b), c)
#define GREX_FMADDF_CALL_fnmadd subtract(c, multiply(a, b))
#define GREX_FMADDF_CALL_fnmsub subtract(negate(multiply(a, b)), c)
#define GREX_FMADDF_CALL(NAME, ...) GREX_FMADDF_CALL_##NAME
#define GREX_FMADDS_CALL_fmadd return (a * b) + c;
#define GREX_FMADDS_CALL_fmsub return (a * b) - c;
#define GREX_FMADDS_CALL_fnmadd return c - (a * b);
#define GREX_FMADDS_CALL_fnmsub return -(a * b) - c;
#define GREX_FMADDS_CALL(NAME, ...) GREX_FMADDS_CALL_##NAME

inline constexpr bool has_fma = false;
#endif

#define GREX_FMADDF(KIND, BITS, SIZE, BITPREFIX, NAME, TAG) \
  inline NativeVector<KIND##BITS, SIZE> fused(NativeVector<KIND##BITS, SIZE> a, \
                                              NativeVector<KIND##BITS, SIZE> b, \
                                              NativeVector<KIND##BITS, SIZE> c, TAG) { \
    return GREX_FMADDF_CALL(NAME, KIND, BITS, BITPREFIX); \
  }
#define GREX_FMADDF_ALL(REGISTERBITS, BITPREFIX, NAME, TAG) \
  GREX_FOREACH_FP_TYPE_OPT_EXT(GREX_FMADDF, REGISTERBITS, BITPREFIX, NAME, TAG)

GREX_FOREACH_X86_64_LEVEL(GREX_FMADDF_ALL, fmadd, MultiplyAdd)
GREX_FOREACH_X86_64_LEVEL(GREX_FMADDF_ALL, fmsub, MultiplySubtract)
GREX_FOREACH_X86_64_LEVEL(GREX_FMADDF_ALL, fnmadd, NegatedMultiplyAdd)
GREX_FOREACH_X86_64_LEVEL(GREX_FMADDF_ALL, fnmsub, NegatedMultiplySubtract)

template<typename Half>
GREX_ALWAYS_INLINE inline SuperVector<Half> fused(SuperVector<Half> a, SuperVector<Half> b,
                                                  SuperVector<Half> c, FusedTag auto tag) {
  return {
    .lower = fused(a.lower, b.lower, c.lower, tag),
    .upper = fused(a.upper, b.upper, c.upper, tag),
  };
}
template<NativeFloatVectorizable T, std::size_t N>
GREX_ALWAYS_INLINE inline SubVector<T, N> fused(SubVector<T, N> a, SubVector<T, N> b,
                                                SubVector<T, N> c, FusedTag auto tag) {
  return SubVector<T, N>{fused(a.full, b.full, c.full, tag)};
}

#define GREX_FMADDS(KIND, BITS, SIZE, NAME, TAG) \
  template<std::same_as<KIND##BITS> T> \
  inline T fused(T a, T b, T c, TAG) { \
    GREX_FMADDS_CALL(NAME, KIND, BITS, SIZE) \
  }
GREX_FOREACH_FP_TYPE_OPT_EXT(GREX_FMADDS, 128, fmadd, MultiplyAdd)
GREX_FOREACH_FP_TYPE_OPT_EXT(GREX_FMADDS, 128, fmsub, MultiplySubtract)
GREX_FOREACH_FP_TYPE_OPT_EXT(GREX_FMADDS, 128, fnmadd, NegatedMultiplyAdd)
GREX_FOREACH_FP_TYPE_OPT_EXT(GREX_FMADDS, 128, fnmsub, NegatedMultiplySubtract)

// Binary16 without AVX512-FP16: round-trip through binary32.
#if !GREX_F16_NATIVE_ARITHMETIC
template<Float16Vector Vec>
GREX_ALWAYS_INLINE inline Vec fused(Vec a, Vec b, Vec c, FusedTag auto tag) {
  return f32_to_f16(fused(f16_to_f32(a), f16_to_f32(b), f16_to_f32(c), tag));
}
template<std::same_as<f16> T>
GREX_ALWAYS_INLINE inline T fused(T a, T b, T c, FusedTag auto tag) {
  const auto r = fused(grex::f16_to_f32(a), grex::f16_to_f32(b), grex::f16_to_f32(c), tag);
  return grex::f32_to_f16(r);
}
#endif
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_FMADD_FAMILY_HPP
