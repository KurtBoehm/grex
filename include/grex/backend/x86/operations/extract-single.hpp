// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_EXTRACT_SINGLE_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_EXTRACT_SINGLE_HPP

#include <cstddef>

#include <immintrin.h>

#include "grex/backend/base.hpp"
#include "grex/backend/macros/for-each.hpp"
#include "grex/backend/x86/instruction-sets.hpp"
#include "grex/backend/x86/macros.hpp"
#include "grex/backend/x86/types.hpp"
#include "grex/base.hpp"

#if GREX_X86_64_LEVEL >= 3
#include "grex/backend/x86/operations/split.hpp"
#endif

namespace grex::backend {
#define GREX_EXTRINGLE_CVT_f64 _mm_cvtsd_f64(v.r)
#define GREX_EXTRINGLE_CVT_f32 _mm_cvtss_f32(v.r)
#define GREX_EXTRINGLE_CVT_i64(KIND) GREX_KINDCAST_SINGLE(i, KIND, 64, _mm_cvtsi128_si64(v.r))
#define GREX_EXTRINGLE_CVT_i32(KIND) GREX_KINDCAST_SINGLE(i, KIND, 32, _mm_cvtsi128_si32(v.r))
#define GREX_EXTRINGLE_CVT_i16(KIND) KIND##16(_mm_cvtsi128_si32(v.r))
#define GREX_EXTRINGLE_CVT_i8(KIND) KIND##8(_mm_cvtsi128_si32(v.r))

#define GREX_EXTRINGLE_CVT_f(KIND, BITS) GREX_EXTRINGLE_CVT_f##BITS
#define GREX_EXTRINGLE_CVT_i(KIND, BITS) GREX_EXTRINGLE_CVT_i##BITS(KIND)
#define GREX_EXTRINGLE_CVT_u(KIND, BITS) GREX_EXTRINGLE_CVT_i##BITS(KIND)

#define GREX_EXTRINGLE_128(KIND, BITS, SIZE) {.value = GREX_EXTRINGLE_CVT_##KIND(KIND, BITS)}
#define GREX_EXTRINGLE_256(...) extract_single(split(v, index_tag<0>))
#define GREX_EXTRINGLE_512(...) extract_single(split(v, index_tag<0>))

#define GREX_EXTRINGLE(KIND, BITS, SIZE, REGISTERBITS) \
  inline Scalar<KIND##BITS> extract_single(Vector<KIND##BITS, SIZE> v) { \
    return GREX_EXTRINGLE_##REGISTERBITS(KIND, BITS, SIZE); \
  }

#define GREX_EXTRINGLE_ALL(REGISTERBITS, BITPREFIX) \
  GREX_FOREACH_TYPE(GREX_EXTRINGLE, REGISTERBITS, REGISTERBITS)
GREX_FOREACH_X86_64_LEVEL(GREX_EXTRINGLE_ALL)

template<Vectorizable T, std::size_t tPart, std::size_t tSize>
inline Scalar<T> extract_single(SubVector<T, tPart, tSize> v) {
  return extract_single(v.full);
}
template<typename THalf>
inline Scalar<typename THalf::Value> extract_single(SuperVector<THalf> v) {
  return extract_single(v.lower);
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_EXTRACT_SINGLE_HPP
